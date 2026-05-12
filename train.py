# ====================================================================
# FINAL, GPU-ACCELERATED VERSION - USING pyiqa
# ====================================================================
# ====================================================================
# FINAL, WORKING, AND CORRECTLY INDENTED VERSION - USING pyiqa
# ====================================================================
import os
import sys
import uuid
import torch
import numpy as np
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state, get_expon_lr_func

# ========================= Optional Imports =========================
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False



def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID') or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[:10])
    os.makedirs(args.model_path, exist_ok=True)
    print(f"Output folder: {args.model_path}")
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss_func, elapsed,
                    testing_iterations, train_cameras, test_cameras,
                    renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    if iteration not in testing_iterations:
        return

    torch.cuda.empty_cache()
    validation_configs = (
        {"name": "test",  "cameras": test_cameras},
        {"name": "train", "cameras": train_cameras[5:30:5] if len(train_cameras) > 0 else []},
    )
    for config in validation_configs:
        cams = config["cameras"]
        if not cams:
            continue
        l1_test, psnr_test, ssim_test, lpips_test, valid_count = 0.0, 0.0, 0.0, 0.0, 0
        for idx, viewpoint in enumerate(cams):
            if viewpoint.image_name.startswith("virtual"):
                continue
            image = torch.clamp(renderFunc(viewpoint, *renderArgs)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            if train_test_exp:
                image = image[..., image.shape[-1] // 2:]
                gt_image = gt_image[..., gt_image.shape[-1] // 2:]
            if tb_writer and idx < 5:
                tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
            l1_test += l1_loss_func(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips(image, gt_image, net_type="vgg").mean().double()
            valid_count += 1
        
        if valid_count == 0:
            print(f"[ITER {iteration}] Evaluating {config['name']}: skipped (no valid GT views)")
            continue

        l1_test /= valid_count
        psnr_test /= valid_count
        ssim_test /= valid_count
        lpips_test /= valid_count
        print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.6f} PSNR {psnr_test:.3f} SSIM {ssim_test:.4f} LPIPS {lpips_test:.4f}")
        if tb_writer:
            tb_writer.add_scalar(f"{config['name']}/l1", l1_test, iteration)
            tb_writer.add_scalar(f"{config['name']}/psnr", psnr_test, iteration)
            tb_writer.add_scalar(f"{config['name']}/ssim", ssim_test, iteration)
            tb_writer.add_scalar(f"{config['name']}/lpips", lpips_test, iteration)
    torch.cuda.empty_cache()


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from,
             ):


    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("Sparse Adam not available. Install 3dgs_accel.")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    original_train_cameras = scene.getTrainCameras()
    original_test_cameras = scene.getTestCameras()
    filtered_train = []
    
    for cam in original_train_cameras:
        if not cam.image_name.startswith("virtual") :
            filtered_train.append(cam)
            
    filtered_train = list(dict.fromkeys(filtered_train))
    filtered_test = [cam for cam in original_test_cameras if not cam.image_name.startswith("virtual")]
    print(f"[INFO] Final training cameras: {len(filtered_train)}, Testing cameras: {len(filtered_test)}")
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    viewpoint_stack = filtered_train.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log, ema_Ll1depth_for_log = 0.0, 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, 0.0, 1.0)*255).byte().permute(1,2,0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None
        
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        if not viewpoint_stack:
            viewpoint_stack = filtered_train.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        
        rand_idx = randint(0, len(viewpoint_indices)-1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)
        
        if (iteration-1) == debug_from:
            pipe.debug = True
        
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        if viewpoint_cam.alpha_mask is not None:
            image *= viewpoint_cam.alpha_mask.cuda()
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)) if FUSED_SSIM_AVAILABLE else ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        Ll1depth = 0.0
        depth_l1_weight_func = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
        
        if depth_l1_weight_func(iteration) > 0 and viewpoint_cam.depth_reliable:
            Ll1depth_pure = torch.abs((render_pkg["depth"] - viewpoint_cam.invdepthmap.cuda()) * viewpoint_cam.depth_mask.cuda()).mean()
            Ll1depth = depth_l1_weight_func(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", "Depth Loss": f"{ema_Ll1depth_for_log:.7f}", "Points": f"{len(gaussians.get_xyz)}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_end.elapsed_time(iter_start), testing_iterations, filtered_train, filtered_test, render, (gaussians, pipe, bg, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
            
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
            
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    # --- FINAL, WORKING ARGUMENTS ---

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print(f"Optimizing {args.model_path}")
    safe_state(args.quiet)
    
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
        
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from,
    )

    print("\nTraining complete.")










