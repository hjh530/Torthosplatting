"""
正射渲染：加载训练好的 3DGS 模型，用正交投影渲染所有 virtual_ 虚拟相机。
使用 Tortho CUDA（diff_gaussian_rasterization_ortho）。
"""
import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser
from scene import Scene
from gaussian_renderer_ortho import render_ortho as render
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state


def render_sets(dataset, iteration, pipeline):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        all_cameras = scene.getTrainCameras() + scene.getTestCameras()
        render_targets = [cam for cam in all_cameras if cam.image_name.startswith("virtual_")]
        render_targets.sort(key=lambda x: x.image_name)

        if not render_targets:
            print("[Warning] No virtual_ cameras found.")
            return

        print(f"Found {len(render_targets)} virtual cameras to render.")

        # 释放非目标相机显存
        target_ids = set(id(cam) for cam in render_targets)
        for cam in all_cameras:
            if id(cam) not in target_ids and hasattr(cam, 'original_image'):
                cam.original_image = None
        torch.cuda.empty_cache()

        # 渲染
        render_path = os.path.join(dataset.model_path, "virtual_views", f"ours_{iteration}", "renders")
        makedirs(render_path, exist_ok=True)

        for view in tqdm(render_targets, desc="Rendering"):
            result = render(view, gaussians, pipeline, background, use_trained_exp=dataset.train_test_exp)
            rendering = result["render"]

            save_name = view.image_name
            if not save_name.endswith(".png"):
                save_name = os.path.splitext(save_name)[0] + ".png"
            torchvision.utils.save_image(rendering, os.path.join(render_path, save_name))

            if hasattr(view, 'original_image'):
                view.original_image = None
            del rendering, result
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Orthographic rendering")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print(f"Rendering {args.model_path}")
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args))
