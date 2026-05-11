"""
Difix 图像修复：对正射渲染图进行单步扩散模型去退化。
"""
import os
import torch
import gc
from tqdm import tqdm
from argparse import ArgumentParser
from diffusers.utils import load_image
from pipeline_difix import DifixPipeline


def main():
    parser = ArgumentParser(description="Difix image restoration")
    parser.add_argument("--input", "-i", required=True, help="输入渲染图目录")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--prompt", default="remove degradation", help="修复提示词")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    print("Loading DifixPipeline...")
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")
    pipe.enable_vae_slicing()

    valid = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    target = ("virtual1_", "virtual2_", "virtual3_", "virtual_")
    files = sorted([f for f in os.listdir(args.input)
                    if f.lower().endswith(valid) and f.startswith(target)])

    print(f"Found {len(files)} images to restore.")
    processed = 0

    for fname in tqdm(files, desc="Restoring"):
        in_path = os.path.join(args.input, fname)
        out_path = os.path.join(args.output, fname)
        try:
            img = load_image(in_path)
            with torch.inference_mode():
                result = pipe(args.prompt, image=img,
                              num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
            result.save(out_path)
            processed += 1
        except Exception as e:
            print(f"\nError {fname}: {e}")
        del img, result
        torch.cuda.empty_cache()
        gc.collect()

    print(f"All done! Processed {processed} images.")


if __name__ == "__main__":
    main()
