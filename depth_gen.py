"""
Minimal depth generation using DepthAnythingV2.
No 3DGS dependencies — only torch, cv2, PIL, numpy.
"""
import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser


def save_tiff(depthmap, path):
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(path, 'TIFF')


def main():
    parser = ArgumentParser(description="Generate monocular depth maps")
    parser.add_argument("-s", "--source_path", required=True, help="Data directory (contains images/)")
    parser.add_argument("--depth_imagepath", required=True, help="Output depth directory")
    parser.add_argument("--encoder", default="vitl", choices=["vits","vitb","vitl","vitg"])
    args = parser.parse_args()

    images_dir = os.path.join(args.source_path, "images")
    depth_dir = args.depth_imagepath
    os.makedirs(depth_dir, exist_ok=True)

    if not os.path.isdir(images_dir):
        print(f"错误：images 目录不存在: {images_dir}")
        sys.exit(1)

    # 加载 DepthAnythingV2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(**model_configs[args.encoder])
    ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", f"depth_anything_v2_{args.encoder}.pth")
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model = model.to(DEVICE).eval()

    # 遍历所有图像
    existing = set(os.listdir(depth_dir)) if os.path.isdir(depth_dir) else set()
    valid = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid)])

    for fname in tqdm(image_files, desc="Generating depth maps"):
        if fname.startswith("virtual"):
            continue
        out_name = os.path.splitext(fname)[0] + '.tiff'
        if out_name in existing:
            continue
        raw = cv2.imread(os.path.join(images_dir, fname))
        if raw is None:
            continue
        depth = model.infer_image(raw)
        save_tiff(depth, os.path.join(depth_dir, out_name))

    print(f"完成。深度图保存至 {depth_dir}")


if __name__ == "__main__":
    main()
