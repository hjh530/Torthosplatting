"""
为 virtual 虚拟相机生成占位图像（灰色，非黑色避免深度模型异常）。
用法: python create_dummy_images_from_intrinsics.py --sparse_dir sparse/1 --images_dir images
"""
import os
import sys
import argparse
from PIL import Image


def read_camera_records(cameras_path):
    headers, records = [], []
    if not os.path.exists(cameras_path):
        return headers, records
    with open(cameras_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n').strip()
            if not line: continue
            if line.startswith('#'):
                headers.append(raw_line.rstrip('\n'))
                continue
            parts = line.split()
            try:
                records.append({
                    'camera_id': int(parts[0]),
                    'model': parts[1],
                    'width': int(parts[2]),
                    'height': int(parts[3]),
                    'params': parts[4:],
                })
            except Exception:
                pass
    return headers, records


def main():
    parser = argparse.ArgumentParser(description="为虚拟相机构建占位图像")
    parser.add_argument("--sparse_dir", required=True, help="包含 images.txt, cameras.txt 的 sparse 目录")
    parser.add_argument("--images_dir", required=True, help="图像输出目录")
    args = parser.parse_args()

    images_txt = os.path.join(args.sparse_dir, "images.txt")
    cameras_txt = os.path.join(args.sparse_dir, "cameras.txt")
    images_dir = args.images_dir

    if not os.path.exists(images_txt):
        print(f"错误：找不到 {images_txt}")
        return

    _, records = read_camera_records(cameras_txt)
    if not records:
        print("无相机记录")
        return

    width, height = records[0]['width'], records[0]['height']
    count = 0

    with open(images_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) < 2:
                continue
            parts = line.split()
            if len(parts) >= 10:
                name = parts[9]
                if name.startswith('virtual'):
                    target = os.path.join(images_dir, name)
                    if not os.path.exists(target):
                        # 灰色占位图，避免深度模型把全黑当异常
                        img = Image.new('RGB', (width, height), (128, 128, 128))
                        img.save(target)
                        count += 1

    print(f"生成 {count} 张虚拟占位图。")


if __name__ == '__main__':
    main()
