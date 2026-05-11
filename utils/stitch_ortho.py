"""
stitch_5x5_grid_feature_matching.py

功能：
 1. 解析 virtual_..._rX_cY.png 文件名。
 2. 第一阶段：按行 (Row) 拼接，生成 5 个横向长条。
 3. 第二阶段：将 5 个长条旋转 90 度后拼接，再转回来 (解决垂直拼接问题)。
 4. 使用 OpenCV SIFT/ORB 特征匹配，无需坐标文件。
"""

import cv2
import os
import re
import numpy as np

# ================= 配置 =================

# GUI 最大显示尺寸（过大的图会按比例缩小显示，点击坐标会自动映射回原图）
# =======================================

def parse_grid_info(filename):
    """ 解析文件名获取行列号 """
    match = re.search(r'_r(\d+)_c(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def stitch_images(img_list, mode="horizontal"):
    """
    调用 OpenCV 拼接器拼接一组图片
    """
    if len(img_list) < 2:
        print("  [跳过] 图片少于2张，无需拼接")
        return img_list[0]

    # 初始化拼接器 (SCANS 模式适合正射/平面)
    stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_SCANS)
    # 稍微降低置信度阈值，防止因为纹理少而拼接失败
    stitcher.setPanoConfidenceThresh(0.99) 

    status, pano = stitcher.stitch(img_list)

    if status != cv2.Stitcher_OK:
        print(f"  [失败] 拼接错误代码: {status}")
        return None
    return pano

def rotate_image(image, angle):
    """ 旋转图片 (90度倍数) """
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


    """根据窗口上限计算显示缩放比例（不放大，只缩小）。"""
    h, w = image.shape[:2]
    scale_w = max_w / float(max(w, 1))
    scale_h = max_h / float(max(h, 1))
    return min(1.0, scale_w, scale_h)


def order_quad_points(pts):
    """
    将四边形点排序为 [tl, tr, br, bl]，便于透视变换。
    pts: (4,2)
    """
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def main(image_folder, output_path):
    print(f"正在读取图片: {image_folder} ...")
    
    # 1. 加载并整理图片到 5x5 网格结构
    grid_map = {}
    files = os.listdir(image_folder)
    for f in files:
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        row, col = parse_grid_info(f)
        if row is None: continue
        path = os.path.join(image_folder, f)
        img = cv2.imread(path)
        if img is None: continue
        if row not in grid_map: grid_map[row] = {}
        grid_map[row][col] = img
    
    sorted_rows = sorted(grid_map.keys())
    print(f"检测到 {len(sorted_rows)} 行数据: {sorted_rows}")
    
    # 2. 第一阶段：逐行拼接
    row_strips = []
    print("\n=== 第一阶段：行拼接 (Horizontal) ===")
    for r in sorted_rows:
        print(f"正在拼接第 {r} 行...")
        cols = sorted(grid_map[r].keys())
        row_imgs = [grid_map[r][c] for c in cols]
        strip = stitch_images(row_imgs)
        if strip is not None:
            row_strips.append(strip)
        else:
            print(f"警告：第 {r} 行拼接失败，最终结果可能会缺失这一行！")
    
    if not row_strips:
        print("错误：所有行都拼接失败。")
        return
    
    # 3. 第二阶段：纵向拼接
    print("\n=== 第二阶段：列拼接 (Vertical) ===")
    print("正在旋转长条以适配算法...")
    rotated_strips = [rotate_image(s, 90) for s in row_strips]
    print("正在合并所有行...")
    final_vertical_pano = stitch_images(rotated_strips)
    if final_vertical_pano is not None:
        print("正在恢复方向...")
        final_result = rotate_image(final_vertical_pano, -90)
        cv2.imwrite(output_path, final_result)
        print(f"\n[成功] 最终正射影像已保存: {output_path}")
    else:
        print("\n[失败] 无法合并行长条。可能是重叠不够或特征点不足。")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="拼接 25 张虚拟视图为正射大图 (SIFT 特征匹配)")
    parser.add_argument("--input", "-i", required=True, help="输入 todm 图像目录")
    parser.add_argument("--output", "-o", required=True, help="输出拼接图路径")
    parser.add_argument("--crop", action="store_true", help="启用交互式 GUI 裁剪")
    args = parser.parse_args()

    main(args.input, args.output)



