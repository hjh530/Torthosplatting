# Torthosplatting

> 室内正射影像自动生成工具 — 手机拍房间，一键出平面图。
>
> Automatic indoor orthophoto generation — capture a room with your phone, get a floor plan in one click.

---

## 这是什么？ / What is this?

用手机或相机围绕房间拍摄一段视频或一组照片，交给 **Torthosplatting**，自动生成一张从正上方俯瞰的房间平面正射图。适用于**室内设计参考、租房看房展示、房产户型记录**。

Shoot a video or photo set walking around a room with your phone. Torthosplatting automatically generates a top-down orthophoto of the room — useful for **interior design, rental property viewing, and real estate floor plans**.

## 效果展示 / Example

| 输入 / Input | 输出 / Output |
|-------------|--------------|
| 手机拍房间视频 → COLMAP 稀疏重建 | 高清正射房间平面图 |

## 工作流程 / Pipeline

```
手机拍照 / Phone photos
    ↓
COLMAP 稀疏重建 / Sparse reconstruction (sparse/0)
    ↓
坐标系自动对齐 / Auto coordinate alignment ──→ 虚拟正射相机生成 / Virtual ortho cameras
    ↓
单目深度估计 / Monocular depth (DepthAnythingV2)
    ↓
3DGS 训练(透视) / Training (perspective) + 正射渲染(正交) / Ortho rendering
    ↓
Difix 扩散修复 / Diffusion restoration
    ↓
SIFT 拼接 / Stitching ──→ 房间正射平面图 / Room orthophoto
```

## 快速开始 / Quick Start

### 环境要求 / Requirements

- Linux, CUDA 11.8, NVIDIA GPU (24GB 推荐)
- Python 3.8, PyTorch 2.0
- Conda

### 安装 / Installation

```bash
git clone https://github.com/yourname/Torthosplatting.git
cd Torthosplatting

# 1. 创建环境 / Create environment
conda env create -f environment.yml
conda activate Torthosplatting

# 2. 下载模型权重 / Download weights
# DepthAnythingV2: https://huggingface.co/spaces/LiheYoung/Depth-Anything-V2
# 放入 checkpoints/depth_anything_v2_vitl.pth

# 3. 编译 CUDA 模块 / Compile CUDA
export CUDA_HOME=/usr/local/cuda-11.8
cd submodules/diff-gaussian-rasterization && pip install -e . && cd ../..
cd submodules/diff-gaussian-rasterization-ortho && pip install -e . && cd ../..
cd submodules/simple-knn && pip install -e . && cd ../..

# 4. 额外依赖 / Extra deps
pip install xformers==0.0.22 --no-deps
pip install triton==2.0.0
```

### 运行 / Run

```bash
./run_pipeline.sh -d <数据名> -g 0
# 示例 / Example:
./run_pipeline.sh -d pinhole1 -g 0
./run_pipeline.sh -d pinhole1,pinhole2 -g 0   # 多个
```

### 输入格式 / Input Format

```
数据目录/
├── sparse/0/          # COLMAP 稀疏重建 (BIN 或 TXT)
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── images/            # 原始照片
```

### 输出 / Output

```
数据目录/ortho_output/
├── virtual_1_r1_c1.png   # 25 张修复后的正射瓦片
├── ...
└── ortho_stitched.jpg     # 最终拼接正射大图
```

## 技术方案 / Technical Design

### 为什么需要两套 CUDA？ / Why Two CUDA Modules?

3DGS 光栅化器内置透视投影的雅可比矩阵。训练时必须用透视投影（与输入照片一致），但最终输出需要正交投影（正射俯瞰）。

直接将透视训练好的模型用正交渲染，数学上不匹配。因此训练和渲染使用两套独立的 CUDA 模块：

| 模块 | 用途 | 投影 | 来源 |
|------|------|------|------|
| `diff-gaussian-rasterization` | 训练 | 透视 | DIFIX 3dgs_accel |
| `diff-gaussian-rasterization-ortho` | 渲染 | 正交 | Tortho |

两套模块的区别仅在于 `computeCov2D` 中的雅可比矩阵：

```
透视 / Perspective:  J = [[f/z,  0,  -f·x/z²], [0,  f/z,  -f·y/z²], [0,0,0]]
正交 / Orthographic: J = [[f,    0,   0      ], [0,  f,    0      ], [0,0,0]]
```

### 坐标系自动对齐 / Auto Coordinate Alignment

`gen_virtual_cams.py` 自动将 COLMAP 任意坐标系转换为"地面水平、墙壁正交"的标准坐标系：

1. **地面拟合** → RANSAC 平面检测 → 法线旋转至 Z 轴
2. **墙面方向** → XY 投影 + Hough 线段 → 旋转对齐 XY 轴
3. **虚拟相机** → 5×5 网格均匀采样 → 高度 = 80% 最大高度 → 方向竖直向下

### Tortho 对 DIFIX 的修改

在 DIFIX 基础上仅修改了 4 个文件以支持正交投影：

```
utils/graphics_utils.py          # getProjectionMatrix() 增加 orthographic 参数
gaussian_renderer_ortho.py       # 正交渲染器（独立文件）
scene/cameras.py                 # Camera 增加 get_full_proj_transform() 方法
forward.cu                       # computeCov2D 增加正交雅可比分支
```

## 项目结构 / Project Structure

```
Torthosplatting/
├── train.py                  # 3DGS 训练
├── render_ortho.py           # 正射渲染
├── restore.py                # Difix 图像修复
├── depth_gen.py              # 深度图生成
├── run_pipeline.sh           # 一键流水线
├── environment.yml           # Conda 环境
│
├── scene/                    # 3DGS 场景模块
├── gaussian_renderer/        # 透视渲染器
├── gaussian_renderer_ortho.py # 正交渲染器
│
├── utils/
│   ├── gen_virtual_cams.py   # 虚拟相机生成 + 坐标系对齐
│   ├── calibrate_depth.py    # 深度尺度校准
│   ├── stitch_ortho.py       # 正射拼接
│   └── gen_dummy_images.py   # 虚拟占位图
│
├── submodules/
│   ├── diff-gaussian-rasterization/        # 训练 CUDA (DIFIX)
│   ├── diff-gaussian-rasterization-ortho/  # 渲染 CUDA (Tortho)
│   └── simple-knn/
│
├── depth_anything_v2/        # DepthAnythingV2
└── checkpoints/              # 模型权重
```

## 致谢 / Acknowledgments

本项目整合了以下优秀工作：

- **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)** — 实时辐射场渲染
- **[2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)** — 深度正则化
- **[DIFIX3D+](https://research.nvidia.com/labs/toronto-ai/difix3d/)** — 扩散模型修复 3D 重建 (CVPR 2025 Oral)
- **[DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2)** — 单目深度估计
- **[COLMAP](https://github.com/colmap/colmap)** — 稀疏重建

## License

学术研究用途。各组件遵循原始项目许可证。Academic use only. Components follow their original licenses.
