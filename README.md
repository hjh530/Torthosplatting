# Torthosplatting / 正射高斯

> **一键式 COLMAP → 正射影像拼接流水线，集成深度估计、3DGS 训练、正射渲染与 Difix 扩散修复。**
>
> *One-click pipeline: COLMAP → orthophoto stitching, integrating depth estimation, 3DGS training, orthographic rendering, and Difix diffusion restoration.*

---

## 简介 / Introduction

**Torthosplatting** 是一个面向近景摄影测量的端到端正射影像生成工具。输入 COLMAP 稀疏重建结果（`sparse/0`），自动完成坐标系对齐、虚拟正射相机生成、单目深度估计、3D Gaussian Splatting 训练、正射渲染、Difix 扩散模型修复，最终输出拼接后的正射大图。

核心创新在于将透视投影训练与正交投影渲染解耦为两套独立的 CUDA 光栅化模块，解决了 3DGS 在正射视角下的边缘伪影问题。

**Torthosplatting** is an end-to-end orthophoto generation tool for close-range photogrammetry. Given a COLMAP sparse reconstruction (`sparse/0`), it automatically performs coordinate alignment, virtual ortho-camera generation, monocular depth estimation, 3D Gaussian Splatting training, orthographic rendering, Difix diffusion restoration, and outputs a stitched orthomosaic.

The key innovation is decoupling perspective-projection training from orthographic-projection rendering into two separate CUDA rasterization modules, solving the edge artifact problem of 3DGS under orthographic views.

## 致谢 / Acknowledgments

本项目整合了以下优秀工作 / This project integrates the following outstanding works:

- **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)** — 实时辐射场渲染 / Real-time radiance field rendering
- **[2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)** — 深度正则化训练 / Depth-regularized training
- **[Tortho Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)** — 正交投影渲染 / Orthographic rendering
- **[DIFIX3D+](https://research.nvidia.com/labs/toronto-ai/difix3d/)** — 单步扩散模型修复 3D 重建 (CVPR 2025 Oral)
- **[DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2)** — 单目深度估计

## 项目结构 / Project Structure

```
Torthosplatting/
├── depth_gen.py              # 深度图生成 / Depth generation
├── train.py                  # 3DGS 训练（透视投影）/ Training (perspective)
├── render_ortho.py           # 正射渲染（正交投影）/ Ortho rendering
├── restore.py                # Difix 图像修复 / Diffusion restoration
├── pipeline_difix.py         # Difix 扩散模型 / Difix pipeline
├── gaussian_renderer_ortho.py # 正交渲染器 / Ortho renderer
├── run_pipeline.sh           # 一键流水线 / One-click pipeline
├── environment.yml           # Conda 环境配置 / Environment config
│
├── scene/                    # 3DGS 场景模块 / Scene module
│   ├── cameras.py            # 相机模型（正交投影支持）
│   ├── dataset_readers.py    # 数据集读取（自动检测 sparse/1）
│   └── gaussian_model.py     # 高斯模型
│
├── gaussian_renderer/        # 透视渲染器（训练用）/ Perspective renderer
├── utils/
│   ├── gen_virtual_cams.py   # 虚拟正射相机生成 / Virtual camera generation
│   ├── gen_dummy_images.py   # 虚拟占位图 / Dummy image generation
│   ├── calibrate_depth.py    # 深度尺度校准 / Depth calibration
│   ├── stitch_ortho.py       # 正射影像拼接 / Ortho stitching
│   ├── graphics_utils.py     # 投影矩阵（正交+透视）
│   ├── read_write_model.py   # COLMAP 模型读写
│   └── ...                   # 其他工具 / Other utilities
│
├── submodules/
│   ├── diff-gaussian-rasterization/       # DIFIX CUDA（训练，透视）
│   ├── diff-gaussian-rasterization-ortho/ # Tortho CUDA（渲染，正交）
│   └── simple-knn/                        # KNN 加速
│
├── depth_anything_v2/        # DepthAnythingV2 模型
└── checkpoints/              # 模型权重 / Model weights
```

## 快速开始 / Quick Start

### 1. 环境 / Environment

- **OS**: Linux, **CUDA**: 11.8, **GPU**: NVIDIA RTX 4090 (24GB)
- **Python**: 3.8, **PyTorch**: 2.0

### 2. 安装 / Installation

```bash
git clone https://github.com/yourname/Torthosplatting.git
cd Torthosplatting

# 创建 conda 环境 / Create conda environment
conda env create -f environment.yml
conda activate Torthosplatting

# 下载 DepthAnythingV2 权重 / Download weights
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything-V2/resolve/main/checkpoints/depth_anything_v2_vitl.pth \
     -O checkpoints/depth_anything_v2_vitl.pth

# 编译 CUDA 模块 / Compile CUDA modules
export CUDA_HOME=/usr/local/cuda-11.8

cd submodules/diff-gaussian-rasterization && pip install -e . && cd ../..
cd submodules/diff-gaussian-rasterization-ortho && pip install -e . && cd ../..
cd submodules/simple-knn && pip install -e . && cd ../..

# 安装额外依赖 / Extra dependencies
pip install xformers==0.0.22 --no-deps
pip install triton==2.0.0
```

### 3. 运行 / Run

```bash
# 输入只需 sparse/0 (BIN) + images/
# Input: only sparse/0 (BIN) + images/

./run_pipeline.sh -d pinhole1 -g 0
./run_pipeline.sh -d pinhole1,pinhole2 -g 0  # 多个数据 / Multiple datasets
./run_pipeline.sh -d /absolute/path -g 0       # 绝对路径 / Absolute path
```

**输出 / Output:** `ortho_output/` — 25 张修复图 + 1 张拼接大图 / 25 restored images + 1 stitched orthomosaic

## 流水线 / Pipeline

| 步骤 Step | 脚本 Script | 说明 Description |
|-----------|-------------|------------------|
| 0 | `gen_virtual_cams.py` | COLMAP 坐标系对齐 + 生成虚拟正射相机 → `sparse/1` |
| 0 | `gen_dummy_images.py` | 虚拟相机灰色占位图 / Placeholder images |
| 1 | `depth_gen.py` | DepthAnythingV2 单目深度估计 |
| 2 | `calibrate_depth.py` | COLMAP 深度与单目深度尺度对齐 |
| 2 | `train.py` | 3DGS 透视投影训练（SparseAdam） |
| 3 | `render_ortho.py` | 正交投影渲染 25 张虚拟视图 |
| 4 | `restore.py` | Difix 单步扩散模型修复 |
| 5 | `stitch_ortho.py` | SIFT 特征匹配拼接正射大图 |

已有输出自动跳过对应步骤 / Steps are skipped if output already exists.

## 正射投影原理 / Orthographic Projection

### 问题 / Problem

3DGS 默认使用透视投影。训练视角（平视/俯视）与渲染视角（正射向下）不一致，导致正射渲染时边缘出现伪影。

3DGS uses perspective projection by default. Training views (oblique/looking-down) differ from the rendering view (top-down orthographic), causing edge artifacts.

### 解决方案：双 CUDA 模块 / Solution: Dual CUDA Modules

| 模块 / Module | 用途 / Use | 投影 / Projection | 雅可比矩阵 / Jacobian |
|--------------|-----------|-------------------|----------------------|
| `diff-gaussian-rasterization` (DIFIX) | 训练 / Training | 透视 / Perspective | `J = [[f/z, 0, -f·x/z²], [0, f/z, -f·y/z²], [0,0,0]]` |
| `diff-gaussian-rasterization-ortho` (Tortho) | 渲染 / Rendering | 正交 / Ortho | `J = [[f, 0, 0], [0, f, 0], [0, 0, 0]]` |

两套 CUDA 共享同一 Python API，通过不同包名隔离，无需在运行时切换。

Two CUDA modules share the same Python API but are isolated via different package names, requiring no runtime switching.

### Tortho 的四处修改 / Four Modifications in Tortho

1. `utils/graphics_utils.py` — `getProjectionMatrix()` 增加 `orthographic` 参数
2. `gaussian_renderer_ortho.py` — 正交渲染器（调用 Tortho CUDA）
3. `scene/cameras.py` — `Camera` 类增加 `get_full_proj_transform()` 方法
4. `forward.cu` — `computeCov2D` 中正交雅可比分支

## 坐标系转换 / Coordinate Transformation

### `gen_virtual_cams.py` 工作流程 / Workflow

1. **读取 COLMAP** / Read COLMAP: 自动检测 BIN/TXT，保留 2D-3D 追踪
2. **地面估计** / Ground estimation: RANSAC 拟合最大平面，自动判定地面/天花板
3. **法线对齐** / Normal alignment: 旋转地面法线至世界 Z 轴 `[0,0,1]`
4. **墙面检测** / Wall detection: XY 投影 + Hough 线段提取主方向，旋转对齐 XY 轴
5. **虚拟相机** / Virtual cameras: XY 包围盒 5×5 均匀采样，高度 80%，方向竖直向下
6. **输出** / Output: `sparse/1` (TXT+BIN) + `xoy_projection.png`

```
原始坐标系 / Original COLMAP
    ↓ RANSAC 地面法线 → A1（地面 → Z）
    ↓ Hough 墙面方向 → A2（墙面 → XY 轴）
    ↓
最终坐标系 / Final: Z=高度/height, XY 对齐墙壁/aligned to walls
```

## License

本项目仅用于学术研究。各组件遵循其原始项目的许可证。

*For academic research only. Individual components follow their original licenses.*
