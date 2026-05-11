# Torthosplatting

正射影像生成与修复统一流水线。将 3D Gaussian Splatting 训练、正射渲染、Difix 修复整合为单一项目，一键生成高质量正射拼接图。

## 致谢

本项目整合了以下优秀工作：

- **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)** — 实时辐射场渲染
- **[2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)** — 深度正则化训练
- **[DIFIX3D+](https://research.nvidia.com/labs/toronto-ai/difix3d/)** — 单步扩散模型修复 3D 重建 (CVPR 2025 Oral)
- **[DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2)** — 单目深度估计

## 项目结构

```
├── depth_gen.py              # 深度图生成（DepthAnythingV2）
├── train.py                  # 3DGS 训练（透视投影，SparseAdam）
├── render_ortho.py           # 正射渲染（正交投影，Tortho CUDA）
├── restore.py                # Difix 图像修复
├── pipeline_difix.py         # Difix 扩散模型 Pipeline
├── gaussian_renderer_ortho.py # 正交渲染器（调用 Tortho CUDA）
├── run_pipeline.sh           # 一键流水线脚本
├── environment.yml           # Conda 环境配置
│
├── scene/                    # 3DGS 场景模块
│   ├── cameras.py            # 相机模型（含正交投影支持）
│   ├── dataset_readers.py    # 数据集读取（自动检测 sparse/1）
│   └── gaussian_model.py     # 高斯模型
│
├── gaussian_renderer/        # 透视渲染器（训练用，DIFIX CUDA）
│   └── __init__.py
│
├── utils/
│   ├── graphics_utils.py     # 投影矩阵（含正交投影矩阵）
│   ├── gen_virtual_cams.py   # 虚拟正射相机生成（BIN+TXT双格式）
│   ├── calibrate_depth.py    # 深度尺度校准
│   ├── stitch_ortho.py       # 正射影像拼接
│   └── gen_dummy_images.py   # 虚拟占位图
│
├── submodules/
│   ├── diff-gaussian-rasterization/       # DIFIX CUDA（透视，训练用）
│   ├── diff-gaussian-rasterization-ortho/ # Tortho CUDA（正交，渲染用）
│   └── simple-knn/                        # KNN 加速
│
├── depth_anything_v2/        # DepthAnythingV2 模型
└── checkpoints/              # 模型权重
```

## 安装

### 1. 环境要求

- Linux, CUDA 11.8, NVIDIA GPU (RTX 4090 推荐)
- Conda 或 Miniconda

### 2. 创建环境

```bash
cd Torthosplatting
conda env create -f environment.yml
conda activate Torthosplatting
```

### 3. 下载 DepthAnythingV2 权重

```bash
# 下载到 checkpoints/ 目录
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything-V2/resolve/main/checkpoints/depth_anything_v2_vitl.pth \
     -O checkpoints/depth_anything_v2_vitl.pth
```

### 4. 编译 CUDA 模块

```bash
conda activate Torthosplatting
export CUDA_HOME=/usr/local/cuda-11.8

# 训练用（DIFIX 3dgs_accel，透视投影 + SparseAdam）
cd submodules/diff-gaussian-rasterization
pip install -e .

# 渲染用（Tortho，正交投影）
cd ../diff-gaussian-rasterization-ortho
pip install -e .

# KNN 加速
cd ../simple-knn
pip install -e .
```

### 5. 安装 Difix 依赖

```bash
conda activate Torthosplatting
pip install xformers==0.0.22 --no-deps
pip install triton==2.0.0
```

## 使用

```bash
# 单个数据
./run_pipeline.sh -d pinhole1 -g 0

# 多个数据（顺序执行）
./run_pipeline.sh -d pinhole1,pinhole2 -g 0

# 绝对路径
./run_pipeline.sh -d /path/to/data -g 0
```

输入数据结构（仅需 `sparse/0` 和 `images/`）：

```
data/
├── sparse/0/     # COLMAP 输出（BIN 或 TXT）
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── images/       # 原始图像
```

## 流水线步骤

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 0 | `gen_virtual_cams.py` | COLMAP 坐标系转换 + 生成虚拟正射相机 → `sparse/1` |
| 0 | `gen_dummy_images.py` | 为虚拟相机生成灰色占位图 |
| 1 | `depth_gen.py` | DepthAnythingV2 生成单目深度图 |
| 2 | `calibrate_depth.py` | 对齐 COLMAP 深度与单目深度尺度 |
| 2 | `train.py` | 3DGS 训练（透视投影，SparseAdam） |
| 3 | `render_ortho.py` | 正射渲染 25 张虚拟视图（正交投影） |
| 4 | `restore.py` | Difix 单步扩散模型修复 |
| 5 | `stitch_ortho.py` | SIFT 特征匹配拼接为正射大图 |

## 正射投影原理

### 核心挑战

3DGS 默认使用透视投影渲染。训练时所有视角均为透视（平视/俯视），但最终需要从正上方（正射）渲染。这导致两个问题：

1. **投影矩阵不同**：透视投影 `x_proj = f * x / z`，正射投影 `x_proj = f * x`
2. **雅可比矩阵不同**：CUDA 光栅化器需要正确的 2D 协方差投影

### 解决方案：双 CUDA 模块

| CUDA 模块 | 用途 | 投影 | 雅可比 |
|-----------|------|------|--------|
| `diff-gaussian-rasterization` (DIFIX) | 训练 | 透视 | `J = [[f/z, 0, -f*x/z²], [0, f/z, -f*y/z²], [0,0,0]]` |
| `diff-gaussian-rasterization-ortho` (Tortho) | 渲染 | 正交 | `J = [[f, 0, 0], [0, f, 0], [0, 0, 0]]` |

### 修改的文件

为支持正交投影，Tortho 在 DIFIX 基础上修改了 4 个文件：

- `utils/graphics_utils.py` — `getProjectionMatrix()` 增加 `orthographic` 参数
- `gaussian_renderer_ortho.py` — 正交渲染器（调用 Tortho CUDA）
- `scene/cameras.py` — `Camera` 类增加 `get_full_proj_transform()`
- `forward.cu` — `computeCov2D` 中正交雅可比分支

## 坐标系转换与虚拟位姿生成

### `gen_virtual_cams.py` 工作流程

1. **读取 COLMAP 模型**：自动检测 BIN/TXT 格式，保留完整 2D-3D 追踪数据

2. **地面估计**：RANSAC 拟合最大平面，自动判定地面/天花板

3. **地面法线对齐**：将地面法线旋转至世界 Z 轴 `[0, 0, 1]`

4. **墙面方向检测**：将点云投影至 XY 平面，Hough 线段检测墙面主方向，旋转对齐

5. **虚拟相机生成**：在 XY 包围盒内均匀采样 5×5=25 个虚拟相机，高度为最高点的 80%，方向竖直向下

6. **输出**：`sparse/1` (TXT + BIN 双格式) + `sparse/1/xoy_projection.png`

### 坐标系变换

```
原始 COLMAP 坐标系
    ↓ RANSAC 地面法线 → A1 (地面 → Z)
    ↓ Hough 墙面方向 → A2 (墙面 → XY 轴)
    ↓
最终坐标系：Z=高度, XY 轴对齐墙壁
```

## 环境依赖

<details>
<summary>environment.yml</summary>

```yaml
name: Torthosplatting
channels: [pytorch, nvidia, conda-forge, defaults]
dependencies:
  - python=3.8.18
  - pytorch=2.0.0
  - cudatoolkit=11.8
  - torchvision=0.15.0
  - plyfile, opencv, scipy, matplotlib, tqdm, joblib
  - diffusers==0.25.1, transformers==4.38.0, huggingface-hub==0.25.1
  - xformers==0.0.22 (--no-deps)
```
</details>

## License

本项目仅用于学术研究。各组件遵循其原始项目的许可证。
