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

### 1. 从透视到正交：投影矩阵与雅可比 / From Perspective to Orthographic

3DGS 渲染管线中，每个 3D 高斯体投影到屏幕需要两步：**投影矩阵**（Python 层计算）和**协方差雅可比**（CUDA 层计算）。训练和渲染的不同在于：

- **训练**：相机模拟输入照片的透视投影 → 使用 DIFIX CUDA
- **渲染**：相机模拟正射俯视的正交投影 → 使用 Tortho CUDA

#### 投影矩阵 / Projection Matrix

`utils/graphics_utils.py` — `getProjectionMatrix()` 根据 `orthographic` 参数生成不同的投影矩阵：

```python
def getProjectionMatrix(znear, zfar, fovX, fovY, orthographic=False):
    if not orthographic:
        # 透视投影：视锥体近大远小
        # Perspective: frustum — objects shrink with distance
        tanHalfFovY = math.tan(fovY / 2)
        tanHalfFovX = math.tan(fovX / 2)
        top = tanHalfFovY * znear
        right = tanHalfFovX * znear

        P = torch.zeros(4, 4)
        P[0, 0] = 2.0 * znear / (right - left)   # 水平缩放 / horizontal scale
        P[1, 1] = 2.0 * znear / (top - bottom)   # 垂直缩放 / vertical scale
        P[0, 2] = (right + left) / (right - left) # 水平偏移 / horizontal shift
        P[1, 2] = (top + bottom) / (top - bottom) # 垂直偏移 / vertical shift
        P[2, 2] = zfar / (zfar - znear)           # 深度映射 / depth mapping
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        P[3, 2] = 1.0
        return P

    else:
        # 正交投影：无透视缩放，物体大小与距离无关
        # Orthographic: no perspective scaling, object size independent of depth
        tanHalfFovY = math.tan(fovY / 2)
        tanHalfFovX = math.tan(fovX / 2)
        top = 5          # 固定视口高度 / fixed viewport height
        right = tanHalfFovX * 5 / tanHalfFovY  # 保持宽高比 / maintain aspect ratio

        P = torch.zeros(4, 4)
        P[0, 0] = 2.0 / (right - left)       # X 缩放（不含 z）/ scale X (no z)
        P[1, 1] = 2.0 / (top - bottom)       # Y 缩放（不含 z）/ scale Y (no z)
        P[0, 3] = -(right + left) / (right - left)  # X 平移 / translate X
        P[1, 3] = -(top + bottom) / (top - bottom)  # Y 平移 / translate Y
        P[2, 2] = -2.0 / (zfar - znear)      # Z 缩放 / scale Z
        P[2, 3] = -(zfar + znear) / (zfar - znear)  # Z 平移 / translate Z
        P[3, 3] = 1.0
        return (right - left) / 2, (top - bottom) / 2, P  # 返回视口半宽高 / return half dims
```

**关键差异**：透视投影中 `P[0,0]` 和 `P[1,1]` 分母含 `znear`（近平面距离），意味着远处物体会缩小；正交投影中去掉了 `znear`，所有深度的物体等大。

**Key difference**: In perspective, `P[0,0]` and `P[1,1]` scale with `znear`, making distant objects smaller. In orthographic, depth is removed — all objects appear the same size regardless of distance.

#### 相机投影调用 / Camera Projection Invocation

`scene/cameras.py` — `Camera.get_full_proj_transform()` 在渲染时调用正交投影矩阵：

```python
class Camera(nn.Module):
    def __init__(self, ...):
        # 初始化时计算透视投影 / Compute perspective at init
        self.projection_matrix = getProjectionMatrix(
            znear, zfar, FoVx, FoVy, orthographic=False
        ).transpose(0,1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)

    def get_full_proj_transform(self, orthographic=False):
        if not orthographic:
            return self.full_proj_transform  # 训练用 / For training
        else:
            # 渲染时重新计算正交投影 / Recompute ortho at render time
            tanfovx, tanfovy, proj_mat = getProjectionMatrix(
                znear, zfar, FoVx, FoVy, orthographic=True
            )
            full_proj = (
                self.world_view_transform.unsqueeze(0)
                .bmm(proj_mat.transpose(0,1).cuda().unsqueeze(0))
            ).squeeze(0)
            return tanfovx, tanfovy, full_proj  # 渲染用 / For rendering
```

#### CUDA 协方差雅可比 / CUDA Covariance Jacobian

`forward.cu` — `computeCov2D()` 是唯一需要在正交和透视间切换的 CUDA 函数。它将 3D 协方差矩阵投影到 2D 屏幕空间：

```c
__device__ float3 computeCov2D(
    const float3& mean, float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    const float* cov3D, const float* viewmatrix,
    const bool orthographic)
{
    float3 t = transformPoint4x3(mean, viewmatrix);
    glm::mat3 J;

    if (orthographic) {
        // 正交投影：focal 直接作为缩放因子，无透视除法
        // Orthographic: focal is direct scale, no perspective division
        J = glm::mat3(
            focal_x, 0.0f,    0,   // ∂u/∂x, ∂u/∂y, ∂u/∂z
            0.0f,    focal_y, 0,   // ∂v/∂x, ∂v/∂y, ∂v/∂z
            0,       0,       0);
    } else {
        // 透视投影：含 1/z 缩放 + 泰勒展开修正项
        // Perspective: includes 1/z scaling + Taylor expansion correction
        const float limx = 1.3f * tan_fovx;
        const float limy = 1.3f * tan_fovy;
        // 限制屏幕空间位置 / Clamp screen-space position
        const float txtz = t.x / t.z;
        const float tytz = t.y / t.z;
        t.x = min(limx, max(-limx, txtz)) * t.z;
        t.y = min(limy, max(-limy, tytz)) * t.z;

        J = glm::mat3(
            focal_x / t.z,  0.0f,          -(focal_x * t.x) / (t.z * t.z),
            0.0f,           focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
            0,              0,             0);
    }
    // ... 后续协方差变换 / rest of covariance transform
}
```

**直观理解**：透视投影中，高斯体越远（t.z 越大），屏幕投影越小（除以 t.z）。正交投影无此效果，所有距离的高斯体等大投影。

**Intuition**: In perspective, farther Gaussians appear smaller (divided by t.z). Orthographic has no such effect — all Gaussians project at equal size regardless of depth.

### 2. 坐标系转换与虚拟位姿生成 / Coordinate Transform & Virtual Camera Generation

`utils/gen_virtual_cams.py` 将 COLMAP 输出的任意坐标系转换为"地面水平、墙壁正交"的标准坐标系，并生成正射俯瞰所需的虚拟相机。

#### 整体变换流程 / Overall Transform Pipeline

```
COLMAP 原始坐标系 / Original COLMAP coordinates
    ↓
步骤1: RANSAC 拟合地面 → 法线旋转至 Z 轴 (矩阵 A1)
    Step 1: RANSAC ground plane → rotate normal to Z-axis (matrix A1)
    ↓
步骤2: XY 投影 + Hough 线段 → 旋转墙面方向对齐 XY 轴 (矩阵 A2)
    Step 2: XY projection + Hough lines → rotate walls to align with XY axes (matrix A2)
    ↓
最终变换: P_final = A2 · A1 · (P_original - P_ground)
    Final transform: P_final = A2 · A1 · (P_original - P_ground)
    ↓
在 XY 包围盒内 5×5 采样虚拟相机 → 正射渲染
    5×5 grid virtual cameras in XY bbox → ortho rendering
```

#### 步骤1: 地面检测 / Step 1: Ground Detection

```python
# RANSAC 平面拟合 / RANSAC plane fitting
normal, p0, inliers = ransac_plane(pts, iters=2000, thresh=0.02)

# 判定地面 vs 天花板 / Determine ground vs ceiling
A1_temp = rotation_from_vector_to_z(normal)  # 法线→Z轴
pts_temp = (pts - p0) @ A1_temp.T
z_clean = pts_temp[:, 2]
span_low = z50 - z10   # 低处点云跨度 / span of lower points
span_high = z90 - z50  # 高处点云跨度 / span of upper points
is_ceiling = span_low >= span_high  # 低处跨度大 → 天花板
if is_ceiling:
    normal = -normal  # 翻转法线 / flip normal
A1 = rotation_from_vector_to_z(normal)  # 最终地面→Z轴
```

`rotation_from_vector_to_z(n)` 使用 Rodrigues 旋转公式，计算将任意法线向量旋转至 `[0,0,1]` 的旋转矩阵。

#### 步骤2: 墙面方向检测 / Step 2: Wall Direction Detection

```python
def align_xy_by_outer_boundary_grid(points):
    # 1. 点云投影至 XY 平面 / Project to XY plane
    projected_xy = points[:, :2]

    # 2. 密度滤波去除离群点 / Density filter to remove outliers
    projected_xy = density_filter_xy(projected_xy)

    # 3. 外轮廓提取 / Extract outer contour
    boundary_pts = extract_outer_contour_xy(projected_xy)

    # 4. Hough 线段检测 / Hough line detection
    hough_segments = detect_hough_line_segments(projected_xy)

    # 5. 筛选边缘线段 / Filter edge segments
    edge_segs = filter_by_boundary_proximity(hough_segments, boundary_pts)

    # 6. 正交方向估计 / Estimate orthogonal directions
    # 在 0-180° 范围内搜索最佳正交方向对
    best_angle = estimate_from_orthogonal_support(edge_segs)
    # 或从最长的边缘线段簇推断
    if best_angle is None:
        best_angle = estimate_from_outer_dominant_lines(edge_segs)

    # 7. 旋转对齐 / Rotate to align
    A2 = rotation_matrix_z(-np.radians(best_angle))
    return A2
```

#### 步骤3: 虚拟相机生成 / Step 3: Virtual Camera Generation

```python
def generate_render_views_grid(images, bbox, target_height):
    # 5×5 网格中心点 / 5×5 grid centers
    centers = compute_virtual_camera_centers_grid(bbox, target_height)
    # 竖直向下的旋转矩阵 / Downward-facing rotation (q=[0,1,0,0])
    q_ortho = np.array([0.0, 1.0, 0.0, 0.0])
    R_ortho = qvec2rotmat(q_ortho)

    for row in range(1, 6):
        for col in range(1, 6):
            C = centers[count - 1]
            t = -R_ortho @ C  # 平移向量 / translation vector
            merged[img_id] = {
                "qvec": q_ortho,
                "tvec": t,
                "camera_id": ref_camera_id,
                "name": f"virtual_{count}_r{row}_c{col}.png",
            }
    return merged
```

### 3. Tortho 对 DIFIX 的四处修改 / Four Changes from DIFIX to Tortho

| 文件 / File | 修改内容 / Change |
|------------|-----------------|
| `utils/graphics_utils.py` | `getProjectionMatrix()` 增加 `orthographic` 参数，正交分支返回 `(半宽, 半高, 矩阵)` 三元组 |
| `gaussian_renderer_ortho.py` | 新建独立正交渲染器，调用 Tortho CUDA 模块 |
| `scene/cameras.py` | `Camera` 增加 `get_full_proj_transform(orthographic)` 方法，运行时动态选择投影 |
| `forward.cu` | `computeCov2D()` 增加 `orthographic` 参数和正交雅可比分支 |

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
