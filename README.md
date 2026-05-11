# Torthosplatting

> Automatic indoor orthophoto generation — capture a room with your phone, get a floor plan in one click.

[中文文档 / Chinese README](README_CN.md)

---

## What is this?

Shoot a video or photo set walking around a room with your phone. **Torthosplatting** automatically generates a top-down orthophoto floor plan — useful for **interior design, rental property viewing, and real estate floor plans**.

## Pipeline

```
Phone photos
    ↓
COLMAP sparse reconstruction (sparse/0)
    ↓
Auto coordinate alignment ──→ Virtual ortho camera generation
    ↓
Monocular depth estimation (DepthAnythingV2)
    ↓
3DGS training (perspective) + Ortho rendering
    ↓
Difix diffusion restoration
    ↓
SIFT stitching ──→ Room orthophoto floor plan
```

## Quick Start

### Requirements

- Linux, CUDA 11.8, NVIDIA GPU (24GB recommended)
- Python 3.8, PyTorch 2.0, Conda

### Installation

```bash
git clone https://github.com/hjh530/Torthosplatting.git
cd Torthosplatting

# 1. Create environment
conda env create -f environment.yml
conda activate Torthosplatting

# 2. Download model weights
# DepthAnythingV2: https://huggingface.co/spaces/LiheYoung/Depth-Anything-V2
# Place at: checkpoints/depth_anything_v2_vitl.pth

# 3. Compile CUDA modules
export CUDA_HOME=/usr/local/cuda-11.8
cd submodules/diff-gaussian-rasterization && pip install -e . && cd ../..
cd submodules/diff-gaussian-rasterization-ortho && pip install -e . && cd ../..
cd submodules/simple-knn && pip install -e . && cd ../..

# 4. Extra dependencies
pip install xformers==0.0.22 --no-deps
pip install triton==2.0.0
```

### Run

```bash
./run_pipeline.sh -d <dataset> -g 0
# Examples:
./run_pipeline.sh -d room1 -g 0
./run_pipeline.sh -d room1,room2 -g 0
```

### Input

```
dataset/
├── sparse/0/          # COLMAP reconstruction (BIN or TXT)
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── images/            # Original photos
```

### Output

```
dataset/ortho_output/
├── virtual_1_r1_c1.png   # 25 restored ortho tiles
├── ...
└── ortho_stitched.jpg     # Final stitched orthomosaic
```

## Technical Design

### 1. From Perspective to Orthographic

3DGS projects each Gaussian to screen coordinates via two components: the **projection matrix** (Python layer) and the **covariance Jacobian** (CUDA layer). Training and rendering differ in their projection type:

- **Training**: Uses perspective projection matching input photos → DIFIX CUDA
- **Rendering**: Uses orthographic projection for top-down view → Tortho CUDA

#### Projection Matrix

`utils/graphics_utils.py` — `getProjectionMatrix()` generates different matrices based on the `orthographic` flag:

```python
def getProjectionMatrix(znear, zfar, fovX, fovY, orthographic=False):
    if not orthographic:
        # Perspective: frustum — objects shrink with distance
        tanHalfFovY = math.tan(fovY / 2)
        tanHalfFovX = math.tan(fovX / 2)
        top = tanHalfFovY * znear
        right = tanHalfFovX * znear
        P = torch.zeros(4, 4)
        P[0, 0] = 2.0 * znear / (right - left)
        P[0, 2] = (right + left) / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        P[3, 2] = 1.0
        return P
    else:
        # Orthographic: no perspective scaling
        tanHalfFovY = math.tan(fovY / 2)
        tanHalfFovX = math.tan(fovX / 2)
        top = 5          # fixed viewport height
        right = tanHalfFovX * 5 / tanHalfFovY  # maintain aspect ratio
        P = torch.zeros(4, 4)
        P[0, 0] = 2.0 / (right - left)        # X scale (no z)
        P[0, 3] = -(right + left) / (right - left)  # X translation
        P[1, 1] = 2.0 / (top - bottom)        # Y scale (no z)
        P[1, 3] = -(top + bottom) / (top - bottom)  # Y translation
        P[2, 2] = -2.0 / (zfar - znear)
        P[2, 3] = -(zfar + znear) / (zfar - znear)
        P[3, 3] = 1.0
        return (right - left) / 2, (top - bottom) / 2, P
```

**Key difference**: Perspective scales `P[0,0]` and `P[1,1]` by `znear`, making distant objects smaller. Orthographic removes `znear` — objects at all depths appear the same size.

#### Camera Projection

`scene/cameras.py` — `Camera.get_full_proj_transform()` selects projection at runtime:

```python
class Camera(nn.Module):
    def __init__(self, ...):
        # Perspective projection at init (for training)
        self.projection_matrix = getProjectionMatrix(
            znear, zfar, FoVx, FoVy, orthographic=False
        ).transpose(0,1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)

    def get_full_proj_transform(self, orthographic=False):
        if not orthographic:
            return self.full_proj_transform
        else:
            # Orthographic at render time
            tanfovx, tanfovy, proj_mat = getProjectionMatrix(
                znear, zfar, FoVx, FoVy, orthographic=True
            )
            full_proj = (
                self.world_view_transform.unsqueeze(0)
                .bmm(proj_mat.transpose(0,1).cuda().unsqueeze(0))
            ).squeeze(0)
            return tanfovx, tanfovy, full_proj
```

#### CUDA Covariance Jacobian

`forward.cu` — `computeCov2D()` is the only CUDA function that differs between orthographic and perspective:

```c
__device__ float3 computeCov2D(..., const bool orthographic) {
    float3 t = transformPoint4x3(mean, viewmatrix);
    glm::mat3 J;

    if (orthographic) {
        // focal is direct scale — no perspective division
        J = glm::mat3(
            focal_x, 0.0f,    0,
            0.0f,    focal_y, 0,
            0,       0,       0);
    } else {
        // Perspective: 1/z scale + Taylor correction
        const float limx = 1.3f * tan_fovx;
        const float limy = 1.3f * tan_fovy;
        const float txtz = t.x / t.z;
        const float tytz = t.y / t.z;
        t.x = min(limx, max(-limx, txtz)) * t.z;
        t.y = min(limy, max(-limy, tytz)) * t.z;
        J = glm::mat3(
            focal_x / t.z,  0.0f,          -(focal_x * t.x) / (t.z * t.z),
            0.0f,           focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
            0,              0,             0);
    }
    // ... rest of covariance transform
}
```

**Intuition**: In perspective, farther Gaussians (larger `t.z`) project smaller on screen (divided by `t.z`). Orthographic has no depth-dependent scaling.

### 2. Coordinate Transform & Virtual Camera Generation

`utils/gen_virtual_cams.py` transforms the arbitrary COLMAP coordinate system into a standardized one where the ground is horizontal and walls are axis-aligned, then generates virtual orthographic cameras.

#### Pipeline

```
COLMAP original coordinates
    ↓
Step 1: RANSAC ground plane → rotate normal to Z-axis (matrix A1)
    ↓
Step 2: XY projection + Hough lines → rotate walls to XY axes (matrix A2)
    ↓
Final: P' = A2 · A1 · (P - P_ground)
    ↓
5×5 grid virtual cameras in XY bbox → ortho rendering
```

#### Step 1: Ground Detection

```python
# RANSAC plane fitting
normal, p0, inliers = ransac_plane(pts, iters=2000, thresh=0.02)

# Determine ground vs ceiling
A1_temp = rotation_from_vector_to_z(normal)
pts_temp = (pts - p0) @ A1_temp.T
span_low = z50 - z10    # span of lower points
span_high = z90 - z50   # span of upper points
if span_low >= span_high:  # ceiling detected → flip
    normal = -normal
A1 = rotation_from_vector_to_z(normal)  # ground → Z
```

`rotation_from_vector_to_z(n)` computes the rotation matrix that aligns an arbitrary normal vector to `[0,0,1]` using the Rodrigues rotation formula.

#### Step 2: Wall Direction Detection

```python
def align_xy_by_outer_boundary_grid(points):
    # 1. Project to XY plane
    projected_xy = points[:, :2]
    # 2. Density filter to remove outliers
    projected_xy = density_filter_xy(projected_xy)
    # 3. Extract outer contour
    boundary_pts = extract_outer_contour_xy(projected_xy)
    # 4. Hough line detection
    hough_segments = detect_hough_line_segments(projected_xy)
    # 5. Filter edge-proximal segments
    edge_segs = filter_by_boundary_proximity(hough_segments, boundary_pts)
    # 6. Estimate best orthogonal direction
    best_angle = estimate_from_orthogonal_support(edge_segs)
    # 7. Rotate to align
    A2 = rotation_matrix_z(-np.radians(best_angle))
    return A2
```

#### Step 3: Virtual Camera Generation

```python
q_ortho = np.array([0.0, 1.0, 0.0, 0.0])  # downward-facing
R_ortho = qvec2rotmat(q_ortho)

for row in range(1, 6):
    for col in range(1, 6):
        C = centers[count - 1]
        t = -R_ortho @ C
        images[img_id] = {
            "qvec": q_ortho, "tvec": t,
            "camera_id": ref_camera_id,
            "name": f"virtual_{count}_r{row}_c{col}.png",
        }
```

### 3. Four Changes from DIFIX to Tortho

| File | Change |
|------|--------|
| `utils/graphics_utils.py` | `getProjectionMatrix()` adds `orthographic` parameter; ortho branch returns `(half_w, half_h, matrix)` tuple |
| `gaussian_renderer_ortho.py` | New standalone orthographic renderer calling Tortho CUDA |
| `scene/cameras.py` | `Camera` adds `get_full_proj_transform(orthographic)` for runtime projection selection |
| `forward.cu` | `computeCov2D()` adds `orthographic` parameter and ortho Jacobian branch |

## Project Structure

```
Torthosplatting/
├── train.py                  # 3DGS training (perspective)
├── render_ortho.py           # Orthographic rendering
├── restore.py                # Difix image restoration
├── depth_gen.py              # Depth map generation
├── run_pipeline.sh           # One-click pipeline
├── environment.yml           # Conda environment
│
├── scene/                    # 3DGS scene module
├── gaussian_renderer/        # Perspective renderer (training)
├── gaussian_renderer_ortho.py # Ortho renderer (rendering)
│
├── utils/
│   ├── gen_virtual_cams.py   # Virtual camera + coordinate alignment
│   ├── calibrate_depth.py    # Depth scale calibration
│   ├── stitch_ortho.py       # Orthomosaic stitching
│   └── gen_dummy_images.py   # Placeholder image generation
│
├── submodules/
│   ├── diff-gaussian-rasterization/        # Training CUDA (DIFIX)
│   ├── diff-gaussian-rasterization-ortho/  # Rendering CUDA (Tortho)
│   └── simple-knn/
│
├── depth_anything_v2/        # DepthAnythingV2 model
└── checkpoints/              # Model weights
```

## Acknowledgments

- **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)** — Real-time radiance field rendering
- **[2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)** — Depth-regularized training
- **[DIFIX3D+](https://research.nvidia.com/labs/toronto-ai/difix3d/)** — Single-step diffusion for 3D restoration (CVPR 2025 Oral)
- **[DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2)** — Monocular depth estimation
- **[COLMAP](https://github.com/colmap/colmap)** — Structure-from-Motion

## License

For academic research use. Individual components follow their original licenses.
