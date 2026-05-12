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

## Results / 效果展示

| playroom | room |
|----------|------|
| ![playroom](assets/playroom.jpg) | ![room](assets/room.jpg) |

| drjohnson | olohuone |
|-----------|----------|
| ![drjohnson](assets/drjohnson.jpg) | ![olohuone](assets/olohuone.jpg) |

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

### 1. From Perspective to Orthographic: Why and How

3DGS renders by projecting each 3D Gaussian ellipsoid onto the 2D image plane. This projection involves two mathematical components:

1. **Projection Matrix** $P$ (Python): maps camera-space coordinates $(x_c, y_c, z_c)$ to clip space $(x_{clip}, y_{clip}, z_{clip})$
2. **Covariance Jacobian** $J$ (CUDA): projects the 3D covariance matrix $\Sigma_{3D}$ to 2D screen space $\Sigma_{2D}$

The relationship between training (perspective) and rendering (orthographic) is:

| Stage | Projection | CUDA Module | Why |
|-------|-----------|-------------|-----|
| Training | Perspective | 3DGS | Must match input photos (pinhole/simple pinhole) |
| Rendering | Orthographic | Tortho | Top-down floor plan needs parallel projection |

#### 1.1 Perspective Projection (Training)

A standard pinhole camera maps a 3D point $(X, Y, Z)$ to pixel coordinates $(u, v)$ :

$$u = f_x \cdot \frac{X}{Z} + c_x, \quad v = f_y \cdot \frac{Y}{Z} + c_y$$

where $f_x, f_y$ are focal lengths in pixels and $c_x, c_y$ is the principal point. The key observation: coordinates are **divided by depth $Z$** — distant objects appear smaller.

The OpenGL-style perspective projection matrix $P_{persp}$ :

$$P_{persp} = \begin{bmatrix}
\frac{2n}{r-l} & 0 & \frac{r+l}{r-l} & 0 \\
0 & \frac{2n}{t-b} & \frac{t+b}{t-b} & 0 \\
0 & 0 & \frac{f}{f-n} & -\frac{fn}{f-n} \\
0 & 0 & 1 & 0
\end{bmatrix}$$

where $n, f$ = near/far planes, $l, r, b, t$ = frustum boundaries at $z=n$ .

**In code** (`utils/graphics_utils.py`):

```python
# Perspective: frustum — znear determines scale
tanHalfFovY = math.tan(fovY / 2)
tanHalfFovX = math.tan(fovX / 2)
top = tanHalfFovY * znear    # ← scales with znear
right = tanHalfFovX * znear  # ← scales with znear
P[0,0] = 2.0 * znear / (right - left)
P[1,1] = 2.0 * znear / (top - bottom)
```

#### 1.2 Orthographic Projection (Rendering)

Orthographic projection discards depth-dependent scaling. A 3D point maps as:

$$u = f_x \cdot X + c_x, \quad v = f_y \cdot Y + c_y$$

No division by $Z$ . Objects at all depths appear the same size — exactly what we want for a floor plan.

The orthographic projection matrix $P_{ortho}$ :

$$P_{ortho} = \begin{bmatrix}
\frac{2}{r-l} & 0 & 0 & -\frac{r+l}{r-l} \\
0 & \frac{2}{t-b} & 0 & -\frac{t+b}{t-b} \\
0 & 0 & -\frac{2}{f-n} & -\frac{f+n}{f-n} \\
0 & 0 & 0 & 1
\end{bmatrix}$$

Key differences from $P_{persp}$ :
- **No $n$ (znear)** in $P[0,0]$ and $P[1,1]$ : scaling independent of depth
- **$P[0,3]$ , $P[1,3]$** instead of $P[0,2]$ , $P[1,2]$ : translation in XY, not perspective shift
- **$P[3,2]=0$** instead of $1$ : no perspective divide in homogeneous coordinates

**In code**:

```python
# Orthographic: fixed viewport, no znear
top = 5                    # fixed viewport height
right = tanHalfFovX * 5 / tanHalfFovY  # maintain aspect ratio
P[0,0] = 2.0 / (right - left)      # no znear → same scale at all depths
P[0,3] = -(right + left) / (right - left)  # XY translation
return (right-left)/2, (top-bottom)/2, P  # return half-dims for renderer
```

#### 1.3 Why the Jacobian Must Also Change

The covariance of a 3D Gaussian $\Sigma_{3D}$ projects to 2D screen space as:

$$\Sigma_{2D} = J \cdot W \cdot \Sigma_{3D} \cdot W^T \cdot J^T$$

where $W$ is the rotational part of the view matrix, and $J$ is the **projective Jacobian** — the derivative of screen coordinates w.r.t. camera coordinates:

$$J = \begin{bmatrix}
\frac{\partial u}{\partial X} & \frac{\partial u}{\partial Y} & \frac{\partial u}{\partial Z} \\
\frac{\partial v}{\partial X} & \frac{\partial v}{\partial Y} & \frac{\partial v}{\partial Z}
\end{bmatrix}$$

For **perspective** projection ($u = f_x \cdot X/Z$ , $v = f_y \cdot Y/Z$):

$$J_{persp} = \begin{bmatrix}
\frac{f_x}{Z} & 0 & -\frac{f_x \cdot X}{Z^2} \\
0 & \frac{f_y}{Z} & -\frac{f_y \cdot Y}{Z^2}
\end{bmatrix}$$

The $-\frac{f \cdot X}{Z^2}$ term is the **Taylor expansion correction**: as the Gaussian moves sideways ($\Delta X$), its depth $Z$ changes its screen position.

For **orthographic** projection ($u = f_x \cdot X$ , $v = f_y \cdot Y$):

$$J_{ortho} = \begin{bmatrix}
f_x & 0 & 0 \\
0 & f_y & 0
\end{bmatrix}$$

No $1/Z$ scaling, no Taylor correction — the derivative is constant.

**In CUDA** (`forward.cu`):

```c
if (orthographic) {
    J = glm::mat3(focal_x, 0, 0,  0, focal_y, 0,  0, 0, 0);
} else {
    J = glm::mat3(
        focal_x / t.z, 0, -(focal_x * t.x) / (t.z * t.z),
        0, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);
}
```

#### 1.4 Why Not Just Train with Orthographic?

Training requires perspective because the input photos are perspective projections. If we trained with orthographic projection, the optimization would receive wrong gradients — the model would try to fit perspective images with an orthographic renderer, producing distorted geometry. By keeping training in perspective, the model learns correct 3D structure from the input photos, and we only switch to orthographic at the final rendering stage.

### 2. Coordinate Transform & Virtual Camera Generation

The COLMAP reconstruction produces an arbitrary coordinate system — the ground may be tilted, walls rotated. `utils/gen_virtual_cams.py` standardizes this and generates top-down virtual cameras.

#### 2.1 Camera-Based Ground Plane Estimation

Since COLMAP axes are arbitrary, we need a reliable reference for "up". Rather than relying on the largest RANSAC plane (which may be a wall), we use **camera positions**: people walk around the room, so camera positions naturally lie on a roughly horizontal plane.

```python
# Fit a plane to all camera positions via SVD
cam_positions = []  # world-space camera centers
for img in images.values():
    R = qvec2rotmat(img["qvec"])
    C = -R.T @ img["tvec"]
    cam_positions.append(C)

cam_center = cam_positions.mean(axis=0)
_, _, Vt = np.linalg.svd(cam_positions - cam_center)
cam_plane_normal = Vt[-1]  # minimum variance → normal to camera plane
```

This `cam_plane_normal` serves as our ground truth "up" vector, since cameras are held at roughly constant height above the floor.

#### 2.2 Step 1: Iterative Wall Exclusion ($A_1$)

RANSAC finds the largest planar surface in the point cloud. In rooms with large blank walls, this may be a wall rather than the floor. We iterate: if the RANSAC normal is perpendicular to the camera-plane normal, it's a wall — exclude its inliers and try again.

```python
for plane_idx in range(max_planes):
    normal, p0, inliers = ransac_plane(remaining_pts, ransac_iters, threshold)
    dot_cam = abs(dot(normal, cam_plane_normal))  # parallel → floor, perpendicular → wall

    if dot_cam > 0.7:  # Ground or ceiling found
        ground_normal, ground_p0 = normal, p0
        break
    else:  # Wall — exclude and continue searching
        remaining_pts = remaining_pts[~inliers]
```

**Ground vs. ceiling**: rotate the normal to Z, compare point spread below vs. above the median:

$$\mathrm{span_{low}} = P_{50}(z) - P_{10}(z), \quad \mathrm{span_{high}} = P_{90}(z) - P_{50}(z)$$

If $\mathrm{span_{low}} \geq \mathrm{span_{high}}$ , flip the normal (ceiling → ground).

**Rodrigues rotation** to align $n$ to $[0,0,1]$ :

$$A_1 = I + [v]_\times + [v]_\times^2 \cdot \frac{1-c}{s^2}$$

where $v = n \times [0,0,1]$ , $c = n \cdot [0,0,1]$ , $s = \|v\|$ .

#### 2.3 Step 2: Wall Alignment ($A_2$)

**XY density projection**: project aligned points to the XY plane, apply density-based outlier filtering using k-NN distances and MAD threshold:

$$\mathrm{keep}(p_i) = \mathrm{knn\_dist}(p_i) \leq \min(P_{97}(\mathrm{knn\_dist}), \mathrm{median} + 4 \cdot \mathrm{MAD})$$

**Hough line detection**: rasterize the projected points, apply Canny edge detection, then probabilistic Hough transform to extract line segments $\{(\mathbf{p}_1, \mathbf{p}_2)\}$ .

**Edge filtering**: keep only segments near the outer contour of the point cloud.

**Orthogonal direction search**: for each candidate angle $\theta \in [0°, 180°)$ , compute the total support from line segments aligned to $\theta$ and $\theta + 90°$ :

$$\mathrm{score}(\theta) = \sum_{\mathrm{seg } \approx \theta} \mathrm{length}(\mathrm{seg}) + \sum_{\mathrm{seg } \approx \theta+90°} \mathrm{length}(\mathrm{seg})$$

Select the angle $\theta^*$ that maximizes this score.

**Rotation to align**: $A_2 = R_z(-\theta^*)$ , where $R_z$ is a rotation about the Z-axis:

$$A_2 = \begin{bmatrix}
\cos\theta^* & -\sin\theta^* & 0 \\
\sin\theta^* & \cos\theta^* & 0 \\
0 & 0 & 1
\end{bmatrix}$$

#### 2.4 Step 3: Virtual Camera Placement

After alignment, the scene has a well-defined XY bounding box. Place 25 virtual cameras in a $5 \times 5$ grid:

$$\mathbf{C}_{ij} = \begin{bmatrix} x_{min} + (i+0.5) \cdot \frac{x_{max}-x_{min}}{5} \\ y_{min} + (j+0.5) \cdot \frac{y_{max}-y_{min}}{5} \\ 0.8 \cdot z_{max} \end{bmatrix}, \quad i,j \in \{0,1,2,3,4\}$$

Each camera faces downward with rotation quaternion $\mathbf{q} = [0, 1, 0, 0]$ (identity rotation, looking along $-Z$ in world coordinates).

The camera extrinsics: $\mathbf{t} = -R_{ortho} \cdot \mathbf{C}$ .

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

### 3. Four Changes from DIFIX to Tortho

| File | Change |
|------|--------|
| `utils/graphics_utils.py` | `getProjectionMatrix()` adds `orthographic` parameter; ortho branch returns `(half_w, half_h, matrix)` |
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
- **[DIFIX3D+](https://research.nvidia.com/labs/toronto-ai/difix3d/)** — Single-step diffusion for 3D restoration (CVPR 2025 Oral)
- **[DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2)** — Monocular depth estimation
- **[COLMAP](https://github.com/colmap/colmap)** — Structure-from-Motion

## License

For academic research use. Individual components follow their original licenses.
