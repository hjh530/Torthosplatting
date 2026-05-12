import numpy as np
import os
import random
from scipy.spatial import cKDTree
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 3DGS BIN 格式读取
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                  read_points3D_binary, read_extrinsics_text,
                                  read_intrinsics_text, read_points3D_text)

# ---------- 路径配置（在 main() 中设置） ----------

# ---------- 参数：地面拟合 ----------
RANSAC_ITER = 2000
DIST_THRESHOLD = 0.02
GROUND_POINT_ID = None

# ---------- 参数：高度统计 ----------
LOWEST_GROUND_PERCENTILE = 1.0
FLOOR_SLICE_PERCENTILE = 35.0
ALIGN_FLOOR_PERCENTILE = 20.0

# 密度去离群
DENSITY_NEIGHBOR_K = 8
DENSITY_KEEP_PERCENTILE = 97.0
DENSITY_MAD_SCALE = 4.0

# Hough 线段提取与边缘约束
HOUGH_USE_MORPHOLOGY = True
HOUGH_CANNY_LOW = 40
HOUGH_CANNY_HIGH = 120
HOUGH_THRESHOLD = 18
HOUGH_MIN_LINE_PIXELS = 8
HOUGH_MIN_LINE_SCALE = 0.06
HOUGH_MAX_LINE_GAP_PIXELS = 8
HOUGH_KEEP_LENGTH_PERCENTILE = 65.0
HOUGH_BOUNDARY_DIST_SCALE = 1.8
HOUGH_BOUNDARY_DIST_MIN = 0.07
HOUGH_BOUNDARY_NEAR_RATIO = 0.55

# 垂直容差
PERP_ANGLE_TOL_DEG = 18.0
PERP_PAIR_MIN_TOTAL_LENGTH = 4.0

# 虚拟相机
VIRTUAL_GRID_SIZE = 5

# ---------- 基础数学工具 ----------
def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x + y*y)]
    ])

def rotmat2qvec(R):
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qvec = np.array([0.25 * s,
                         (R[2,1]-R[1,2])/s,
                         (R[0,2]-R[2,0])/s,
                         (R[1,0]-R[0,1])/s])
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            s = np.sqrt(max(0,1+R[0,0]-R[1,1]-R[2,2]))*2
            qvec = np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
        elif i == 1:
            s = np.sqrt(max(0,1+R[1,1]-R[0,0]-R[2,2]))*2
            qvec = np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
        else:
            s = np.sqrt(max(0,1+R[2,2]-R[0,0]-R[1,1]))*2
            qvec = np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])
    qvec = normalize(qvec)
    if qvec[0] < 0: qvec = -qvec
    return qvec

def normalize(v):
    n = np.linalg.norm(v)
    return v if n==0 else v/n

def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def fold_angle_to_axis(angle_deg_180):
    axis = angle_deg_180 % 90.0
    if axis > 45: axis -= 90
    return axis

def axis_angle_gap(a, b):
    d = abs(a - b) % 90.0
    return min(d, 90.0 - d)

# ---------- 读写函数 ----------
def read_cameras_txt(path):
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            line = line.replace(',', ' ')
            e = line.split()
            cams[int(e[0])] = {"model": e[1], "width": int(e[2]),
                               "height": int(e[3]), "params": np.array(list(map(float, e[4:])))}
    return cams

# ---------- BIN 格式写函数 ----------
import struct

CAMERA_MODEL_IDS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2,
                     "RADIAL": 3, "OPENCV": 4, "OPENCV_FISHEYE": 5,
                     "FULL_OPENCV": 6, "FOV": 7, "SIMPLE_RADIAL_FISHEYE": 8,
                     "THIN_PRISM_FISHEYE": 9}

def write_next_bytes(fid, data, fmt):
    if isinstance(data, np.ndarray):
        data = data.flatten()
    fid.write(struct.pack(fmt, *data) if isinstance(data, (list, np.ndarray)) else struct.pack(fmt, data))

def write_cameras_bin(path, cams):
    with open(path, "wb") as f:
        write_next_bytes(f, len(cams), "Q")
        for cid in sorted(cams):
            cam = cams[cid]
            model_id = CAMERA_MODEL_IDS.get(cam["model"], 1)
            params = np.array(cam["params"], dtype=np.float64)
            write_next_bytes(f, cid, "i")
            write_next_bytes(f, model_id, "i")
            write_next_bytes(f, cam["width"], "Q")
            write_next_bytes(f, cam["height"], "Q")
            for p in params:
                write_next_bytes(f, p, "d")

def write_images_bin(path, images):
    with open(path, "wb") as f:
        write_next_bytes(f, len(images), "Q")
        for img_id in sorted(images):
            img = images[img_id]
            qvec = np.array(img["qvec"], dtype=np.float64)
            tvec = np.array(img["tvec"], dtype=np.float64)
            write_next_bytes(f, img_id, "i")
            for v in qvec: write_next_bytes(f, v, "d")
            for v in tvec: write_next_bytes(f, v, "d")
            write_next_bytes(f, img["camera_id"], "i")
            name = img["name"] + "\0"
            f.write(name.encode())
            pts = img.get("points2D", [])
            write_next_bytes(f, len(pts), "Q")
            for x, y, pid in pts:
                write_next_bytes(f, float(x), "d")
                write_next_bytes(f, float(y), "d")
                write_next_bytes(f, int(pid), "q")

def write_points3D_bin(path, points3D, point_tracks):
    with open(path, "wb") as f:
        write_next_bytes(f, len(points3D), "Q")
        for pid in sorted(points3D):
            pt = points3D[pid]
            write_next_bytes(f, pid, "q")
            write_next_bytes(f, pt["xyz"].astype(np.float64), "ddd")
            rgb = pt.get("rgb", (0, 0, 0))
            write_next_bytes(f, int(rgb[0]), "B")
            write_next_bytes(f, int(rgb[1]), "B")
            write_next_bytes(f, int(rgb[2]), "B")
            write_next_bytes(f, float(pt["error"]), "d")
            tracks = point_tracks.get(pid, [])
            write_next_bytes(f, len(tracks), "Q")
            for img_id, idx in tracks:
                write_next_bytes(f, img_id, "i")
                write_next_bytes(f, idx, "i")


def write_cameras_txt(path, cams):
    with open(path, 'w') as f:
        f.write("# Camera list (ID, model, width, height, params)\n")
        for cid in sorted(cams):
            cam = cams[cid]
            f.write(f"{cid} {cam['model']} {cam['width']} {cam['height']} "
                    + " ".join(map(str, cam['params'])) + "\n")

def read_images_txt(path):
    images, point_tracks = {}, {}
    with open(path, 'r') as f: lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line: i+=1; continue
        line = line.replace(',', ' ')
        e = line.split()
        try:
            img_id = int(e[0])
            qvec = np.array(list(map(float, e[1:5])))
            tvec = np.array(list(map(float, e[5:8])))
            cam_id = int(e[8])
            name = e[9]
        except ValueError:
            i+=1; continue
        i+=1
        pts2d = []
        if i < len(lines):
            l2 = lines[i].strip()
            if l2:
                l2 = l2.replace(',', ' ')
                e2 = l2.split()
                for idx in range(0, len(e2), 3):
                    x, y, pid = float(e2[idx]), float(e2[idx+1]), int(e2[idx+2])
                    pts2d.append((x, y, pid))
                    if pid != -1: point_tracks.setdefault(pid, []).append((img_id, idx//3))
        images[img_id] = {"qvec": qvec, "tvec": tvec, "camera_id": cam_id,
                          "name": name, "points2D": pts2d}
        i+=1
    return images, point_tracks

def write_images_txt(path, images):
    with open(path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img_id in sorted(images):
            img = images[img_id]
            f.write(f"{img_id} {' '.join(map(str, img['qvec']))} {' '.join(map(str, img['tvec']))} "
                    f"{img['camera_id']} {img['name']}\n")
            if img['points2D']:
                f.write(" ".join(f"{x} {y} {pid}" for x,y,pid in img['points2D']) + "\n")
            else:
                f.write("\n")

def read_points3D_txt(path):
    pts = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            line = line.replace(',', ' ')
            e = line.split()
            pts[int(e[0])] = {"xyz": np.array(list(map(float, e[1:4]))),
                              "rgb": (int(e[4]), int(e[5]), int(e[6])),
                              "error": float(e[7])}
    return pts

def write_points3D_txt(path, points3D, point_tracks):
    with open(path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points3D)}\n")
        for pid in sorted(points3D):
            pt = points3D[pid]
            line = f"{pid} {' '.join(map(str, pt['xyz']))} {' '.join(map(str, pt['rgb']))} {pt['error']}"
            if pid in point_tracks:
                line += " " + " ".join(f"{img_id} {idx}" for img_id, idx in point_tracks[pid])
            f.write(line + "\n")

# ---------- 几何工具 ----------
def fit_plane_from_points(pts):
    centroid = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts - centroid)
    return Vt[-1] / np.linalg.norm(Vt[-1]), centroid

def ransac_plane(pts, iters=2000, thresh=0.02):
    best_inliers, best_n, best_p0 = [], None, None
    N = len(pts)
    for _ in range(iters):
        ids = random.sample(range(N), 3)
        p1 = pts[ids[0]]
        v1 = pts[ids[1]] - p1
        v2 = pts[ids[2]] - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-6: continue
        n /= norm
        dists = np.abs((pts - p1).dot(n))
        inliers = np.where(dists < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers, best_n, best_p0 = inliers, n, p1
    if best_n is None:
        best_n, best_p0 = fit_plane_from_points(pts)
    return best_n, best_p0, best_inliers

def rotation_from_vector_to_z(n):
    target = np.array([0., 0., 1.])
    n = normalize(n)
    if np.allclose(n, target): return np.eye(3)
    if np.allclose(n, -target): return np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    v = np.cross(n, target)
    s = np.linalg.norm(v)
    c = np.dot(n, target)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s**2))

def extract_outer_contour_xy(points_xy, cell_size=None):
    if len(points_xy) < 30:
        return points_xy.copy(), 0.1

    image, min_xy, cell_size = rasterize_xy_points(points_xy, cell_size)
    h, w = image.shape

    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return points_xy.copy(), cell_size

    largest = max(contours, key=cv2.contourArea)
    pixel_pts = largest.reshape(-1, 2).astype(float)

    step = max(1, len(pixel_pts) // 800)
    pixel_pts = pixel_pts[::step]

    boundary_real = min_xy + (pixel_pts + 0.5) * cell_size
    return boundary_real, cell_size

def density_filter_xy(points_xy, neighbor_k=8, keep_percentile=97.0, mad_scale=4.0):
    n = len(points_xy)
    if n <= max(20, neighbor_k+1):
        return points_xy
    k_eff = min(neighbor_k+1, n)
    tree = cKDTree(points_xy)
    dists, _ = tree.query(points_xy, k=k_eff)
    if dists.ndim == 1 or dists.shape[1] < 2:
        return points_xy
    knn_dist = dists[:, -1]
    p_thr = np.percentile(knn_dist, keep_percentile)
    med = np.median(knn_dist)
    mad = np.median(np.abs(knn_dist - med)) + 1e-9
    thr = min(p_thr, med + mad_scale * mad)
    keep = knn_dist <= thr
    kept = points_xy[keep]
    if len(kept) < max(200, int(0.55*n)):
        return points_xy
    return kept

def select_mid_height_points(points):
    if len(points) == 0: return points
    z_low = np.percentile(points[:,2], 10)
    z_high = np.percentile(points[:,2], 90)
    candidates = points[(points[:,2] >= z_low) & (points[:,2] <= z_high)]
    return candidates if len(candidates) > 0 else points

def prepare_projected_xy(points, min_filtered_points=None):
    if len(points) == 0:
        return np.empty((0,2), dtype=float)
    candidates = select_mid_height_points(points)
    proj = density_filter_xy(candidates[:, :2])
    if min_filtered_points and len(proj) < min_filtered_points:
        return candidates[:, :2]
    return proj

def rasterize_xy_points(points_xy, cell_size=None):
    if len(points_xy) == 0:
        return np.zeros((1,1), dtype=np.uint8), np.zeros(2), 1.0
    min_xy = points_xy.min(axis=0)
    max_xy = points_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    if cell_size is None:
        cell_size = float(np.clip(max(span) / 280.0, 0.02, 0.10))
    ij = np.floor((points_xy - min_xy) / cell_size).astype(int)
    ij -= ij.min(axis=0, keepdims=True)
    w = max(int(ij[:,0].max())+1, 1)
    h = max(int(ij[:,1].max())+1, 1)
    image = np.zeros((h, w), dtype=np.uint8)
    image[ij[:,1], ij[:,0]] = 255
    return image, min_xy, float(cell_size)

def detect_hough_line_segments(points_xy, use_morphology=True):
    if len(points_xy) < 30:
        return [], 0.0
    image, min_xy, cell_size = rasterize_xy_points(points_xy)
    if use_morphology:
        kernel = np.ones((3,3), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
        image = cv2.dilate(image, kernel, iterations=1)
    edges = cv2.Canny(image, HOUGH_CANNY_LOW, HOUGH_CANNY_HIGH)
    min_side = max(1, int(min(image.shape)))
    min_len = max(HOUGH_MIN_LINE_PIXELS, int(min_side * HOUGH_MIN_LINE_SCALE))
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180.0, threshold=HOUGH_THRESHOLD,
                            minLineLength=min_len, maxLineGap=HOUGH_MAX_LINE_GAP_PIXELS)
    if lines is None or len(lines) < 6:
        loose_thr = max(10, HOUGH_THRESHOLD-6)
        loose_min_len = max(5, int(min_len*0.65))
        loose_max_gap = max(HOUGH_MAX_LINE_GAP_PIXELS+4, int(HOUGH_MAX_LINE_GAP_PIXELS*1.5))
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180.0, threshold=loose_thr,
                                minLineLength=loose_min_len, maxLineGap=loose_max_gap)
    if lines is None:
        return [], cell_size
    segments = []
    for x1,y1,x2,y2 in lines[:,0]:
        p1 = np.array([min_xy[0] + (x1+0.5)*cell_size, min_xy[1] + (y1+0.5)*cell_size])
        p2 = np.array([min_xy[0] + (x2+0.5)*cell_size, min_xy[1] + (y2+0.5)*cell_size])
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        if length < cell_size*6: continue
        angle_180 = float(np.degrees(np.arctan2(vec[1], vec[0])) % 180.0)
        segments.append({
            "x1": float(p1[0]), "y1": float(p1[1]),
            "x2": float(p2[0]), "y2": float(p2[1]),
            "length": length,
            "angle_deg_180": angle_180,
            "angle_axis_deg": fold_angle_to_axis(angle_180)
        })
    return segments, cell_size

def detect_boundary_hough_segments(projected_xy, use_morphology=True):
    boundary_pts, bound_cell = extract_outer_contour_xy(projected_xy)
    hough_segs, raster_cell = detect_hough_line_segments(projected_xy, use_morphology)

    if len(boundary_pts) < 8:
        edge_segs = sorted(hough_segs, key=lambda s: -s["length"])[:12]
    else:
        tree = cKDTree(boundary_pts)
        dist_thr = max(bound_cell * 2.5, HOUGH_BOUNDARY_DIST_MIN)
        edge_segs = []
        for seg in hough_segs:
            p1 = np.array([seg["x1"], seg["y1"]])
            p2 = np.array([seg["x2"], seg["y2"]])
            d1, d2 = tree.query(p1)[0], tree.query(p2)[0]
            if d1 > dist_thr and d2 > dist_thr:
                continue
            sample_pts = np.linspace(p1, p2, 5)
            dists, _ = tree.query(sample_pts)
            if np.mean(dists <= dist_thr) < 0.6:
                continue
            edge_segs.append(seg)
        if len(edge_segs) < 6:
            edge_segs = sorted(hough_segs, key=lambda s: -s["length"])[:12]

    return boundary_pts, bound_cell, hough_segs, raster_cell, edge_segs

def estimate_angle_from_hough_segments(segments):
    if not segments:
        return None
    hist = np.zeros(180)
    for seg in segments:
        bin_idx = min(int(round((seg["angle_axis_deg"]+45)*2)), 179)
        hist[bin_idx] += seg["length"]
    return np.argmax(hist)/2 - 45.0

def estimate_angle_from_outer_dominant_lines(edge_segments, tol_deg=8.0):
    if len(edge_segments) < 3:
        return None
    lengths = np.array([s["length"] for s in edge_segments], dtype=float)
    total_len = lengths.sum()
    best = None
    for i, seg_i in enumerate(edge_segments):
        a0 = seg_i["angle_axis_deg"]
        indices = [j for j, s in enumerate(edge_segments)
                   if axis_angle_gap(s["angle_axis_deg"], a0) <= tol_deg]
        if len(indices) < 2: continue
        cluster_len = lengths[indices].sum()
        max_len = lengths[indices].max()
        support_ratio = cluster_len / total_len
        score = cluster_len + 0.3*max_len
        if best is None or score > best["score"]:
            angles = np.array([edge_segments[j]["angle_axis_deg"] for j in indices])
            weighted_angle = float(np.average(angles, weights=lengths[indices]))
            best = {"angle_axis_deg": weighted_angle,
                    "support_ratio": support_ratio,
                    "max_len": max_len,
                    "support_count": len(indices),
                    "score": score,
                    "segment_indices": indices}
    if best and best["support_ratio"] >= 0.36 and best["max_len"] >= 1.8:
        return best
    return None

def estimate_angle_from_orthogonal_support(segments, tol_deg=9.0):
    if len(segments) < 4:
        return None
    angles = np.array([s["angle_deg_180"] for s in segments], dtype=float)
    weights = np.array([s["length"] for s in segments], dtype=float)
    total_w = weights.sum()
    if total_w < 1e-9:
        return None
    best = None
    for theta in range(180):
        theta_orth = (theta+90)%180
        d0 = np.minimum(np.abs(angles-theta), 180-np.abs(angles-theta))
        d1 = np.minimum(np.abs(angles-theta_orth), 180-np.abs(angles-theta_orth))
        s0 = weights[d0<=tol_deg].sum()
        s1 = weights[d1<=tol_deg].sum()
        score = s0 + s1 + 0.8*min(s0,s1)
        if best is None or score > best["score"]:
            best = {"theta": float(theta), "s0": s0, "s1": s1, "score": score}
    if best is None: return None
    main = max(best["s0"], best["s1"])
    orth = min(best["s0"], best["s1"])
    main_theta = best["theta"] if best["s0"]>=best["s1"] else (best["theta"]+90)%180
    return {"angle_axis_deg": fold_angle_to_axis(main_theta),
            "main_ratio": main/total_w,
            "balance_ratio": orth/total_w,
            "main_support": main,
            "ortho_support": orth,
            "score": best["score"]}

def choose_perpendicular_segment_pair(segments, tol_deg=PERP_ANGLE_TOL_DEG):
    if len(segments) < 2: return None
    best = None
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            delta = abs(segments[i]["angle_deg_180"] - segments[j]["angle_deg_180"])
            delta = min(delta, 180-delta)
            err = abs(delta-90)
            if err > tol_deg: continue
            total_len = segments[i]["length"] + segments[j]["length"]
            score = total_len - 0.25*err
            dominant = segments[i] if segments[i]["length"]>=segments[j]["length"] else segments[j]
            cand = {"score": score, "angle_axis_deg": dominant["angle_axis_deg"],
                    "perp_err_deg": err, "total_length": total_len, "pair_idx": (i,j)}
            if best is None or cand["score"] > best["score"] or \
               (np.isclose(cand["score"], best["score"]) and cand["perp_err_deg"]<best["perp_err_deg"]):
                best = cand
    if best and best["total_length"] >= PERP_PAIR_MIN_TOTAL_LENGTH:
        return best
    return None

# ---------- 核心对齐 ----------
def align_xy_by_outer_boundary_grid(points):
    if len(points) < 50:
        return np.eye(3), {"selected_segments": [], "boundary_points": [],
                           "hough_segments": [], "edge_segments": [],
                           "cell_size": 0, "raster_cell_size": 0,
                           "chosen_angle": None, "method": "none"}

    print("正在进行基于投影图像线段的 XY 对齐...")
    projected_xy = prepare_projected_xy(points)
    if len(projected_xy) < 30:
        print("候选点不足，保持当前 XY 方向。")
        return np.eye(3), {"selected_segments": [], "boundary_points": [],
                           "hough_segments": [], "edge_segments": [],
                           "cell_size": 0, "raster_cell_size": 0,
                           "chosen_angle": None, "method": "none"}

    boundary_pts, bound_cell, hough_segs, raster_cell, edge_segs = \
        detect_boundary_hough_segments(projected_xy, use_morphology=HOUGH_USE_MORPHOLOGY)

    source_segments = edge_segs if len(edge_segs) >= 4 else hough_segs

    outer_dom = estimate_angle_from_outer_dominant_lines(edge_segs)
    orth_cand = estimate_angle_from_orthogonal_support(source_segments)
    perp_pair = choose_perpendicular_segment_pair(source_segments)
    hough_angle = estimate_angle_from_hough_segments(hough_segs)

    chosen_angle = None
    method = ""
    selected_segments = []

    if outer_dom is not None:
        chosen_angle = outer_dom["angle_axis_deg"]
        method = "outer_dominant"

    if chosen_angle is None:
        use_orth = (orth_cand is not None and
                    orth_cand["main_ratio"] >= 0.22 and
                    orth_cand["balance_ratio"] >= 0.05)
        if use_orth and perp_pair is not None:
            gap = axis_angle_gap(perp_pair["angle_axis_deg"], orth_cand["angle_axis_deg"])
            if gap > 18.0 and perp_pair["total_length"] > 1.8 * orth_cand["ortho_support"]:
                use_orth = False
        if use_orth:
            chosen_angle = orth_cand["angle_axis_deg"]
            method = "global_orth"

    if chosen_angle is None and perp_pair is not None:
        total_len = sum(s["length"] for s in source_segments) + 1e-9
        ratio = perp_pair["total_length"] / total_len
        if perp_pair["perp_err_deg"] <= 20.0 and ratio >= 0.14:
            chosen_angle = perp_pair["angle_axis_deg"]
            method = "perp_pair"

    if chosen_angle is None and hough_angle is not None and len(hough_segs) >= 4:
        chosen_angle = hough_angle
        method = "hough_hist"

    if chosen_angle is not None:
        if method == "perp_pair" and perp_pair is not None:
            i, j = perp_pair["pair_idx"]
            if i < len(source_segments) and j < len(source_segments):
                selected_segments = [source_segments[i], source_segments[j]]
        elif method == "outer_dominant" and outer_dom is not None:
            indices = outer_dom.get("segment_indices", [])
            selected_segments = [edge_segs[idx] for idx in indices if idx < len(edge_segs)]
        else:
            tol = 8.0
            selected_segments = [s for s in source_segments
                                 if axis_angle_gap(s["angle_axis_deg"], chosen_angle) <= tol]

    diagnostics = {
        "selected_segments": selected_segments,
        "boundary_points": boundary_pts,
        "hough_segments": hough_segs,
        "edge_segments": edge_segs,
        "cell_size": bound_cell,
        "raster_cell_size": raster_cell,
        "chosen_angle": chosen_angle,
        "method": method,
    }

    if chosen_angle is not None:
        print(f"Projection-Hough({method}) angle: {chosen_angle:.2f} deg, "
              f"selected_lines={len(selected_segments)}, "
              f"hough_lines={len(hough_segs)}, edge_lines={len(edge_segs)}, "
              f"boundary_points={len(boundary_pts)}, boundary_cell={bound_cell:.2f}, "
              f"raster_cell={raster_cell:.3f}, morph={HOUGH_USE_MORPHOLOGY}")
        return rotation_matrix_z(-np.radians(chosen_angle)), diagnostics
    else:
        print("投影线段方向不稳定，保持当前 XY 方向。")
        return np.eye(3), diagnostics

# ---------- 包围盒与辅助 ----------
def get_robust_bbox(points, lower=2.0, upper=98.0):
    if len(points)==0: return (0,0,0,0,0,0)
    return (np.percentile(points[:,0], lower), np.percentile(points[:,0], upper),
            np.percentile(points[:,1], lower), np.percentile(points[:,1], upper),
            np.percentile(points[:,2], lower), np.percentile(points[:,2], upper))

def select_floor_slice(points, percentile=FLOOR_SLICE_PERCENTILE):
    if len(points)==0: return points
    z = np.percentile(points[:,2], percentile)
    return points[points[:,2] <= z]

def compute_lowest_ground_reference(points):
    if len(points)==0: return np.zeros(3)
    z = np.percentile(points[:,2], LOWEST_GROUND_PERCENTILE)
    floor = select_floor_slice(points, ALIGN_FLOOR_PERCENTILE)
    xy = np.median(floor[:,:2], axis=0) if len(floor)>0 else np.zeros(2)
    return np.array([xy[0], xy[1], z])

def compute_virtual_camera_centers_grid(bbox, height):
    minx, maxx, miny, maxy = bbox[:4]
    dx, dy = maxx-minx, maxy-miny
    centers = []
    for row in range(VIRTUAL_GRID_SIZE):
        fy = (row+0.5)/VIRTUAL_GRID_SIZE
        for col in range(VIRTUAL_GRID_SIZE):
            fx = (col+0.5)/VIRTUAL_GRID_SIZE
            centers.append([minx + fx*dx, miny + fy*dy, height])
    return np.array(centers, dtype=float)

# ---------- 绘图 ----------
def save_xoy_projection_plot(points_array, bbox, target_height, save_path,
                             use_morphology=True, highlight_segments=None):
    if len(points_array) == 0: return
    proj = prepare_projected_xy(points_array)
    bound, cell, hough_segs, rcell, edge_segs = detect_boundary_hough_segments(proj, use_morphology)
    virt = compute_virtual_camera_centers_grid(bbox, target_height)
    minx, maxx, miny, maxy = bbox[:4]

    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(proj[:,0], proj[:,1], s=3, c="#7aa6c2", alpha=0.35, label="Projected points")
    if len(bound) > 0:
        ax.scatter(bound[:,0], bound[:,1], s=8, c="#c84b31", alpha=0.8, label="Boundary points")
    for idx, seg in enumerate(hough_segs):
        ax.plot([seg["x1"],seg["x2"]], [seg["y1"],seg["y2"]],
                color="#1b1f24", linewidth=1.1, alpha=0.55,
                label="Hough lines" if idx==0 else None)
    if highlight_segments:
        for idx, seg in enumerate(highlight_segments):
            ax.plot([seg["x1"],seg["x2"]], [seg["y1"],seg["y2"]],
                    color="red", linewidth=2.5, alpha=0.9,
                    label="Selected orientation lines" if idx==0 else None)
    if len(virt) > 0:
        ax.scatter(virt[:,0], virt[:,1], s=45, c="#7b2cbf", marker="x", linewidths=2, label="Virtual camera centers")
    rect_x = [minx, maxx, maxx, minx, minx]
    rect_y = [miny, miny, maxy, maxy, miny]
    ax.plot(rect_x, rect_y, color="#1f7a1f", linewidth=2, label="XY bounding box")
    ax.set_title(f"XOY projection (morph={use_morphology})")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.axis("equal"); ax.grid(True, alpha=0.2); ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# ---------- 虚拟相机生成 ----------
def generate_render_views_grid(images, bbox, target_height):
    merged = dict(images)
    next_id = max(images.keys(), default=0) + 1
    # 取第一个非虚拟相机的 camera_id 作为参考
    ref_cam = next((img for img in images.values() if not img["name"].startswith("virtual")), None)
    if ref_cam is None: return images
    ref_id = ref_cam["camera_id"]
    q_ortho = np.array([0.0, 1.0, 0.0, 0.0])   # 竖直向下
    R_ortho = qvec2rotmat(q_ortho)
    centers = compute_virtual_camera_centers_grid(bbox, target_height)
    count = 1
    for row in range(1, VIRTUAL_GRID_SIZE+1):
        for col in range(1, VIRTUAL_GRID_SIZE+1):
            C = centers[count-1]
            t = -R_ortho.dot(C)
            merged[next_id] = {"qvec": q_ortho, "tvec": t, "camera_id": ref_id,
                               "name": f"virtual_{count}_r{row}_c{col}.png", "points2D": []}
            next_id += 1; count += 1
    return merged

# ---------- 去噪函数 ----------
def denoise_points_z_percentile(pts, lower=2, upper=98):
    if len(pts) == 0:
        return pts
    z = pts[:, 2]
    z_low, z_high = np.percentile(z, [lower, upper])
    mask = (z >= z_low) & (z <= z_high)
    return pts[mask]

# ---------- 完整 BIN 读取（保留点 ID 和追踪数据）----------
def _read_next_bytes(fid, num_bytes, fmt):
    data = fid.read(num_bytes)
    if len(data) < num_bytes:
        return []
    return struct.unpack("<" + fmt, data)


def _read_colmap_bin_full(images_bin, cameras_bin, points_bin):
    """完整读取 COLMAP BIN，保留原始 ID 和追踪数据。返回 images, point_tracks, points3D, cameras。"""
    import struct

    # 读点
    points3D = {}
    with open(points_bin, "rb") as f:
        num_pts = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_pts):
            props = _read_next_bytes(f, 43, "QdddBBBd")
            pid = int(props[0])
            xyz = np.array(props[1:4])
            rgb = (int(props[4]), int(props[5]), int(props[6]))
            error = float(props[7])
            track_len = _read_next_bytes(f, 8, "Q")[0]
            tracks = []
            if track_len > 0:
                track_data = _read_next_bytes(f, 8 * track_len, "i" * (2 * track_len))
                for t in range(track_len):
                    tracks.append((track_data[2*t], track_data[2*t+1]))
            points3D[pid] = {"xyz": xyz, "rgb": rgb, "error": error}
            if tracks:
                # 暂存 tracks 后面构建 point_tracks
                points3D[pid]["_tracks"] = tracks

    # 读相机
    _MODEL_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
    _MODEL_NAMES = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
                    4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV", 7: "FOV",
                    8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE"}
    cameras = {}
    with open(cameras_bin, "rb") as f:
        num_cams = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cams):
            props = _read_next_bytes(f, 24, "iiQQ")
            cid = props[0]
            model_id = props[1]
            w, h = props[2], props[3]
            model_name = _MODEL_NAMES.get(model_id, "PINHOLE")
            n_params = _MODEL_PARAMS.get(model_id, 4)
            params_data = _read_next_bytes(f, 8 * n_params, "d" * n_params)
            cameras[cid] = {"model": model_name, "width": w, "height": h,
                           "params": np.array(params_data)}

    # 读图像
    images = {}
    point_tracks = {}
    with open(images_bin, "rb") as f:
        num_imgs = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_imgs):
            props = _read_next_bytes(f, 64, "idddddddi")
            img_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            cam_id = props[8]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\0" or not ch:
                    break
                name += ch
            name = name.decode()
            n_pts2d = _read_next_bytes(f, 8, "Q")[0]
            pts2d = []
            if n_pts2d > 0:
                pts_data = _read_next_bytes(f, 24 * n_pts2d, "ddq" * n_pts2d)
                for i in range(n_pts2d):
                    x, y = pts_data[3*i], pts_data[3*i+1]
                    pid = int(pts_data[3*i+2])
                    pts2d.append((float(x), float(y), pid))
                    if pid not in point_tracks:
                        point_tracks[pid] = []
                    point_tracks[pid].append((img_id, len(pts2d) - 1))
            images[img_id] = {"qvec": qvec, "tvec": tvec, "camera_id": cam_id,
                             "name": name, "points2D": pts2d}

    # 确保所有点都有 track 条目
    for pid in points3D:
        if pid not in point_tracks:
            point_tracks[pid] = []

    return images, point_tracks, points3D, cameras


# ---------- 主流程 ----------
def main():
    global images_txt, points_txt, cameras_txt
    global final_output_images, final_output_points, final_output_cameras
    global projection_plot_path, output_camerassparse_dir, sparse_dir

    images_txt = os.path.join(sparse_dir, "images.txt")
    points_txt = os.path.join(sparse_dir, "points3D.txt")
    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    final_output_images = os.path.join(output_camerassparse_dir, "images.txt")
    final_output_points = os.path.join(output_camerassparse_dir, "points3D.txt")
    final_output_cameras = os.path.join(output_camerassparse_dir, "cameras.txt")
    projection_plot_path = os.path.join(output_camerassparse_dir, "xoy_projection.png")

    random.seed(0); np.random.seed(0)
    os.makedirs(output_camerassparse_dir, exist_ok=True)
    # 自动检测 BIN / TXT 格式
    images_bin = os.path.join(sparse_dir, "images.bin")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    points_bin = os.path.join(sparse_dir, "points3D.bin")

    if os.path.exists(images_bin) and os.path.exists(cameras_bin):
        print("检测到 BIN 格式，使用 3DGS 读取...")
        try:
            extri = read_extrinsics_binary(images_bin)
            intri = read_intrinsics_binary(cameras_bin)
            # 完整读取 BIN（保留点 ID 和追踪数据）
            images, point_tracks, points3D, cameras = _read_colmap_bin_full(
                images_bin, cameras_bin, points_bin)
        except Exception as e:
            print(f"BIN 读取失败: {e}，回退 TXT...")
            images, point_tracks = read_images_txt(images_txt)
            points3D = read_points3D_txt(points_txt)
            cameras = read_cameras_txt(cameras_txt)
    else:
        print("检测到 TXT 格式...")
        try:
            images, point_tracks = read_images_txt(images_txt)
            points3D = read_points3D_txt(points_txt)
            cameras = read_cameras_txt(cameras_txt)
        except Exception as e:
            print(f"读取失败: {e}"); return

    all_ids = sorted(points3D.keys())
    pts = np.vstack([points3D[pid]["xyz"] for pid in all_ids])

    # ---------- 从相机位置估计地面参考法线 ----------
    cam_positions = []
    for img in images.values():
        if img["name"].startswith("virtual"):
            continue
        R = qvec2rotmat(img["qvec"])
        C = -R.T @ img["tvec"]  # 相机在世界坐标系中的位置
        cam_positions.append(C)
    cam_positions = np.array(cam_positions)

    # SVD 拟合相机位置平面 → 法线应接近重力方向
    cam_center = cam_positions.mean(axis=0)
    _, _, Vt = np.linalg.svd(cam_positions - cam_center)
    cam_plane_normal = normalize(Vt[-1])  # 最小方差方向 = 法线
    # 确保法线朝上（与相机平均高度方向一致）
    if cam_plane_normal[2] < 0:
        cam_plane_normal = -cam_plane_normal
    print(f"相机分布平面法线: {cam_plane_normal}, 相机数: {len(cam_positions)}")

    # ---------- 迭代平面提取：排除墙面直到找到地面 ----------
    ransac_iters = max(200, int(len(pts) * 0.05))
    remaining_pts = pts
    ground_normal = None
    ground_p0 = None
    max_planes = 5

    for plane_idx in range(max_planes):
        if len(remaining_pts) < 100:
            break

        normal_i, p0_i, inliers_i = ransac_plane(remaining_pts, ransac_iters, DIST_THRESHOLD)
        normal_i = normalize(normal_i)
        # 关键：与相机分布平面法线比较
        # 平行 → 地面/天花板；垂直 → 墙面
        dot_cam = abs(np.dot(normal_i, cam_plane_normal))

        if dot_cam > 0.7:  # 与相机平面平行 → 地面或天花板
            ground_normal = normal_i
            ground_p0 = p0_i
            print(f"平面{plane_idx+1}: 地面/天花板 (dot_cam={dot_cam:.2f})")
            break
        else:
            # 墙面：排除内点，继续搜索
            remaining_pts = remaining_pts[~np.isin(np.arange(len(remaining_pts)), inliers_i)]
            print(f"平面{plane_idx+1}: 墙面 (dot_cam={dot_cam:.2f}), 排除 {len(inliers_i)} 点, 剩余 {len(remaining_pts)}")

    if ground_normal is None:
        print("所有平面均为墙面，用底部点估计地面...")
        z_along_n = pts.dot(cam_plane_normal)
        z_low = np.percentile(z_along_n, 10)
        z_high = np.percentile(z_along_n, 90)
        bottom = pts[z_along_n < z_low + 0.3 * (z_high - z_low)]
        if len(bottom) > 100:
            ground_normal, ground_p0, _ = ransac_plane(bottom, ransac_iters, DIST_THRESHOLD)
            ground_normal = normalize(ground_normal)
        else:
            ground_normal = cam_plane_normal
            ground_p0 = np.zeros(3)

    normal = ground_normal
    p0 = ground_p0

    A1_temp = rotation_from_vector_to_z(normal)
    pts_temp = (pts - p0).dot(A1_temp.T)

    pts_temp_clean = denoise_points_z_percentile(pts_temp, lower=2, upper=98)
    z_clean = pts_temp_clean[:, 2]

    z10, z50, z90 = np.percentile(z_clean, [0, 50, 100])
    span_low  = z50 - z10
    span_high = z90 - z50

    print(f"去噪点云: {len(pts_temp_clean)} 点, 低处跨度={span_low:.3f}, 高处跨度={span_high:.3f}")
    is_ceiling = (span_low >= span_high)

    if is_ceiling:
        print("检测为天花板 → 翻转法线，以真实地面为基准。")
        normal = -normal
    else:
        print("检测为地面，保持原法线方向。")

    A1 = rotation_from_vector_to_z(normal)

    if GROUND_POINT_ID is not None and GROUND_POINT_ID in points3D:
        P_ground = points3D[GROUND_POINT_ID]["xyz"].copy()
    else:
        pts_a1 = pts.dot(A1.T)
        ground_ref = compute_lowest_ground_reference(pts_a1)
        P_ground = A1.T.dot(ground_ref)

    # ---------- XY 对齐 ----------
    pts_z = (pts - P_ground).dot(A1.T)
    A2, diag = align_xy_by_outer_boundary_grid(pts_z)
    A_final = A2.dot(A1)

    # ---------- 变换点云与相机 ----------
    new_points = {}
    all_aligned_pts = []
    for pid, pt in points3D.items():
        xyz = A_final.dot(pt["xyz"] - P_ground)
        new_points[pid] = {"xyz": xyz, "rgb": pt["rgb"], "error": pt["error"]}
        all_aligned_pts.append(xyz)
    all_aligned_pts = np.array(all_aligned_pts)

    new_images = {}
    for img_id, img in images.items():
        R_i = qvec2rotmat(img["qvec"])
        C_old = -R_i.T.dot(img["tvec"])
        R_new = R_i.dot(A_final.T)
        C_new = A_final.dot(C_old - P_ground)
        new_images[img_id] = {"qvec": rotmat2qvec(R_new), "tvec": -R_new.dot(C_new),
                              "camera_id": img["camera_id"],
                              "name": img["name"], "points2D": img["points2D"]}

    # ---------- 包围盒与虚拟相机高度 ----------
    if len(all_aligned_pts) > 100:
        minx, maxx, miny, maxy, minz, maxz = get_robust_bbox(all_aligned_pts, 1, 99)
        bbox = (minx, maxx, miny, maxy, minz, maxz)

        target_h = 0.80 * maxz
        print(f"虚拟相机高度 (最高点80%): {target_h:.3f}")

        highlights = []
        if diag["selected_segments"]:
            for seg in diag["selected_segments"]:
                p1 = A2.dot([seg["x1"], seg["y1"], 0])[:2]
                p2 = A2.dot([seg["x2"], seg["y2"], 0])[:2]
                highlights.append({"x1": p1[0], "y1": p1[1],
                                   "x2": p2[0], "y2": p2[1]})

        save_xoy_projection_plot(all_aligned_pts, bbox, target_h,
                                 projection_plot_path, use_morphology=True,
                                 highlight_segments=highlights)
        print("投影图已保存。")
    else:
        bbox = (0,0,0,0,0,0); target_h = 2.5

    merged = generate_render_views_grid(new_images, bbox, target_h)
    write_images_txt(final_output_images, merged)
    write_points3D_txt(final_output_points, new_points, point_tracks)
    write_cameras_txt(final_output_cameras, cameras)

    # 同时输出 BIN 格式
    write_images_bin(os.path.join(output_camerassparse_dir, "images.bin"), merged)
    write_points3D_bin(os.path.join(output_camerassparse_dir, "points3D.bin"), new_points, point_tracks)
    write_cameras_bin(os.path.join(output_camerassparse_dir, "cameras.bin"), cameras)
    print("完成（TXT + BIN）。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成正射虚拟相机并对齐场景")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入 sparse 目录 (如 sparse/0)")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出 sparse 目录 (如 sparse/1)")
    args = parser.parse_args()

    sparse_dir = os.path.abspath(args.input)
    output_camerassparse_dir = os.path.abspath(args.output)
    images_txt = os.path.join(sparse_dir, "images.txt")
    points_txt = os.path.join(sparse_dir, "points3D.txt")
    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    final_output_images = os.path.join(output_camerassparse_dir, "images.txt")
    final_output_points = os.path.join(output_camerassparse_dir, "points3D.txt")
    final_output_cameras = os.path.join(output_camerassparse_dir, "cameras.txt")
    projection_plot_path = os.path.join(output_camerassparse_dir, "xoy_projection.png")

    main()