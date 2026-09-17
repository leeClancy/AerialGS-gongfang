from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def camera_centers_from_w2c(w2c: np.ndarray) -> np.ndarray:
    """w2c (N,4,4) -> camera centers in world (N,3), C = -R^T t."""
    mats = np.asarray(w2c, dtype=np.float64)
    if mats.ndim == 2:
        mats = mats[None, ...]
    rot = mats[:, :3, :3]
    trans = mats[:, :3, 3]
    return np.einsum("nij,ni->nj", rot, -trans)


def apply_similarity_to_points(points: np.ndarray, scale: float, origin: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return pts.astype(np.float64)
    return scale * (pts - np.asarray(origin, dtype=np.float64).reshape(1, 3))


def apply_similarity_to_w2c(w2c: np.ndarray, scale: float, origin: np.ndarray) -> np.ndarray:
    mats = np.asarray(w2c, dtype=np.float64).copy()
    if mats.ndim == 2:
        mats = mats[None, ...]
        squeeze = True
    else:
        squeeze = False
    centers = camera_centers_from_w2c(mats)
    centers_n = scale * (centers - np.asarray(origin, dtype=np.float64).reshape(1, 3))
    rot = mats[:, :3, :3]
    mats[:, :3, 3] = np.einsum("nij,nj->ni", rot, -centers_n)
    return mats[0] if squeeze else mats


def normalize_similarity(
    points: np.ndarray,
    w2c: np.ndarray,
    *,
    target_median: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """把相机中心的中位半径缩放到 target_median，并对点与 w2c 应用同一相似变换。

    X_n = scale * (X - origin)
    返回 (points_n, w2c_n, scene_scale, scale, origin)。
    scene_scale 为归一化后相机中心到其均值的最大距离 * 1.1。
    """
    mats = np.asarray(w2c, dtype=np.float64)
    if mats.ndim == 2:
        mats = mats[None, ...]
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    centers = camera_centers_from_w2c(mats)
    origin = centers.mean(axis=0) if len(centers) else np.zeros(3, dtype=np.float64)
    radii = np.linalg.norm(centers - origin, axis=1) if len(centers) else np.array([1.0])
    median = float(np.median(radii)) if len(radii) else 1.0
    if not np.isfinite(median) or median < 1e-12:
        median = 1.0
    scale = float(target_median / median)
    points_n = apply_similarity_to_points(pts, scale, origin)
    w2c_n = apply_similarity_to_w2c(mats, scale, origin)
    centers_n = camera_centers_from_w2c(w2c_n)
    if len(centers_n) == 0:
        scene_scale = 1.0
    else:
        scene_scale = float(np.linalg.norm(centers_n - centers_n.mean(axis=0), axis=1).max())
        if not np.isfinite(scene_scale) or scene_scale < 1e-3:
            scene_scale = 1.0
        scene_scale *= 1.1
    return points_n, w2c_n, scene_scale, scale, origin


def knn_sq_dists(points: np.ndarray, k: int = 4) -> np.ndarray:
    """CPU KD-tree 第 k 邻域距离平方，不构造 NxN 距离矩阵。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    k_query = int(min(max(k, 1) + 1, n))
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k_query, workers=-1)
    dists = np.asarray(dists, dtype=np.float64)
    if dists.ndim == 1:
        return np.zeros(n, dtype=np.float64)
    return np.square(dists[:, -1])
