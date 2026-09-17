from __future__ import annotations

import time

import numpy as np

from trainer.geometry import (
    apply_similarity_to_points,
    apply_similarity_to_w2c,
    camera_centers_from_w2c,
    knn_sq_dists,
    normalize_similarity,
)


def _w2c(center: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = -center
    return mat


def test_camera_centers_from_w2c():
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    w2c = np.stack([_w2c(c) for c in centers])
    got = camera_centers_from_w2c(w2c)
    np.testing.assert_allclose(got, centers, atol=1e-9)


def test_camera_center_with_rotation():
    center = np.array([3.0, -2.0, 5.0])
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -rotation @ center
    np.testing.assert_allclose(camera_centers_from_w2c(w2c)[0], center, atol=1e-9)


def test_similarity_scales_camera_space_consistently():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, -1.0]])
    w2c = np.stack([_w2c(np.array([10.0, 0.0, 0.0])), _w2c(np.array([-10.0, 0.0, 0.0]))])
    points_n, w2c_n, scene_scale, scale, origin = normalize_similarity(points, w2c)
    assert scale > 0
    assert scene_scale > 0
    np.testing.assert_allclose(origin, np.array([0.0, 0.0, 0.0]), atol=1e-9)

    def cam_coords(mats, pts):
        out = []
        for x in pts:
            out.append(mats[0, :3, :3] @ x + mats[0, :3, 3])
        return np.stack(out)

    old = cam_coords(w2c, points)
    new = cam_coords(w2c_n, points_n)
    np.testing.assert_allclose(new, scale * old, atol=1e-8)
    np.testing.assert_allclose(
        apply_similarity_to_points(points, scale, origin),
        points_n,
        atol=1e-12,
    )
    np.testing.assert_allclose(apply_similarity_to_w2c(w2c, scale, origin), w2c_n, atol=1e-12)


def test_knn_sq_dists_grid_and_not_quadratic_time():
    xs, ys, zs = np.mgrid[0:8, 0:8, 0:8]
    pts = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float64)
    t0 = time.perf_counter()
    d2 = knn_sq_dists(pts, k=1)
    elapsed = time.perf_counter() - t0
    assert d2.shape == (pts.shape[0],)
    np.testing.assert_allclose(np.sqrt(d2), 1.0, atol=1e-6)
    assert elapsed < 2.0
    big = np.random.default_rng(0).normal(size=(8000, 3))
    t1 = time.perf_counter()
    out = knn_sq_dists(big, k=4)
    assert out.shape == (8000,)
    assert np.all(np.isfinite(out))
    assert time.perf_counter() - t1 < 5.0
