from __future__ import annotations

from pathlib import Path

import numpy as np


def flatten_sh_rest(shn: np.ndarray | None, n: int) -> np.ndarray:
    """graphdeco / 标准 3DGS：SH rest 按 [N, 3, K] 再 flatten，而不是 [N, K, 3]。"""
    if shn is None:
        return np.zeros((n, 0), dtype=np.float32)
    arr = np.asarray(shn, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((n, 0), dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != n:
            raise ValueError(f"shn N={arr.shape[0]} 与 means N={n} 不一致")
        if arr.shape[-1] == 3:
            arr = np.transpose(arr, (0, 2, 1))
        return np.ascontiguousarray(arr.reshape(n, -1))
    if arr.ndim == 2:
        if arr.shape[0] != n:
            raise ValueError(f"shn N={arr.shape[0]} 与 means N={n} 不一致")
        return np.ascontiguousarray(arr.astype(np.float32, copy=False))
    raise ValueError(f"不支持的 shn 形状 {arr.shape}")


def sh_rest_property_names(rest_dim: int) -> list[str]:
    return [f"f_rest_{i}" for i in range(rest_dim)]


def write_gaussian_ply(
    path: str | Path,
    *,
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    opacities: np.ndarray,
    sh0: np.ndarray,
    shn: np.ndarray | None = None,
) -> None:
    """写出标准 3DGS PLY（Inria / graphdeco 属性布局）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(means.shape[0])
    rest = flatten_sh_rest(shn, n)
    rest_dim = int(rest.shape[1])
    props = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]
    for name in sh_rest_property_names(rest_dim):
        props.append((name, "f4"))
    props.extend(
        [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
    )
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
    ]
    for name, _ in props:
        header.append(f"property float {name}")
    header.append("end_header")
    path.write_bytes(("\n".join(header) + "\n").encode("ascii"))
    zeros = np.zeros((n, 3), dtype=np.float32)
    chunks = [
        means.astype(np.float32).reshape(n, 3),
        zeros,
        np.asarray(sh0, dtype=np.float32).reshape(n, 3),
    ]
    if rest_dim:
        chunks.append(rest.astype(np.float32).reshape(n, rest_dim))
    chunks.extend(
        [
            opacities.astype(np.float32).reshape(n, 1),
            scales.astype(np.float32).reshape(n, 3),
            rotations.astype(np.float32).reshape(n, 4),
        ]
    )
    blob = np.concatenate(chunks, axis=1).astype("<f4", copy=False)
    with path.open("ab") as fh:
        fh.write(blob.tobytes())
