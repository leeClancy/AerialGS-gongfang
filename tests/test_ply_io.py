from __future__ import annotations

from pathlib import Path

import numpy as np

from trainer.ply_io import flatten_sh_rest, sh_rest_property_names, write_gaussian_ply


def test_flatten_sh_rest_graphdeco_nk3_to_n3k():
    n, k = 2, 4
    shn = np.zeros((n, k, 3), dtype=np.float32)
    shn[0, 0, 0] = 1.0  # coeff 0, R
    shn[0, 1, 2] = 2.0  # coeff 1, B
    shn[1, 3, 1] = 3.0  # coeff 3, G
    flat = flatten_sh_rest(shn, n)
    assert flat.shape == (n, 3 * k)
    # [N,3,K] flatten: R all K, then G all K, then B all K
    assert flat[0, 0] == 1.0  # R, k0
    assert flat[0, k + 0] == 0.0
    assert flat[0, 2 * k + 1] == 2.0  # B, k1
    assert flat[1, k + 3] == 3.0  # G, k3
    names = sh_rest_property_names(flat.shape[1])
    assert names[0] == "f_rest_0"
    assert names[-1] == f"f_rest_{3 * k - 1}"
    assert names[2 * k + 1] == "f_rest_9"


def test_write_ply_rest_order(tmp_path: Path):
    n = 1
    means = np.zeros((n, 3), dtype=np.float32)
    scales = np.zeros((n, 3), dtype=np.float32)
    rots = np.array([[1, 0, 0, 0]], dtype=np.float32)
    opac = np.zeros((n,), dtype=np.float32)
    sh0 = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    shn = np.zeros((n, 2, 3), dtype=np.float32)
    shn[0, 0, 0] = 9.0
    path = tmp_path / "g.ply"
    write_gaussian_ply(path, means=means, scales=scales, rotations=rots, opacities=opac, sh0=sh0, shn=shn)
    raw = path.read_bytes()
    header, _, body = raw.partition(b"end_header\n")
    assert b"property float f_rest_0" in header
    assert b"property float f_dc_0" in header
    floats = np.frombuffer(body, dtype="<f4")
    # x y z nx ny nz f_dc_0..2 then f_rest (3*2=6) ...
    rest = floats[9:15]
    assert rest.shape == (6,)
    assert rest[0] == 9.0  # R, k0
    assert rest[1] == 0.0
