from __future__ import annotations

from pathlib import Path

from trainer.train import OOM_HINT, parse_args


def test_train_cli_defaults():
    args = parse_args(
        [
            "--data_dir",
            "scene",
            "--result_dir",
            "out",
            "--data_factor",
            "4",
            "--max_steps",
            "15000",
            "--save_ply",
        ]
    )
    assert args.data_factor == 4
    assert args.max_steps == 15000
    assert args.save_ply is True
    assert "OOM" in OOM_HINT or "显存" in OOM_HINT


def test_backend_never_uses_shell_true():
    root = Path(__file__).resolve().parents[1] / "backend"
    hits = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "shell=True" in text:
            hits.append(str(path))
    assert hits == []


def test_packaging_script_keeps_cache_out_of_app_and_zip_hash_outer():
    root = Path(__file__).resolve().parents[1]
    script = (root / "packaging" / "build_portable.ps1").read_text(encoding="utf-8")
    versions = (root / "packaging" / "versions.json").read_text(encoding="utf-8")
    req = (root / "packaging" / "requirements-portable.txt").read_text(encoding="utf-8")
    assert 'foreach ($name in @("backend", "trainer", "web"))' in script
    assert 'foreach ($name in @("backend", "trainer", "web", "packaging"))' not in script
    assert "ARTIFACT_SHA256.txt" in script
    assert "Copy-Item -Force (Join-Path $OutRoot \"SHA256SUMS.txt\") (Join-Path $Stage \"SHA256SUMS.txt\")" not in script
    assert "Assert-NativeExit" in script
    assert "gsplat_wheel_filename" in script
    assert "2.4.1+cu124" in versions
    assert "0.19.1+cu124" in versions
    assert "gsplat-1.5.3+pt24cu124-cp310-cp310-win_amd64.whl" in versions
    assert "62fae62e2cf233233527ba890fd322825476118edd4bd27a4e6cb36b1723003e" in versions
    assert not any(line.strip().startswith("pyceres") for line in req.splitlines())
    assert not any(line.strip().startswith("gsplat==") for line in req.splitlines())
    assert "torch==2.4.1+cu124" in req
    text = (root / "trainer" / "train.py").read_text(encoding="utf-8")
    assert "2.4.1+cu124" in text
    assert "pt24cu124" in text
    assert "torch.cdist" not in text
    assert "call_soon_threadsafe(async_queue.put_nowait, line)" in (
        root / "backend" / "app" / "main.py"
    ).read_text(encoding="utf-8")
    assert "async_queue.put_nowait(line)" not in (
        (root / "backend" / "app" / "main.py").read_text(encoding="utf-8").split("job_logs_sse")[1].split("job_ws")[0]
    )

