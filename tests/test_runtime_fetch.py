from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.runtime_fetch import RuntimeInstaller, candidate_urls, file_sha256


def test_github_mirrors_are_prefixed():
    versions = {
        "mirrors": {"github_prefixes": ["https://ghfast.top/", "https://gh-proxy.com/"]},
    }
    urls = candidate_urls(
        "https://github.com/colmap/colmap/releases/download/4.2.0/colmap.zip",
        versions=versions,
    )
    assert urls[0].startswith("https://github.com/")
    assert any(item.startswith("https://ghfast.top/https://github.com/") for item in urls)


def test_import_file_verifies_hash(tmp_runtime, tmp_path: Path):
    payload = b"colmap-artifact-bytes-for-test"
    src = tmp_path / "colmap-x64-windows-cuda.zip"
    src.write_bytes(payload)
    installer = RuntimeInstaller(tmp_runtime)
    colmap = next(item for item in installer.artifacts if item.id == "colmap")
    colmap.sha256 = hashlib.sha256(payload).hexdigest()
    colmap.filename = src.name
    status = installer.import_file("colmap", src, resume=False)
    cached = Path(status["cache_dir"]) / src.name
    assert cached.is_file()
    assert file_sha256(cached) == colmap.sha256


def test_import_file_rejects_bad_hash(tmp_runtime, tmp_path: Path):
    src = tmp_path / "colmap-x64-windows-cuda.zip"
    src.write_bytes(b"not-the-real-colmap")
    installer = RuntimeInstaller(tmp_runtime)
    colmap = next(item for item in installer.artifacts if item.id == "colmap")
    colmap.sha256 = "0" * 64
    colmap.filename = src.name
    try:
        installer.import_file("colmap", src, resume=False)
        assert False, "expected hash mismatch"
    except ValueError as exc:
        assert "SHA256" in str(exc)


def test_extract_python_zip(tmp_runtime, tmp_path: Path):
    zpath = tmp_path / "python.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("python.exe", b"fake-python")
        zf.writestr("python310._pth", "python310.zip\n")
    installer = RuntimeInstaller(tmp_runtime)
    python = next(item for item in installer.artifacts if item.id == "python")
    installer._install_artifact(python, zpath)
    assert (tmp_runtime.runtime_dir / "python" / "python.exe").is_file()
    pth = (tmp_runtime.runtime_dir / "python" / "python310._pth").read_text(encoding="ascii")
    assert "import site" in pth


def test_runtime_status_api_fake(tmp_runtime):
    app = create_app(tmp_runtime, fake_runner=True)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.json()["fake"] is True
        status = client.get("/api/runtime/status")
        assert status.status_code == 200
        body = status.json()
        assert "artifacts" in body
        assert body["phase"] in {"idle", "ready"}
        listed = client.post("/api/fs/list", json={"path": str(tmp_runtime.runtime_dir), "include_files": True})
        assert listed.status_code == 200
