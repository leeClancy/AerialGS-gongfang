from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from backend.app.config import RuntimePaths, build_paths


@pytest.fixture
def tmp_runtime(tmp_path: Path) -> RuntimePaths:
    colmap = tmp_path / "runtime" / "colmap" / "bin" / "colmap.exe"
    glomap = tmp_path / "runtime" / "glomap" / "bin" / "glomap.exe"
    colmap.parent.mkdir(parents=True)
    glomap.parent.mkdir(parents=True)
    colmap.write_bytes(b"fake")
    glomap.write_bytes(b"fake")
    web = Path(__file__).resolve().parents[1] / "web"
    trainer = Path(__file__).resolve().parents[1] / "trainer"
    data = tmp_path / "data"
    work = tmp_path / "work"
    data.mkdir()
    work.mkdir()
    return RuntimePaths(
        root=tmp_path,
        app_root=Path(__file__).resolve().parents[1],
        web_dir=web,
        trainer_dir=trainer,
        data_dir=data,
        work_dir=work,
        runtime_dir=tmp_path / "runtime",
        python_exe=Path("python"),
        colmap_exe=colmap,
        glomap_exe=glomap,
        vocab_tree=None,
        versions={},
        portable=False,
    )


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "src_中文"
    folder.mkdir()
    for i, name in enumerate(["DJI_10.jpg", "DJI_2.jpg", "DJI_1.jpg"], start=1):
        img = Image.new("RGB", (32, 24), (i * 40, 80, 120))
        img.save(folder / name, format="JPEG")
    return folder
