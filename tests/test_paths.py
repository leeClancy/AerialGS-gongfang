from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend.app.config import (
    choose_listen_port,
    default_data_dir,
    default_work_dir,
    is_ascii_path,
    is_port_available,
    keep_usable_path,
    prefer_ascii_root,
)
from backend.app.images import inspect_images, link_or_copy, materialize_work_images


def test_chinese_source_ascii_work_names(tmp_path: Path):
    src = tmp_path / "源图目录"
    src.mkdir()
    Image.new("RGB", (16, 16), "red").save(src / "航拍_1.jpg")
    Image.new("RGB", (16, 16), "blue").save(src / "航拍_2.jpg")
    recs = inspect_images(src, "filename")
    work = tmp_path / "work" / "images"
    mapping = materialize_work_images(recs, work)
    assert all(item["ascii_name"].isascii() for item in mapping)
    assert mapping[0]["ascii_name"] == "000001.jpg"
    assert not is_ascii_path(src)
    assert is_ascii_path(mapping[0]["ascii_name"])
    assert (work / mapping[0]["ascii_name"]).exists()
    assert Path(mapping[0]["source"]).exists()


def test_hardlink_fallback_to_copy(tmp_path: Path):
    src = tmp_path / "a.jpg"
    dst = tmp_path / "out" / "000001.jpg"
    Image.new("RGB", (8, 8), "green").save(src)
    with patch("backend.app.images.os.link", side_effect=OSError("cross-device")):
        method = link_or_copy(src, dst)
    assert method == "copy"
    assert dst.is_file()
    assert src.is_file()


def test_hardlink_success(tmp_path: Path):
    src = tmp_path / "a.jpg"
    dst = tmp_path / "b.jpg"
    Image.new("RGB", (8, 8), "green").save(src)
    method = link_or_copy(src, dst)
    assert method in {"hardlink", "copy"}
    assert dst.is_file()


def test_materialize_removes_stale_frames(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        Image.new("RGB", (8, 8), "white").save(src / name)
    recs = inspect_images(src, "filename")
    work = tmp_path / "images"
    materialize_work_images(recs, work)
    assert {p.name for p in work.iterdir()} == {"000001.jpg", "000002.jpg", "000003.jpg"}
    recs2 = [recs[0]]
    recs2[0].ascii_name = "000001.jpg"
    mapping = materialize_work_images(recs2, work)
    names = {p.name for p in work.iterdir() if p.is_file()}
    assert names == {"000001.jpg"}
    assert mapping[0]["ascii_name"] == "000001.jpg"


def test_choose_listen_port_skips_occupied():
    import socket

    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("127.0.0.1", 0))
    occupied = busy.getsockname()[1]
    try:
        assert not is_port_available("127.0.0.1", occupied)
        picked = choose_listen_port("127.0.0.1", occupied, extra=5)
        assert picked != occupied
        assert is_port_available("127.0.0.1", picked)
    finally:
        busy.close()


def test_default_data_dir_uses_ascii_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AERIALGS_DATA", raising=False)
    monkeypatch.delenv("AERIALGS_WORK", raising=False)
    root = tmp_path / "中文目录"
    root.mkdir()
    data = default_data_dir(root)
    work = default_work_dir(data)
    assert is_ascii_path(data)
    assert is_ascii_path(work)
    assert data.name == "AerialGS_Projects"
    assert work == data / "work"


def test_prefer_ascii_root_without_runtime_stays(tmp_path: Path):
    root = tmp_path / "中文目录"
    root.mkdir()
    assert prefer_ascii_root(root) == root.resolve()


def test_keep_usable_path_prefers_ascii_junction():
    link = Path(r"F:\AerialGS-Portable")
    if not (link / "runtime" / "python" / "python.exe").is_file():
        return
    got = keep_usable_path(link)
    assert is_ascii_path(got)
    assert "AerialGS-Portable" in str(got)
