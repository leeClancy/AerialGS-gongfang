from pathlib import Path

from PIL import Image

from backend.app.images import inspect_images, natural_key, sort_records


def test_natural_filename_sort(tmp_path: Path):
    folder = tmp_path / "imgs"
    folder.mkdir()
    for name in ["DJI_10.jpg", "DJI_2.jpg", "DJI_1.jpg"]:
        Image.new("RGB", (8, 8), "white").save(folder / name)
    recs = inspect_images(folder, "filename")
    assert [r.name for r in recs] == ["DJI_1.jpg", "DJI_2.jpg", "DJI_10.jpg"]
    assert [r.ascii_name for r in recs] == ["000001.jpg", "000002.jpg", "000003.jpg"]


def test_natural_key_units():
    assert natural_key("img2") < natural_key("img10")
    assert natural_key("A_9.JPG") < natural_key("A_10.JPG")


def test_exif_time_sort(tmp_path: Path):
    folder = tmp_path / "imgs"
    folder.mkdir()
    late = folder / "z.jpg"
    early = folder / "a.jpg"
    Image.new("RGB", (8, 8), "white").save(late)
    Image.new("RGB", (8, 8), "white").save(early)
    recs = inspect_images(folder, "filename")
    recs[0].exif_time = "2026:09:17 18:00:00"
    recs[1].exif_time = "2026:09:17 10:00:00"
    ordered = sort_records(recs, "exif_time")
    times = [r.exif_time for r in ordered]
    assert times[0] <= times[1]
