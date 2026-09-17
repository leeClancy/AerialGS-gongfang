from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ExifTags

from backend.app.config import IMAGE_EXTENSIONS

_DATETIME_TAGS = {"DateTimeOriginal", "DateTimeDigitized", "DateTime"}
_NATURAL_SPLIT = re.compile(r"(\d+)")


def natural_key(name: str) -> list[Any]:
    parts: list[Any] = []
    for token in _NATURAL_SPLIT.split(name):
        if token.isdigit():
            parts.append(int(token))
        else:
            parts.append(token.casefold())
    return parts


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def scan_images(source_dir: str | Path) -> list[Path]:
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"源目录不存在: {root}")
    files = [p for p in root.iterdir() if is_image_file(p)]
    return files


def read_exif_datetime(path: Path) -> str | None:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            tagged = {}
            for key, value in exif.items():
                name = ExifTags.TAGS.get(key, str(key))
                tagged[name] = value
            ifd = getattr(exif, "get_ifd", None)
            if callable(ifd):
                try:
                    from PIL.ExifTags import IFD

                    tagged.update(
                        {
                            ExifTags.TAGS.get(k, str(k)): v
                            for k, v in ifd(IFD.Exif).items()
                        }
                    )
                except Exception:
                    pass
            for name in _DATETIME_TAGS:
                value = tagged.get(name)
                if value:
                    return str(value)
    except Exception:
        return None
    return None


def read_image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None


@dataclass
class ImageRecord:
    source: Path
    name: str
    ascii_name: str
    sort_key: list[Any]
    exif_time: str | None
    width: int | None
    height: int | None
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "name": self.name,
            "ascii_name": self.ascii_name,
            "exif_time": self.exif_time,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
        }


def inspect_images(source_dir: str | Path, sort_mode: str = "filename") -> list[ImageRecord]:
    files = scan_images(source_dir)
    records: list[ImageRecord] = []
    for path in files:
        size = read_image_size(path)
        records.append(
            ImageRecord(
                source=path.resolve(),
                name=path.name,
                ascii_name="",
                sort_key=[],
                exif_time=read_exif_datetime(path),
                width=size[0] if size else None,
                height=size[1] if size else None,
                size_bytes=path.stat().st_size,
            )
        )
    return sort_records(records, sort_mode)


def sort_records(records: Iterable[ImageRecord], sort_mode: str = "filename") -> list[ImageRecord]:
    items = list(records)
    mode = (sort_mode or "filename").strip().lower()
    if mode in {"exif", "exif_time", "time"}:
        items.sort(
            key=lambda rec: (
                rec.exif_time is None,
                rec.exif_time or "",
                natural_key(rec.name),
            )
        )
    else:
        items.sort(key=lambda rec: natural_key(rec.name))
    for index, rec in enumerate(items, start=1):
        rec.ascii_name = f"{index:06d}{rec.source.suffix.lower()}"
        rec.sort_key = natural_key(rec.name)
    return items


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def materialize_work_images(
    records: list[ImageRecord],
    images_dir: Path,
) -> list[dict[str, Any]]:
    mapping = []
    images_dir.mkdir(parents=True, exist_ok=True)
    keep = {rec.ascii_name for rec in records}
    for child in list(images_dir.iterdir()):
        if child.is_file() or child.is_symlink():
            if child.name not in keep:
                child.unlink()
    for rec in records:
        dst = images_dir / rec.ascii_name
        method = link_or_copy(rec.source, dst)
        mapping.append(
            {
                **rec.as_dict(),
                "work_path": str(dst),
                "method": method,
            }
        )
    return mapping


def downscale_images(src_dir: Path, dst_dir: Path, factor: int) -> int:
    if factor <= 1:
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(src_dir.iterdir(), key=lambda p: p.name):
        if not is_image_file(src):
            continue
        with Image.open(src) as img:
            img = img.convert("RGB")
            width = max(1, img.width // factor)
            height = max(1, img.height // factor)
            resized = img.resize((width, height), Image.Resampling.LANCZOS)
            resized.save(dst_dir / src.name, quality=92)
        count += 1
    return count
