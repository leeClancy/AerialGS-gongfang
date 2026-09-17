from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable

CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

CAMERA_MODEL_IDS = {name: model_id for model_id, (name, _) in CAMERA_MODELS.items()}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass
class RegisteredImage:
    id: int
    name: str
    camera_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    num_points2d: int = 0


@dataclass
class ReconstructionSummary:
    path: Path
    cameras: list[Camera] = field(default_factory=list)
    images: list[RegisteredImage] = field(default_factory=list)
    num_points3d: int = 0
    format: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    @property
    def num_images(self) -> int:
        return len(self.images)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "num_cameras": len(self.cameras),
            "num_images": self.num_images,
            "num_points3d": self.num_points3d,
            "format": self.format,
            "warnings": list(self.warnings),
            "image_names": [img.name for img in self.images],
        }


def _read_next_bytes(fid: BinaryIO, num_bytes: int) -> bytes:
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise ValueError("COLMAP 二进制模型不完整")
    return data


def _read_binary_cameras(path: Path) -> list[Camera]:
    cameras: list[Camera] = []
    with path.open("rb") as fid:
        num_cameras = struct.unpack("<Q", _read_next_bytes(fid, 8))[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = struct.unpack(
                "<iiQQ", _read_next_bytes(fid, 24)
            )
            model_name, num_params = CAMERA_MODELS.get(model_id, ("UNKNOWN", 0))
            params = struct.unpack(
                "<" + "d" * num_params, _read_next_bytes(fid, 8 * num_params)
            )
            cameras.append(
                Camera(camera_id, model_name, int(width), int(height), list(params))
            )
    return cameras


def _read_binary_images(path: Path) -> list[RegisteredImage]:
    images: list[RegisteredImage] = []
    with path.open("rb") as fid:
        num_images = struct.unpack("<Q", _read_next_bytes(fid, 8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", _read_next_bytes(fid, 4))[0]
            qvec = struct.unpack("<dddd", _read_next_bytes(fid, 32))
            tvec = struct.unpack("<ddd", _read_next_bytes(fid, 24))
            camera_id = struct.unpack("<i", _read_next_bytes(fid, 4))[0]
            name_chars = []
            while True:
                ch = fid.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                name_chars.append(ch)
            name = b"".join(name_chars).decode("utf-8", errors="replace")
            num_points2d = struct.unpack("<Q", _read_next_bytes(fid, 8))[0]
            fid.seek(num_points2d * 24, 1)
            images.append(
                RegisteredImage(
                    id=int(image_id),
                    name=name,
                    camera_id=int(camera_id),
                    qvec=tuple(qvec),  # type: ignore[arg-type]
                    tvec=tuple(tvec),  # type: ignore[arg-type]
                    num_points2d=int(num_points2d),
                )
            )
    return images


def _read_binary_points_count(path: Path) -> int:
    with path.open("rb") as fid:
        return int(struct.unpack("<Q", _read_next_bytes(fid, 8))[0])


def _parse_text_cameras(path: Path) -> list[Camera]:
    cameras: list[Camera] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cameras.append(
            Camera(
                id=int(parts[0]),
                model=parts[1],
                width=int(parts[2]),
                height=int(parts[3]),
                params=[float(x) for x in parts[4:]],
            )
        )
    return cameras


def _parse_text_images(path: Path) -> list[RegisteredImage]:
    images: list[RegisteredImage] = []
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    index = 0
    while index < len(lines):
        parts = lines[index].split()
        image = RegisteredImage(
            id=int(parts[0]),
            qvec=(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])),
            tvec=(float(parts[5]), float(parts[6]), float(parts[7])),
            camera_id=int(parts[8]),
            name=" ".join(parts[9:]),
        )
        index += 1
        if index < len(lines) and not lines[index].split()[0].lstrip("-").isdigit():
            # POINTS2D line
            coords = lines[index].split()
            image.num_points2d = len(coords) // 3
            index += 1
        elif index < len(lines):
            maybe = lines[index].split()
            if len(maybe) % 3 == 0 and maybe and not Path(maybe[-1]).suffix:
                image.num_points2d = len(maybe) // 3
                index += 1
        images.append(image)
    return images


def _parse_text_points_count(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            count += 1
    return count


def load_reconstruction(model_dir: str | Path) -> ReconstructionSummary:
    folder = Path(model_dir)
    summary = ReconstructionSummary(path=folder)
    bin_cameras = folder / "cameras.bin"
    bin_images = folder / "images.bin"
    txt_cameras = folder / "cameras.txt"
    txt_images = folder / "images.txt"
    if bin_cameras.is_file() and bin_images.is_file():
        summary.format = "bin"
        summary.cameras = _read_binary_cameras(bin_cameras)
        summary.images = _read_binary_images(bin_images)
        points = folder / "points3D.bin"
        if points.is_file():
            summary.num_points3d = _read_binary_points_count(points)
    elif txt_cameras.is_file() and txt_images.is_file():
        summary.format = "txt"
        summary.cameras = _parse_text_cameras(txt_cameras)
        summary.images = _parse_text_images(txt_images)
        points = folder / "points3D.txt"
        if points.is_file():
            summary.num_points3d = _parse_text_points_count(points)
    else:
        summary.warnings.append("未找到 cameras/images 模型文件")
    if len(summary.cameras) > 1:
        summary.warnings.append("检测到多个相机，无人机序列通常应共享单相机内参")
    return summary


def iter_model_dirs(sparse_dir: str | Path) -> list[Path]:
    root = Path(sparse_dir)
    if not root.is_dir():
        return []
    numbered = []
    for child in root.iterdir():
        if child.is_dir() and (
            (child / "images.bin").exists() or (child / "images.txt").exists()
        ):
            numbered.append(child)
    if numbered:
        def sort_key(path: Path) -> tuple[int, str]:
            try:
                return (0, f"{int(path.name):08d}")
            except ValueError:
                return (1, path.name)

        return sorted(numbered, key=sort_key)
    if (root / "images.bin").exists() or (root / "images.txt").exists():
        return [root]
    return []


def select_largest_model(
    sparse_dir: str | Path,
    *,
    total_images: int | None = None,
    min_registration_ratio: float = 0.5,
) -> dict[str, Any]:
    models = []
    for folder in iter_model_dirs(sparse_dir):
        summary = load_reconstruction(folder)
        models.append(summary)
    if not models:
        return {
            "ok": False,
            "selected": None,
            "models": [],
            "warnings": ["没有可用的稀疏重建模型"],
            "block_training": True,
        }
    models.sort(key=lambda item: (item.num_images, item.num_points3d), reverse=True)
    selected = models[0]
    ratio = None
    warnings = list(selected.warnings)
    if total_images:
        ratio = selected.num_images / max(total_images, 1)
        if ratio < min_registration_ratio:
            warnings.append(
                f"注册率 {ratio:.1%} 低于阈值 {min_registration_ratio:.0%}，不建议继续训练"
            )
    if selected.num_images < 3:
        warnings.append("注册图片过少，无法进行稳定的高斯训练")
    block = bool(
        selected.num_images < 3
        or (ratio is not None and ratio < min_registration_ratio)
    )
    return {
        "ok": not block,
        "selected": selected.as_dict(),
        "registration_ratio": ratio,
        "models": [item.as_dict() for item in models],
        "warnings": warnings,
        "block_training": block,
    }


def write_text_model(
    model_dir: str | Path,
    *,
    image_names: Iterable[str],
    width: int = 1920,
    height: int = 1080,
    num_points: int = 8,
) -> Path:
    folder = Path(model_dir)
    folder.mkdir(parents=True, exist_ok=True)
    names = list(image_names)
    cameras = folder / "cameras.txt"
    images = folder / "images.txt"
    points = folder / "points3D.txt"
    cameras.write_text(
        "# Camera list\n"
        f"1 SIMPLE_RADIAL {width} {height} 1000 {width / 2:.1f} {height / 2:.1f} 0.02\n",
        encoding="utf-8",
    )
    image_lines = ["# Image list"]
    for idx, name in enumerate(names, start=1):
        image_lines.append(f"{idx} 1 0 0 0 0 0 {idx * 0.1:.3f} 1 {name}")
        image_lines.append("")
    images.write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    point_lines = ["# 3D point list"]
    for idx in range(1, num_points + 1):
        point_lines.append(f"{idx} {idx * 0.2:.3f} 0.0 0.0 128 128 128 1.0")
    points.write_text("\n".join(point_lines) + "\n", encoding="utf-8")
    return folder


def copy_model(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
        "project.ini",
    ):
        file = src / name
        if file.is_file():
            (dst / name).write_bytes(file.read_bytes())
