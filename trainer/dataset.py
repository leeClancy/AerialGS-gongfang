from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from trainer.geometry import normalize_similarity
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _camera_k(model: str, params: list[float], width: int, height: int, factor: int) -> np.ndarray:
    model = model.upper()
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        f, cx, cy = params[:3]
        fx = fy = f
    elif model in {"RADIAL", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    else:
        fx = fy = params[0]
        cx, cy = width / 2, height / 2
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    k[:2] /= float(factor)
    return k


def _load_pycolmap(sparse_dir: Path) -> dict[str, Any]:
    import pycolmap

    rec = pycolmap.Reconstruction(str(sparse_dir))
    images = []
    points = []
    colors = []
    rec_images = rec.images
    rec_cameras = rec.cameras
    if hasattr(rec, "reg_image_ids"):
        image_ids = list(rec.reg_image_ids())
    else:
        image_ids = list(rec_images.keys())
    for image_id in image_ids:
        image = rec_images[image_id]
        name = getattr(image, "name", None) or getattr(image, "name_", None)
        cam_id = int(getattr(image, "camera_id"))
        camera = rec_cameras[cam_id]
        model = str(getattr(camera, "model_name", None) or getattr(camera, "model", "PINHOLE"))
        if hasattr(model, "name"):
            model = model.name
        params = list(np.asarray(camera.params, dtype=np.float64))
        width, height = int(camera.width), int(camera.height)
        if hasattr(image, "cam_from_world") and image.cam_from_world is not None:
            cfw = image.cam_from_world
            rot = getattr(cfw, "rotation", None)
            if rot is not None and hasattr(rot, "quat"):
                quat = np.asarray(rot.quat, dtype=np.float64)
                # pycolmap quat is often x,y,z,w
                if quat.shape[-1] == 4:
                    qvec = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)
                else:
                    qvec = np.array([1, 0, 0, 0], dtype=np.float64)
            elif hasattr(cfw, "matrix"):
                mat = np.asarray(cfw.matrix())
                r = mat[:3, :3]
                tvec = mat[:3, 3]
                qvec = _rotmat_to_qvec(r)
            else:
                qvec = np.array([1, 0, 0, 0], dtype=np.float64)
            tvec = np.asarray(getattr(cfw, "translation"), dtype=np.float64)
        elif hasattr(image, "qvec"):
            qvec = np.asarray(image.qvec, dtype=np.float64)
            tvec = np.asarray(image.tvec, dtype=np.float64)
        else:
            continue
        images.append(
            {
                "name": name,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": cam_id,
                "model": str(model),
                "params": params,
                "width": width,
                "height": height,
            }
        )
    if hasattr(rec, "points3D"):
        for _pid, pt in rec.points3D.items():
            xyz = np.asarray(getattr(pt, "xyz", None) if hasattr(pt, "xyz") else pt.xyz, dtype=np.float64)
            rgb = np.asarray(getattr(pt, "color", [128, 128, 128]), dtype=np.float64)
            points.append(xyz)
            colors.append(rgb / 255.0)
    return {"images": images, "points": np.asarray(points, dtype=np.float32), "colors": np.asarray(colors, dtype=np.float32)}


def _rotmat_to_qvec(r: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    trace = np.trace(r)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1)
        q[0] = 0.25 / s
        q[1] = (r[2, 1] - r[1, 2]) * s
        q[2] = (r[0, 2] - r[2, 0]) * s
        q[3] = (r[1, 0] - r[0, 1]) * s
    else:
        q = np.array([1, 0, 0, 0], dtype=np.float64)
    return q


def _load_fallback(sparse_dir: Path) -> dict[str, Any]:
    from backend.app.reconstruction import load_reconstruction

    summary = load_reconstruction(sparse_dir)
    cameras = {cam.id: cam for cam in summary.cameras}
    images = []
    for img in summary.images:
        cam = cameras[img.camera_id]
        images.append(
            {
                "name": img.name,
                "qvec": np.asarray(img.qvec, dtype=np.float64),
                "tvec": np.asarray(img.tvec, dtype=np.float64),
                "camera_id": img.camera_id,
                "model": cam.model,
                "params": cam.params,
                "width": cam.width,
                "height": cam.height,
            }
        )
    return {
        "images": images,
        "points": np.zeros((0, 3), dtype=np.float32),
        "colors": np.zeros((0, 3), dtype=np.float32),
    }


@dataclass
class Frame:
    image_path: Path
    viewmat: np.ndarray
    K: np.ndarray
    width: int
    height: int
    rgb: np.ndarray | None = None


class ColmapScene:
    def __init__(self, data_dir: str | Path, factor: int = 4) -> None:
        self.data_dir = Path(data_dir)
        self.factor = max(1, int(factor))
        sparse = self.data_dir / "sparse" / "0"
        if not sparse.exists():
            sparse = self.data_dir / "sparse"
        try:
            payload = _load_pycolmap(sparse)
            self.loader = "pycolmap"
        except Exception:
            payload = _load_fallback(sparse)
            self.loader = "backend_parser"
        self.images_meta = payload["images"]
        raw_points = np.asarray(payload["points"], dtype=np.float64).reshape(-1, 3)
        self.colors = payload["colors"]
        w2cs = []
        for meta in self.images_meta:
            rot = _qvec_to_rotmat(np.asarray(meta["qvec"]))
            trans = np.asarray(meta["tvec"], dtype=np.float64).reshape(3)
            view = np.eye(4, dtype=np.float64)
            view[:3, :3] = rot
            view[:3, 3] = trans
            w2cs.append(view)
        if w2cs:
            stacked = np.stack(w2cs, axis=0)
            self.points, stacked, self.scene_scale, self.norm_scale, self.norm_origin = normalize_similarity(
                raw_points, stacked
            )
            self.points = self.points.astype(np.float32)
            w2cs = [stacked[i] for i in range(stacked.shape[0])]
        else:
            self.points = raw_points.astype(np.float32)
            self.scene_scale = 1.0
            self.norm_scale = 1.0
            self.norm_origin = np.zeros(3, dtype=np.float64)
        self._normalized_w2c = w2cs
        self.image_dir = self._resolve_image_dir()
        self.frames = self._build_frames()
        if not self.frames:
            raise RuntimeError(f"未能从 {sparse} 加载已注册图片")

    def _resolve_image_dir(self) -> Path:
        if self.factor > 1:
            candidate = self.data_dir / f"images_{self.factor}"
            if candidate.is_dir():
                return candidate
        return self.data_dir / "images"

    def _build_frames(self) -> list[Frame]:
        frames: list[Frame] = []
        for meta, w2c in zip(self.images_meta, self._normalized_w2c):
            path = self.image_dir / Path(meta["name"]).name
            if not path.is_file():
                fallback = self.data_dir / "images" / Path(meta["name"]).name
                path = fallback if fallback.is_file() else path
            if not path.is_file():
                continue
            k = _camera_k(meta["model"], meta["params"], meta["width"], meta["height"], self.factor)
            width = max(1, int(meta["width"] // self.factor))
            height = max(1, int(meta["height"] // self.factor))
            frames.append(
                Frame(
                    image_path=path,
                    viewmat=np.asarray(w2c, dtype=np.float32),
                    K=k.astype(np.float32),
                    width=width,
                    height=height,
                )
            )
        return frames

    def load_rgb(self, frame: Frame) -> np.ndarray:
        with Image.open(frame.image_path) as img:
            img = img.convert("RGB")
            if img.size != (frame.width, frame.height):
                img = img.resize((frame.width, frame.height), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr
