from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


APP_HOST = "127.0.0.1"
APP_PORT = 8765
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "label": "低",
        "data_factor": 8,
        "max_steps": 7000,
        "strategy": "default",
        "save_steps": [7000],
        "max_splats": 1_800_000,
        "grow_grad2d": 0.0009,
        "sh_degree": 0,
        "coarse_to_fine": True,
        "opacity_reset_every": 0,
        "max_num_features": 2048,
        "max_image_size": 1000,
        "max_num_matches": 8192,
        "note": "最快出预览，细节少",
    },
    "balanced": {
        "label": "中",
        "data_factor": 4,
        "max_steps": 15000,
        "strategy": "default",
        "save_steps": [7000, 15000],
        "max_splats": 4_500_000,
        "grow_grad2d": 0.0002,
        "sh_degree": 1,
        "coarse_to_fine": True,
        "opacity_reset_every": 0,
        "max_num_features": 4096,
        "max_image_size": 1600,
        "max_num_matches": 16384,
        "note": "12GB 显存默认平衡点",
    },
    "quality": {
        "label": "高",
        "data_factor": 2,
        "max_steps": 30000,
        "strategy": "default",
        "save_steps": [15000, 30000],
        "max_splats": 8_400_000,
        "grow_grad2d": 0.0002,
        "sh_degree": 3,
        "coarse_to_fine": True,
        "opacity_reset_every": 3000,
        "max_num_features": 8192,
        "max_image_size": 3200,
        "max_num_matches": 32768,
        "note": "更清晰，更吃显存和时间",
    },
}


def _is_portable_layout(root: Path) -> bool:
    return (root / "runtime" / "python" / "python.exe").is_file() or (
        root / "runtime" / "python" / "python.exe"
    ).is_file()


def keep_usable_path(path: Path) -> Path:
    """Prefer an ASCII path, including directory junctions, over the resolved Unicode target."""
    abs_path = path.absolute()
    try:
        resolved = path.resolve()
    except OSError:
        return abs_path
    if is_ascii_path(resolved):
        return resolved
    if is_ascii_path(abs_path):
        return abs_path
    return resolved


def detect_root() -> Path:
    env = os.environ.get("AERIALGS_ROOT")
    if env:
        return keep_usable_path(Path(env))
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], here.parents[1], Path.cwd()):
        if (candidate / "web" / "index.html").is_file() or (
            candidate / "app" / "web" / "index.html"
        ).is_file():
            return candidate
        if (candidate / "backend" / "app" / "main.py").is_file():
            return candidate
    return Path.cwd().resolve()


def load_versions(root: Path) -> dict[str, Any]:
    for rel in ("packaging/versions.json", "app/packaging/versions.json"):
        path = root / rel
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


@dataclass
class RuntimePaths:
    root: Path
    app_root: Path
    web_dir: Path
    trainer_dir: Path
    data_dir: Path
    work_dir: Path
    runtime_dir: Path
    python_exe: Path
    colmap_exe: Path | None
    glomap_exe: Path | None
    vocab_tree: Path | None
    versions: dict[str, Any] = field(default_factory=dict)
    portable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "app_root": str(self.app_root),
            "web_dir": str(self.web_dir),
            "work_dir": str(self.work_dir),
            "python_exe": str(self.python_exe),
            "colmap_exe": str(self.colmap_exe) if self.colmap_exe else None,
            "glomap_exe": str(self.glomap_exe) if self.glomap_exe else None,
            "vocab_tree": str(self.vocab_tree) if self.vocab_tree else None,
            "portable": self.portable,
        }


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_tool(name: str, runtime_dir: Path, extra: list[Path] | None = None) -> Path | None:
    extra = extra or []
    names = []
    if name.lower() == "colmap":
        names = ["colmap.exe", "COLMAP.exe", "COLMAP.bat", "colmap.bat"]
        search_dirs = [
            runtime_dir / "colmap" / "bin",
            runtime_dir / "colmap",
            runtime_dir / "COLMAP" / "bin",
            runtime_dir / "COLMAP",
        ]
    elif name.lower() == "glomap":
        names = ["glomap.exe", "GLOMAP.exe"]
        search_dirs = [
            runtime_dir / "glomap" / "bin",
            runtime_dir / "glomap",
            runtime_dir / "GLOMAP" / "bin",
            runtime_dir / "GLOMAP",
        ]
    else:
        names = [f"{name}.exe", name]
        search_dirs = [runtime_dir / name / "bin", runtime_dir / name]

    candidates: list[Path] = []
    for folder in search_dirs:
        if folder.is_dir():
            for fname in names:
                candidates.append(folder / fname)
            for found in folder.rglob(names[0]):
                candidates.append(found)
    candidates.extend(extra)
    hit = _first_existing(candidates)
    if hit:
        return hit
    path_env = os.environ.get("PATH", "")
    for item in path_env.split(os.pathsep):
        if not item:
            continue
        for fname in names:
            candidate = Path(item) / fname
            if candidate.is_file():
                return candidate
    return None


def is_ascii_path(path: Path | str) -> bool:
    text = str(path)
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ascii_drive_dir(root: Path, name: str) -> Path:
    drive = root.drive or Path.home().drive or "C:"
    return Path(drive + os.sep) / name


def default_data_dir(root: Path) -> Path:
    specified = os.environ.get("AERIALGS_DATA")
    if specified:
        return Path(specified).expanduser().resolve()
    nested = (root / "data").resolve()
    if is_ascii_path(nested):
        return nested
    return ascii_drive_dir(root, "AerialGS_Projects")


def default_work_dir(data_dir: Path) -> Path:
    specified = os.environ.get("AERIALGS_WORK")
    if specified:
        return Path(specified).expanduser().resolve()
    nested = (data_dir / "work").resolve()
    if is_ascii_path(nested):
        return nested
    return ascii_drive_dir(data_dir, "AerialGS_Projects") / "work"


def prefer_ascii_root(root: Path) -> Path:
    """COLMAP/CUDA tools are unreliable under non-ASCII paths. Junction to a short ASCII root."""
    root = keep_usable_path(root)
    if is_ascii_path(root):
        return root
    if not (root / "runtime" / "python" / "python.exe").is_file():
        return root
    dest = ascii_drive_dir(root, "AerialGS-Portable")
    marker = dest / "runtime" / "python" / "python.exe"
    if marker.is_file():
        try:
            if dest.resolve() == root.resolve():
                return keep_usable_path(dest)
        except OSError:
            pass
        return root
    if dest.exists():
        return root
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(dest), str(root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode == 0 and marker.is_file():
        return keep_usable_path(dest)
    return root


def build_paths(root: Path | None = None) -> RuntimePaths:
    root = prefer_ascii_root(root or detect_root())
    os.environ["AERIALGS_ROOT"] = str(root)
    portable = (root / "runtime" / "python" / "python.exe").is_file()
    app_root = root / "app" if (root / "app" / "backend").is_dir() else root
    web_dir = app_root / "web" if (app_root / "web").is_dir() else root / "web"
    trainer_dir = app_root / "trainer" if (app_root / "trainer").is_dir() else root / "trainer"
    runtime_dir = root / "runtime"
    data_dir = default_data_dir(root)
    work_dir = default_work_dir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    python_exe = runtime_dir / "python" / "python.exe"
    if not python_exe.is_file():
        python_exe = Path(sys.executable)

    colmap_env = os.environ.get("AERIALGS_COLMAP")
    glomap_env = os.environ.get("AERIALGS_GLOMAP")
    vocab_env = os.environ.get("AERIALGS_VOCAB")

    colmap_exe = Path(colmap_env) if colmap_env else resolve_tool("colmap", runtime_dir)
    glomap_exe = Path(glomap_env) if glomap_env else resolve_tool("glomap", runtime_dir)
    vocab_tree = None
    vocab_candidates = [
        Path(vocab_env) if vocab_env else None,
        runtime_dir / "vocab" / "vocab_tree_faiss_flickr100K_words32K.bin",
        app_root / "packaging" / "cache" / "vocab_tree_faiss_flickr100K_words32K.bin",
    ]
    vocab_tree = _first_existing([p for p in vocab_candidates if p is not None])

    return RuntimePaths(
        root=root,
        app_root=app_root,
        web_dir=web_dir,
        trainer_dir=trainer_dir,
        data_dir=data_dir,
        work_dir=work_dir,
        runtime_dir=runtime_dir,
        python_exe=python_exe,
        colmap_exe=colmap_exe,
        glomap_exe=glomap_exe,
        vocab_tree=vocab_tree,
        versions=load_versions(root) or load_versions(app_root),
        portable=portable,
    )


def is_port_available(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def choose_listen_port(host: str, preferred: int, extra: int = 20) -> int:
    """Pick a loopback port. 8765 is commonly taken by Baidu Pinyin on Windows."""
    candidates = [preferred, 18765, *range(preferred + 1, preferred + extra + 1)]
    seen: set[int] = set()
    for port in candidates:
        if port in seen or port <= 0 or port > 65535:
            continue
        seen.add(port)
        if is_port_available(host, port):
            return port
    raise RuntimeError(f"无法在 {host} 上找到可用端口（首选 {preferred}）")
