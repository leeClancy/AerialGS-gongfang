from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


CACHE_DIR_NAMES = ("images", "colmap", "gsplat_scene", "train")


def find_ply(work: Path) -> Path | None:
    work = Path(work)
    for rel in ("output", "train/ply"):
        folder = work / rel
        if not folder.is_dir():
            continue
        files = sorted(p for p in folder.glob("*.ply") if p.is_file())
        if files:
            return files[-1]
    return None


def find_stats(work: Path) -> Path | None:
    work = Path(work)
    for rel in ("output/stats.json", "train/stats.json"):
        path = work / rel
        if path.is_file():
            return path
    return None


def purge_work_cache(work: Path | str) -> dict[str, Any]:
    """Delete bulky intermediates. Keep PLY/stats under output/ so the disk does not fill up."""
    root = Path(work)
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    ply_dir = root / "train" / "ply"
    if ply_dir.is_dir():
        for ply in sorted(ply_dir.glob("*.ply")):
            dest = output / ply.name
            if ply.resolve() != dest.resolve():
                shutil.copy2(ply, dest)
            kept.append(str(dest))
    stats = root / "train" / "stats.json"
    if stats.is_file():
        dest = output / "stats.json"
        shutil.copy2(stats, dest)
        kept.append(str(dest))
    for name in CACHE_DIR_NAMES:
        folder = root / name
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
    return {"output": str(output), "kept": kept}
