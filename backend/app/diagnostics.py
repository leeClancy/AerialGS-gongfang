from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.app.config import RuntimePaths, is_ascii_path


@dataclass
class Diagnostics:
    os_name: str
    arch: str
    python_version: str
    loopback_only: bool
    windows_x64: bool
    nvidia_gpu: bool
    gpu_name: str | None = None
    driver_version: str | None = None
    vram_mb: int | None = None
    vram_free_mb: int | None = None
    cpu_count: int = 0
    disk_free_gb: float | None = None
    colmap: dict[str, Any] = field(default_factory=dict)
    glomap: dict[str, Any] = field(default_factory=dict)
    vocab_tree: dict[str, Any] = field(default_factory=dict)
    gsplat: dict[str, Any] = field(default_factory=dict)
    torch: dict[str, Any] = field(default_factory=dict)
    ready_for_tasks: bool = False
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stage_devices: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "os_name": self.os_name,
            "arch": self.arch,
            "python_version": self.python_version,
            "loopback_only": self.loopback_only,
            "windows_x64": self.windows_x64,
            "nvidia_gpu": self.nvidia_gpu,
            "gpu_name": self.gpu_name,
            "driver_version": self.driver_version,
            "vram_mb": self.vram_mb,
            "vram_free_mb": self.vram_free_mb,
            "cpu_count": self.cpu_count,
            "disk_free_gb": self.disk_free_gb,
            "colmap": self.colmap,
            "glomap": self.glomap,
            "vocab_tree": self.vocab_tree,
            "gsplat": self.gsplat,
            "torch": self.torch,
            "ready_for_tasks": self.ready_for_tasks,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "stage_devices": self.stage_devices,
        }


def _run_query(argv: list[str], timeout: float = 8.0) -> str:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip()


def _query_nvidia_smi() -> dict[str, Any]:
    output = _run_query(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return {}
    line = output.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return {}
    try:
        vram = int(float(parts[2]))
        free = int(float(parts[3]))
    except ValueError:
        vram, free = None, None
    return {
        "gpu_name": parts[0],
        "driver_version": parts[1],
        "vram_mb": vram,
        "vram_free_mb": free,
    }


def _probe_import(module: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {module}; print(getattr({module}, '__version__', 'ok'))"],
            capture_output=True,
            text=True,
            timeout=20,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}
    if completed.returncode != 0:
        return {"available": False, "error": (completed.stderr or "")[-400:]}
    return {"available": True, "version": (completed.stdout or "").strip()}


def collect_diagnostics(paths: RuntimePaths, *, allow_cpu_preview: bool = True) -> Diagnostics:
    os_name = platform.system()
    arch = platform.machine()
    windows_x64 = os_name == "Windows" and arch.lower() in {"amd64", "x86_64"}
    nvidia = _query_nvidia_smi()
    diag = Diagnostics(
        os_name=os_name,
        arch=arch,
        python_version=platform.python_version(),
        loopback_only=True,
        windows_x64=windows_x64,
        nvidia_gpu=bool(nvidia),
        gpu_name=nvidia.get("gpu_name"),
        driver_version=nvidia.get("driver_version"),
        vram_mb=nvidia.get("vram_mb"),
        vram_free_mb=nvidia.get("vram_free_mb"),
        cpu_count=os.cpu_count() or 0,
        stage_devices={
            "prepare_images": "CPU",
            "feature_extract": "GPU",
            "feature_match": "GPU",
            "view_graph_calibrator": "CPU",
            "global_mapper": "CPU",
            "glomap_mapper": "CPU",
            "incremental_mapper": "CPU",
            "validate_model": "CPU",
            "undistort_scene": "CPU",
            "export_scene": "CPU",
            "train_gsplat": "GPU",
        },
    )
    try:
        usage = shutil.disk_usage(paths.work_dir)
        diag.disk_free_gb = round(usage.free / (1024**3), 2)
    except OSError:
        diag.disk_free_gb = None

    if not windows_x64:
        diag.blockers.append("本工具仅支持 Windows 10/11 x64。")
    if not nvidia:
        diag.blockers.append("未检测到 NVIDIA GPU / 驱动（nvidia-smi 不可用）。不会静默安装驱动。")
    elif diag.vram_mb is not None and diag.vram_mb < 6000:
        diag.warnings.append("显存低于 6GB，建议使用“快速”预设并增大缩放倍率。")

    if paths.colmap_exe and paths.colmap_exe.exists():
        diag.colmap = {"available": True, "path": str(paths.colmap_exe)}
    else:
        diag.colmap = {"available": False}
        diag.blockers.append(
            "未找到 COLMAP 4.2 CUDA。启动后会自动下载；失败可在弹窗里手动导入，或设置 AERIALGS_COLMAP。"
        )

    if paths.glomap_exe and paths.glomap_exe.exists():
        diag.glomap = {"available": True, "path": str(paths.glomap_exe)}
    else:
        diag.glomap = {"available": False}
        diag.warnings.append("未找到独立 GLOMAP 1.2。默认流水线仍可使用 COLMAP global_mapper。")

    if paths.vocab_tree and paths.vocab_tree.exists():
        diag.vocab_tree = {"available": True, "path": str(paths.vocab_tree)}
    else:
        diag.vocab_tree = {"available": False}
        diag.warnings.append("未找到离线词袋文件，顺序匹配的回环检测可能不可用。")

    diag.torch = _probe_import("torch")
    diag.gsplat = _probe_import("gsplat")
    if not diag.torch.get("available"):
        diag.warnings.append("当前 Python 未安装 PyTorch CUDA。启动后会自动下载；失败可手动导入官方 wheel。")
    if not diag.gsplat.get("available"):
        diag.warnings.append("当前 Python 未安装 gsplat。启动后会自动下载官方预编译 wheel，不会伪造训练。")

    if not is_ascii_path(paths.work_dir):
        diag.warnings.append(
            "工作目录包含非英文字符。已尽量改用盘符根目录 AerialGS_Projects；"
            "若仍告警，请设置 AERIALGS_WORK=C:\\AerialGS-Cache。"
        )

    if diag.disk_free_gb is not None and diag.disk_free_gb < 20:
        diag.warnings.append("工作磁盘剩余空间不足 20GB，大型序列可能失败。")

    diag.ready_for_tasks = not diag.blockers
    if allow_cpu_preview and not diag.nvidia_gpu:
        # UI 仍可打开，但不能启动 GPU 任务
        pass
    return diag


def oom_hint(vram_mb: int | None = None) -> str:
    extra = f"当前检测到显存约 {vram_mb} MB。" if vram_mb else ""
    return (
        "CUDA 显存不足（OOM）。已尝试保留最近检查点。请改用更大的缩放倍率（例如 factor=8）、"
        f"更少的训练步数，或切换“快速”预设。{extra}"
        "不要在 CPU 上训练：本工具是 NVIDIA GPU 专用。"
    )
