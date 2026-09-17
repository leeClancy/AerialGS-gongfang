from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from backend.app.config import RuntimePaths, load_versions


USER_AGENT = "AerialGS-Runtime/1.0"
CHUNK = 256 * 1024


def load_artifact_lock(root: Path) -> dict[str, Any]:
    for rel in ("packaging/artifact-lock.json", "app/packaging/artifact-lock.json"):
        path = root / rel
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
    return {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def unique_urls(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def github_mirror_urls(url: str, prefixes: list[str]) -> list[str]:
    if "github.com" not in url:
        return []
    mirrored = []
    for prefix in prefixes:
        prefix = prefix.rstrip("/") + "/"
        mirrored.append(prefix + url)
    return mirrored


def candidate_urls(primary: str, extra: list[str] | None = None, versions: dict[str, Any] | None = None) -> list[str]:
    versions = versions or {}
    prefixes = list((versions.get("mirrors") or {}).get("github_prefixes") or [])
    extras = [str(x) for x in (extra or []) if x]
    return unique_urls([primary, *extras, *github_mirror_urls(primary, prefixes)])


def cache_dir_for(paths: RuntimePaths) -> Path:
    cache = paths.root / "packaging" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def requirements_file(paths: RuntimePaths) -> Path | None:
    for rel in ("packaging/requirements-portable.txt", "app/packaging/requirements-portable.txt"):
        for base in (paths.app_root, paths.root):
            hit = base / rel
            if hit.is_file():
                return hit
    return None


def probe_modules(python_exe: Path, modules: list[str], timeout: float = 45.0) -> bool:
    if not python_exe.is_file():
        return False
    code = "; ".join(f"import {name}" for name in modules)
    try:
        completed = subprocess.run(
            [str(python_exe), "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


@dataclass
class Artifact:
    id: str
    label: str
    filename: str
    urls: list[str]
    sha256: str | None
    kind: str
    required: bool = True
    size_hint: int | None = None
    state: str = "pending"
    received: int = 0
    error: str = ""
    installed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "filename": self.filename,
            "urls": self.urls,
            "url": self.urls[0] if self.urls else "",
            "sha256": self.sha256,
            "kind": self.kind,
            "required": self.required,
            "size_hint": self.size_hint,
            "state": self.state,
            "received": self.received,
            "error": self.error,
            "installed": self.installed,
            "progress": _progress(self.received, self.size_hint),
        }


def _progress(received: int, size_hint: int | None) -> int:
    if not size_hint:
        return 0
    return max(0, min(100, int(received * 100 / size_hint)))


def build_artifacts(paths: RuntimePaths) -> list[Artifact]:
    versions = load_versions(paths.root) or load_versions(paths.app_root)
    lock = load_artifact_lock(paths.root) or load_artifact_lock(paths.app_root)
    pkgs = versions.get("python_packages") or {}
    py = versions.get("python") or {}
    colmap = versions.get("colmap") or {}
    glomap = versions.get("glomap") or {}
    vocab = versions.get("vocab_tree") or {}
    get_pip = versions.get("get_pip") or {}

    def sha(*keys: str) -> str | None:
        for key in keys:
            value = None
            if "." not in key:
                value = lock.get(key)
            if not value:
                continue
            return str(value).lower()
        return None

    artifacts = [
        Artifact(
            id="python",
            label="Python 3.10 嵌入运行时",
            filename=str(py.get("filename") or "python-embed-amd64.zip"),
            urls=candidate_urls(str(py.get("url") or ""), list(py.get("mirrors") or []), versions),
            sha256=(str(py.get("sha256") or sha("python") or "") or None),
            kind="zip_python",
            size_hint=int(py.get("bytes") or 0) or None,
        ),
        Artifact(
            id="get_pip",
            label="get-pip.py",
            filename=str(get_pip.get("filename") or "get-pip.py"),
            urls=candidate_urls(str(get_pip.get("url") or ""), list(get_pip.get("mirrors") or []), versions),
            sha256=(str(get_pip.get("sha256") or sha("get_pip") or "") or None),
            kind="file",
        ),
        Artifact(
            id="colmap",
            label="COLMAP 4.2 CUDA",
            filename=str(colmap.get("filename") or "colmap-x64-windows-cuda.zip"),
            urls=candidate_urls(str(colmap.get("url") or ""), list(colmap.get("mirrors") or []), versions),
            sha256=(str(colmap.get("sha256") or sha("colmap") or "") or None),
            kind="zip_colmap",
            size_hint=int(colmap.get("bytes") or 0) or None,
        ),
        Artifact(
            id="glomap",
            label="独立 GLOMAP 1.2",
            filename=str(glomap.get("filename") or "glomap-x64-windows-cuda.zip"),
            urls=candidate_urls(str(glomap.get("url") or ""), list(glomap.get("mirrors") or []), versions),
            sha256=(str(glomap.get("sha256") or sha("glomap") or "") or None),
            kind="zip_glomap",
            required=False,
            size_hint=int(glomap.get("bytes") or 0) or None,
        ),
        Artifact(
            id="vocab_tree",
            label="COLMAP 词袋（回环检测）",
            filename=str(vocab.get("filename") or "vocab_tree_faiss_flickr100K_words32K.bin"),
            urls=candidate_urls(str(vocab.get("url") or ""), list(vocab.get("mirrors") or []), versions),
            sha256=(str(vocab.get("sha256") or sha("vocab_tree") or "") or None),
            kind="file_vocab",
            required=False,
            size_hint=int(vocab.get("bytes") or 0) or None,
        ),
        Artifact(
            id="torch",
            label="PyTorch 2.4.1 CUDA 12.4",
            filename=str(pkgs.get("torch_wheel_filename") or "torch-2.4.1+cu124-cp310-cp310-win_amd64.whl"),
            urls=candidate_urls(
                str(pkgs.get("torch_wheel_url") or ""),
                list(pkgs.get("torch_wheel_mirrors") or []),
                versions,
            ),
            sha256=(str(pkgs.get("torch_sha256") or "") or None),
            kind="wheel",
            size_hint=int(pkgs.get("torch_bytes") or 0) or None,
        ),
        Artifact(
            id="torchvision",
            label="torchvision 0.19.1 CUDA",
            filename=str(pkgs.get("torchvision_wheel_filename") or "torchvision-0.19.1+cu124-cp310-cp310-win_amd64.whl"),
            urls=candidate_urls(
                str(pkgs.get("torchvision_wheel_url") or ""),
                list(pkgs.get("torchvision_wheel_mirrors") or []),
                versions,
            ),
            sha256=(str(pkgs.get("torchvision_sha256") or "") or None),
            kind="wheel",
        ),
        Artifact(
            id="gsplat",
            label="gsplat 预编译 wheel",
            filename=str(pkgs.get("gsplat_wheel_filename") or "gsplat-1.5.3+pt24cu124-cp310-cp310-win_amd64.whl"),
            urls=candidate_urls(str(pkgs.get("gsplat_wheel_url") or ""), list(pkgs.get("gsplat_wheel_mirrors") or []), versions),
            sha256=(str(pkgs.get("gsplat_sha256") or sha("gsplat") or "") or None),
            kind="wheel_gsplat",
        ),
    ]
    return [item for item in artifacts if item.urls and item.urls[0]]


class RuntimeInstaller:
    def __init__(self, paths: RuntimePaths):
        self.paths = paths
        self.cache = cache_dir_for(paths)
        self.artifacts = build_artifacts(paths)
        self.phase = "idle"
        self.message = ""
        self.log_lines: list[str] = []
        self.error = ""
        self.manual_needed = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._refresh_installed()

    def _log(self, line: str) -> None:
        with self._lock:
            self.log_lines.append(line)
            self.log_lines = self.log_lines[-80:]
            self.message = line

    def _refresh_installed(self, probe_python: bool = False) -> None:
        runtime = self.paths.runtime_dir
        python_exe = runtime / "python" / "python.exe"
        mapping = {
            "python": python_exe.is_file(),
            "get_pip": (self.cache / "get-pip.py").is_file() or python_exe.is_file(),
            "colmap": bool(self.paths.colmap_exe and Path(self.paths.colmap_exe).is_file())
            or any((runtime / "colmap").rglob("colmap.exe")),
            "glomap": bool(self.paths.glomap_exe and Path(self.paths.glomap_exe).is_file())
            or any((runtime / "glomap").rglob("glomap.exe")),
            "vocab_tree": bool(self.paths.vocab_tree and Path(self.paths.vocab_tree).is_file())
            or (runtime / "vocab" / "vocab_tree_faiss_flickr100K_words32K.bin").is_file(),
            "torch": False,
            "torchvision": False,
            "gsplat": False,
        }
        if python_exe.is_file() and (probe_python or self.phase in {"idle", "ready", "needs_manual"}):
            mapping["torch"] = probe_modules(python_exe, ["torch"], timeout=12.0)
            mapping["torchvision"] = probe_modules(python_exe, ["torchvision"], timeout=12.0) if mapping["torch"] else False
            mapping["gsplat"] = probe_modules(python_exe, ["gsplat"], timeout=12.0) if mapping["torch"] else False
        for artifact in self.artifacts:
            if mapping.get(artifact.id):
                artifact.installed = True
                artifact.state = "done"
                artifact.error = ""

    def needed(self) -> bool:
        self._refresh_installed()
        return any(item.required and not item.installed for item in self.artifacts)

    def restart_required(self) -> bool:
        runtime_py = self.paths.runtime_dir / "python" / "python.exe"
        current = Path(self.paths.python_exe)
        if not runtime_py.is_file():
            return False
        if current.resolve() == runtime_py.resolve():
            return False
        return probe_modules(runtime_py, ["torch", "gsplat"])

    def status(self) -> dict[str, Any]:
        with self._lock:
            artifacts = [item.as_dict() for item in self.artifacts]
            failed = [item for item in self.artifacts if item.state == "failed" and item.required]
            return {
                "needed": any(item.required and not item.installed for item in self.artifacts),
                "ready": not any(item.required and not item.installed for item in self.artifacts),
                "phase": self.phase,
                "message": self.message,
                "error": self.error,
                "manual_needed": self.manual_needed or bool(failed),
                "busy": self.phase in {"downloading", "installing", "extracting"},
                "restart_required": self.restart_required(),
                "cache_dir": str(self.cache),
                "runtime_dir": str(self.paths.runtime_dir),
                "artifacts": artifacts,
                "log": list(self.log_lines[-20:]),
            }

    def start(self, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self.status()
            if not force and not self.needed() and not self.restart_required():
                self.phase = "ready"
                return self.status()
            self._cancel.clear()
            self.manual_needed = False
            self.error = ""
            self.phase = "downloading"
            self._thread = threading.Thread(target=self._run, name="aerialgs-runtime-fetch", daemon=True)
            self._thread.start()
        return self.status()

    def cancel(self) -> dict[str, Any]:
        self._cancel.set()
        self._log("已请求取消下载")
        return self.status()

    def import_file(self, artifact_id: str, source: str | Path, resume: bool = True) -> dict[str, Any]:
        artifact = self._find(artifact_id)
        src = Path(source)
        if not src.is_file():
            raise FileNotFoundError(f"找不到文件：{src}")
        dest = self.cache / artifact.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        try:
            self._verify(artifact, dest)
        except ValueError as exc:
            artifact.state = "failed"
            artifact.error = str(exc)
            raise
        artifact.state = "pending"
        artifact.error = ""
        artifact.received = dest.stat().st_size
        self._log(f"已导入 {artifact.label}：{dest.name}")
        if resume and not (self._thread and self._thread.is_alive()):
            self.start(force=True)
        return self.status()

    def _find(self, artifact_id: str) -> Artifact:
        for item in self.artifacts:
            if item.id == artifact_id:
                return item
        raise KeyError(f"未知组件：{artifact_id}")

    def _run(self) -> None:
        try:
            self._refresh_installed()
            for artifact in self.artifacts:
                if self._cancel.is_set():
                    self.phase = "cancelled"
                    self._log("下载已取消")
                    return
                if artifact.installed:
                    continue
                try:
                    cached = self._ensure_cached(artifact)
                    if cached is None:
                        continue
                    self._install_artifact(artifact, cached)
                    artifact.installed = True
                    artifact.state = "done"
                    artifact.error = ""
                except Exception as exc:
                    artifact.state = "failed"
                    artifact.error = str(exc)
                    self._log(f"{artifact.label} 失败：{exc}")
                    if artifact.required:
                        self.manual_needed = True
                        self.error = str(exc)
                        self.phase = "needs_manual"
                        return
            self.phase = "installing"
            self._install_python_packages()
            self._refresh_installed()
            if self.needed():
                self.phase = "needs_manual"
                self.manual_needed = True
                self.error = self.error or "运行时仍不完整，请改用手动下载。"
                return
            self.phase = "ready"
            self._log("运行时已就绪。若当前还是开发环境，请重启以切换到 GPU 运行时。")
        except Exception as exc:
            self.phase = "needs_manual"
            self.manual_needed = True
            self.error = str(exc)
            self._log(f"准备运行时失败：{exc}")

    def _ensure_cached(self, artifact: Artifact) -> Path | None:
        dest = self.cache / artifact.filename
        if dest.is_file():
            try:
                self._verify(artifact, dest)
                artifact.received = dest.stat().st_size
                self._log(f"使用已缓存文件 {artifact.filename}")
                return dest
            except ValueError:
                dest.unlink(missing_ok=True)
        artifact.state = "downloading"
        last_error = "没有可用下载地址"
        for url in artifact.urls:
            if self._cancel.is_set():
                return None
            try:
                self._log(f"下载 {artifact.label}")
                self._log(url)
                self._download(url, dest, artifact)
                self._verify(artifact, dest)
                return dest
            except Exception as exc:
                last_error = str(exc)
                self._log(f"地址失败：{last_error}")
                dest.unlink(missing_ok=True)
        raise RuntimeError(last_error)

    def _download(self, url: str, dest: Path, artifact: Artifact) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=45) as resp, tmp.open("wb") as handle:
            total = int(resp.headers.get("Content-Length") or 0)
            if total:
                artifact.size_hint = total
            received = 0
            while True:
                if self._cancel.is_set():
                    raise RuntimeError("已取消")
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                artifact.received = received
        tmp.replace(dest)

    def _verify(self, artifact: Artifact, path: Path) -> None:
        artifact.state = "verifying"
        if not path.is_file() or path.stat().st_size < 16:
            raise ValueError(f"{artifact.filename} 文件过小或损坏")
        if artifact.sha256:
            actual = file_sha256(path)
            if actual.lower() != artifact.sha256.lower():
                raise ValueError(f"{artifact.filename} SHA256 不匹配")

    def _install_artifact(self, artifact: Artifact, cached: Path) -> None:
        artifact.state = "installing"
        runtime = self.paths.runtime_dir
        runtime.mkdir(parents=True, exist_ok=True)
        if artifact.kind == "zip_python":
            dest = runtime / "python"
            dest.mkdir(parents=True, exist_ok=True)
            _extract_zip(cached, dest)
            _write_python_pth(dest, self.paths)
        elif artifact.kind == "zip_colmap":
            _extract_tool_zip(cached, runtime / "colmap")
        elif artifact.kind == "zip_glomap":
            _extract_tool_zip(cached, runtime / "glomap")
        elif artifact.kind == "file_vocab":
            vocab_dir = runtime / "vocab"
            vocab_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, vocab_dir / artifact.filename)
        elif artifact.kind in {"file", "wheel", "wheel_gsplat"}:
            pass
        self._log(f"已安装 {artifact.label}")

    def _install_python_packages(self) -> None:
        python_exe = self.paths.runtime_dir / "python" / "python.exe"
        if not python_exe.is_file():
            raise RuntimeError("嵌入式 Python 尚未安装")
        if probe_modules(python_exe, ["torch", "gsplat"]):
            return
        get_pip = self.cache / "get-pip.py"
        if get_pip.is_file():
            self._log("安装 pip")
            self._python([str(python_exe), str(get_pip), "--no-warn-script-location"])
        req = requirements_file(self.paths)
        versions = load_versions(self.paths.root) or load_versions(self.paths.app_root)
        indexes = list((versions.get("mirrors") or {}).get("pypi") or [])
        cmd = [str(python_exe), "-m", "pip", "install", "--no-warn-script-location", "--upgrade", "pip"]
        self._python(cmd)
        if req and req.is_file():
            pip_cmd = [
                str(python_exe),
                "-m",
                "pip",
                "install",
                "--no-warn-script-location",
                "--find-links",
                str(self.cache),
            ]
            extra = (versions.get("python_packages") or {}).get("extra_index_url")
            if extra:
                pip_cmd.extend(["--extra-index-url", str(extra)])
            for index in indexes:
                pip_cmd.extend(["--extra-index-url", index])
            pip_cmd.extend(["-r", str(req)])
            self._log("安装 PyTorch / 训练依赖（体积较大，请耐心等待）")
            self._python(pip_cmd)
        gsplat = next((item for item in self.artifacts if item.id == "gsplat"), None)
        if gsplat:
            wheel = self.cache / gsplat.filename
            if wheel.is_file():
                self._log("安装预编译 gsplat")
                self._python(
                    [
                        str(python_exe),
                        "-m",
                        "pip",
                        "install",
                        "--no-warn-script-location",
                        "--force-reinstall",
                        "--no-deps",
                        str(wheel),
                    ]
                )

    def _python(self, argv: list[str]) -> None:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.stdout:
            self._log(completed.stdout.strip().splitlines()[-1][:240])
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "pip 失败").strip().splitlines()
            raise RuntimeError(detail[-1][:400] if detail else "pip 失败")


def _extract_zip(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest)


def _extract_tool_zip(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}_unpacked"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(tmp)
    children = list(tmp.iterdir())
    source = children[0] if len(children) == 1 and children[0].is_dir() else tmp
    for child in source.iterdir():
        target = dest / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
    shutil.rmtree(tmp, ignore_errors=True)


def _write_python_pth(python_dir: Path, paths: RuntimePaths) -> None:
    pth = next(python_dir.glob("python*._pth"), None)
    if pth is None:
        pth = python_dir / "python310._pth"
    app_rel = "..\\..\\app" if (paths.root / "app" / "backend").is_dir() else "..\\.."
    pth.write_text(f"python310.zip\n.\nLib\\site-packages\n{app_rel}\nimport site\n", encoding="ascii")


def launcher_bat(paths: RuntimePaths) -> Path | None:
    for folder in (paths.root, paths.app_root, paths.app_root.parent):
        hits = list(folder.glob("*.bat"))
        if hits:
            return hits[0]
    return None


def schedule_relaunch(paths: RuntimePaths, delay: float = 0.7) -> None:
    bat = launcher_bat(paths)
    if bat is None:
        raise RuntimeError("找不到启动脚本")

    def _go() -> None:
        time.sleep(delay)
        subprocess.Popen(
            ["cmd.exe", "/c", "start", "AerialGS", str(bat)],
            cwd=str(bat.parent),
            close_fds=True,
        )
        os._exit(0)

    threading.Thread(target=_go, name="aerialgs-relaunch", daemon=True).start()
