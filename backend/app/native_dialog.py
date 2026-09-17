from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import uuid
from ctypes import HRESULT, POINTER, byref, c_void_p
from ctypes.wintypes import DWORD, HWND, LPWSTR
from pathlib import Path
from urllib.parse import unquote, urlparse


FOS_PICKFOLDERS = 0x00000020
FOS_FORCEFILESYSTEM = 0x00000040
FOS_ALLOWMULTISELECT = 0x00000200
FOS_PATHMUSTEXIST = 0x00000800
SIGDN_FILESYSPATH = 0x80058000
CLSCTX_INPROC_SERVER = 0x1
COINIT_APARTMENTTHREADED = 0x2
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NEW_CONSOLE = 0x00000010
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __init__(self, text: str) -> None:
        parsed = uuid.UUID(text)
        self.Data1 = parsed.time_low
        self.Data2 = parsed.time_mid
        self.Data3 = parsed.time_hi_version
        for i, byte in enumerate(parsed.bytes[8:]):
            self.Data4[i] = byte


CLSID_FileOpenDialog = GUID("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")
IID_IFileOpenDialog = GUID("D57C7288-D4AD-4768-BE02-9D969532D960")
IID_IShellItem = GUID("43826D1E-E718-42EE-8E99-B53430935690")
IID_IShellItemArray = GUID("B63EA76D-1F85-456F-A19C-48159EFA858B")


class IUnknown(ctypes.Structure):
    pass


def _vtable(obj, index):
    ptr = ctypes.cast(obj, POINTER(c_void_p)).contents.value
    vtbl = ctypes.cast(ptr, POINTER(c_void_p)).contents.value
    return ctypes.cast(vtbl + index * ctypes.sizeof(c_void_p), POINTER(c_void_p)).contents.value


def normalize_dropped_path(raw: str) -> Path | None:
    text = (raw or "").strip().strip('"').replace("\r", "")
    if not text:
        return None
    if text.startswith("file:"):
        parsed = urlparse(text)
        local = unquote(parsed.path or "")
        if sys.platform == "win32" and local.startswith("/") and len(local) >= 3 and local[2] == ":":
            local = local[1:]
        text = local.replace("/", "\\") if sys.platform == "win32" else local
    path = Path(text)
    try:
        path = path.expanduser()
        if path.exists():
            return path.resolve()
    except OSError:
        return None
    return None


def parse_uri_list(payload: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for line in (payload or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path = normalize_dropped_path(line)
        if path is None:
            continue
        folder = path if path.is_dir() else path.parent
        key = str(folder)
        if key in seen:
            continue
        seen.add(key)
        found.append(key)
    return found


def unique_dirs(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in paths:
        path = normalize_dropped_path(item)
        if path is None or not path.is_dir():
            continue
        key = str(path)
        if key.casefold() in seen:
            continue
        seen.add(key.casefold())
        out.append(key)
    return out


def pick_folders(title: str = "选择一个或多个文件夹", allow_multiple: bool = True) -> list[str]:
    if os.environ.get("AERIALGS_PICKER_CHILD") == "1":
        return _pick_folders_inline(title, allow_multiple=allow_multiple)
    errors: list[str] = []
    for picker in (_spawn_powershell_picker, _spawn_python_picker):
        try:
            return picker("folders" if allow_multiple else "folder", title, allow_multiple)
        except Exception as exc:
            errors.append(f"{picker.__name__}: {exc}")
    try:
        return _pick_folders_inline(title, allow_multiple=allow_multiple)
    except Exception as exc:
        errors.append(str(exc))
        raise RuntimeError("无法打开系统文件夹窗口：" + " | ".join(errors)) from exc


def pick_folder(title: str = "选择文件夹") -> str | None:
    folders = pick_folders(title, allow_multiple=False)
    return folders[0] if folders else None


def pick_file(title: str = "选择文件", filetypes: tuple[tuple[str, str], ...] | None = None) -> str | None:
    if os.environ.get("AERIALGS_PICKER_CHILD") == "1":
        return _tk_pick_file(title, filetypes)
    errors: list[str] = []
    for picker in (_spawn_powershell_picker, _spawn_python_picker):
        try:
            picked = picker("file", title, False)
            return picked[0] if picked else None
        except Exception as exc:
            errors.append(str(exc))
    return _tk_pick_file(title, filetypes)


def _app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _powershell_exe() -> str:
    root = os.environ.get("SystemRoot") or os.environ.get("WINDIR") or r"C:\Windows"
    candidate = Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if candidate.is_file():
        return str(candidate)
    return "powershell.exe"


def _creation_flag_sets() -> list[int]:
    if sys.platform != "win32":
        return [0]
    return [
        CREATE_NEW_CONSOLE | CREATE_BREAKAWAY_FROM_JOB,
        CREATE_NEW_CONSOLE,
        CREATE_NEW_PROCESS_GROUP,
        0,
    ]


def _read_picker_output(out_file: Path, *, want_dirs: bool) -> list[str]:
    if not out_file.is_file():
        return []
    blob = out_file.read_text(encoding="utf-8-sig").strip()
    if not blob:
        return []
    data = json.loads(blob)
    if isinstance(data, list):
        raw = [item for item in data if item]
    elif isinstance(data, str):
        raw = [data]
    else:
        raw = []
    if want_dirs:
        return unique_dirs(raw)
    found: list[str] = []
    for item in raw:
        path = normalize_dropped_path(str(item))
        if path is not None:
            found.append(str(path))
    return found


def _run_picker_process(args: list[str], env: dict[str, str], cwd: str) -> None:
    last_error: Exception | None = None
    si = None
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 1
        try:
            ctypes.windll.user32.AllowSetForegroundWindow(-1)
        except Exception:
            pass
    for flags in _creation_flag_sets():
        try:
            subprocess.run(
                args,
                timeout=3600,
                env=env,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                startupinfo=si,
                creationflags=flags,
            )
            return
        except OSError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error


def _spawn_powershell_picker(mode: str, title: str, allow_multiple: bool) -> list[str]:
    if sys.platform != "win32":
        raise RuntimeError("powershell picker is Windows-only")
    script = Path(__file__).with_name("folder_picker.ps1")
    if not script.is_file():
        raise FileNotFoundError(script)
    handle = tempfile.NamedTemporaryFile(prefix="aerialgs-pick-", suffix=".json", delete=False)
    out_file = Path(handle.name)
    handle.close()
    env = os.environ.copy()
    env["AERIALGS_PICKER_TITLE"] = title
    env["AERIALGS_PICKER_OUT"] = str(out_file)
    env["AERIALGS_PICKER_MULTI"] = "1" if allow_multiple else "0"
    args = [
        _powershell_exe(),
        "-NoLogo",
        "-NoProfile",
        "-STA",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Mode",
        "file" if mode == "file" else "folders",
        "-OutFile",
        str(out_file),
        "-Title",
        title,
    ]
    try:
        _run_picker_process(args, env, str(script.parent))
        return _read_picker_output(out_file, want_dirs=mode != "file")
    finally:
        out_file.unlink(missing_ok=True)


def _spawn_python_picker(mode: str, title: str, allow_multiple: bool) -> list[str]:
    env = os.environ.copy()
    root = str(_app_root())
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH", "")]))
    env["AERIALGS_PICKER_CHILD"] = "1"
    handle = tempfile.NamedTemporaryFile(prefix="aerialgs-pick-", suffix=".json", delete=False)
    out_file = Path(handle.name)
    handle.close()
    env["AERIALGS_PICKER_OUT"] = str(out_file)
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        mode,
        title,
        "1" if allow_multiple else "0",
        str(out_file),
    ]
    try:
        _run_picker_process(args, env, root)
        return _read_picker_output(out_file, want_dirs=mode != "file")
    finally:
        out_file.unlink(missing_ok=True)


def _pick_folders_inline(title: str, allow_multiple: bool = True) -> list[str]:
    _ensure_sta()
    if sys.platform != "win32":
        one = _tk_pick_folder(title)
        return [one] if one else []
    try:
        return _com_pick_folders(title, allow_multiple=allow_multiple)
    except Exception:
        one = _tk_pick_folder(title)
        return [one] if one else []


def _ensure_sta() -> None:
    if sys.platform != "win32":
        return
    ole32 = ctypes.windll.ole32
    hr = ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    if hr not in (0, 1, -2147417850):
        pass


def _tk_pick_file(title: str, filetypes: tuple[tuple[str, str], ...] | None = None) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        types = filetypes or (("所有文件", "*.*"),)
        path = filedialog.askopenfilename(title=title, filetypes=types)
        root.destroy()
        return str(Path(path)) if path else None
    except Exception:
        return None


def _tk_pick_folder(title: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        path = filedialog.askdirectory(title=title, mustexist=True)
        root.destroy()
        return str(Path(path)) if path else None
    except Exception:
        return None


def _com_pick_folders(title: str, allow_multiple: bool = True) -> list[str]:
    ole32 = ctypes.OleDLL("ole32")
    ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    try:
        ctypes.windll.user32.AllowSetForegroundWindow(-1)
    except Exception:
        pass
    dialog = c_void_p()
    hr = ole32.CoCreateInstance(
        byref(CLSID_FileOpenDialog),
        None,
        CLSCTX_INPROC_SERVER,
        byref(IID_IFileOpenDialog),
        byref(dialog),
    )
    if hr != 0 or not dialog.value:
        raise OSError(f"CoCreateInstance failed: {hr}")

    Show = ctypes.WINFUNCTYPE(HRESULT, c_void_p, HWND)(_vtable(dialog, 3))
    SetOptions = ctypes.WINFUNCTYPE(HRESULT, c_void_p, DWORD)(_vtable(dialog, 9))
    GetOptions = ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(DWORD))(_vtable(dialog, 10))
    SetTitle = ctypes.WINFUNCTYPE(HRESULT, c_void_p, LPCWSTR_TYPE := ctypes.c_wchar_p)(_vtable(dialog, 17))
    GetResults = ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(c_void_p))(_vtable(dialog, 26))
    Release = ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(_vtable(dialog, 2))

    try:
        options = DWORD(0)
        GetOptions(dialog, byref(options))
        flags = options.value | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST
        if allow_multiple:
            flags |= FOS_ALLOWMULTISELECT
        SetOptions(dialog, flags)
        SetTitle(dialog, title)
        try:
            ctypes.windll.user32.AllowSetForegroundWindow(-1)
        except Exception:
            pass
        hr = Show(dialog, None)
        if hr != 0:  # cancelled
            return []
        results = c_void_p()
        hr = GetResults(dialog, byref(results))
        if hr != 0 or not results.value:
            return []
        GetCount = ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(DWORD))(_vtable(results, 7))
        GetItemAt = ctypes.WINFUNCTYPE(HRESULT, c_void_p, DWORD, POINTER(c_void_p))(_vtable(results, 8))
        ReleaseArr = ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(_vtable(results, 2))
        count = DWORD(0)
        GetCount(results, byref(count))
        paths: list[str] = []
        for index in range(count.value):
            item = c_void_p()
            if GetItemAt(results, index, byref(item)) != 0 or not item.value:
                continue
            GetDisplayName = ctypes.WINFUNCTYPE(HRESULT, c_void_p, ctypes.c_int, POINTER(LPWSTR))(_vtable(item, 5))
            ReleaseItem = ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(_vtable(item, 2))
            name = LPWSTR()
            if GetDisplayName(item, SIGDN_FILESYSPATH, byref(name)) == 0 and name.value:
                paths.append(name.value)
                ctypes.windll.ole32.CoTaskMemFree(name)
            ReleaseItem(item)
        ReleaseArr(results)
        return unique_dirs(paths)
    finally:
        Release(dialog)


if __name__ == "__main__":
    os.environ["AERIALGS_PICKER_CHILD"] = "1"
    mode = sys.argv[1] if len(sys.argv) > 1 else "folders"
    title = sys.argv[2] if len(sys.argv) > 2 else "选择文件夹"
    multi = (sys.argv[3] if len(sys.argv) > 3 else "1") != "0"
    out_file = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(os.environ.get("AERIALGS_PICKER_OUT") or "")
    _ensure_sta()
    if mode == "file":
        picked = _tk_pick_file(title)
        payload = [picked] if picked else []
    else:
        payload = _pick_folders_inline(title, allow_multiple=multi)
    text = json.dumps(payload, ensure_ascii=False)
    if str(out_file):
        out_file.write_text(text, encoding="utf-8")
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
