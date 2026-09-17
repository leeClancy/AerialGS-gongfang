from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_SUSPENDED = 0x00000004
PROCESS_ALL_ACCESS = 0x001F0FFF
JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

LogFn = Callable[[str], None]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_ulong),
        ("cntUsage", ctypes.c_ulong),
        ("th32ProcessID", ctypes.c_ulong),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", ctypes.c_ulong),
        ("cntThreads", ctypes.c_ulong),
        ("th32ParentProcessID", ctypes.c_ulong),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_ulong),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def _kernel32():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    return k32


class WindowsJob:
    def __init__(self) -> None:
        self._k32 = _kernel32()
        self.handle = self._k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError("CreateJobObjectW 失败")
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self._k32.SetInformationJobObject(
            self.handle,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            raise OSError("SetInformationJobObject 失败")

    def assign(self, pid: int) -> bool:
        proc = self._k32.OpenProcess(PROCESS_ALL_ACCESS, False, int(pid))
        if not proc:
            return False
        try:
            return bool(self._k32.AssignProcessToJobObject(self.handle, proc))
        finally:
            self._k32.CloseHandle(proc)

    def terminate(self, exit_code: int = 1) -> None:
        self._k32.TerminateJobObject(self.handle, exit_code)

    def close(self) -> None:
        if self.handle:
            self._k32.CloseHandle(self.handle)
            self.handle = None


def list_child_pids(root_pid: int) -> list[int]:
    if os.name != "nt":
        return []
    k32 = _kernel32()
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == INVALID_HANDLE_VALUE:
        return []
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    children: dict[int, list[int]] = {}
    try:
        more = k32.Process32FirstW(snap, ctypes.byref(entry))
        while more:
            children.setdefault(entry.th32ParentProcessID, []).append(entry.th32ProcessID)
            more = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    result: list[int] = []
    stack = list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        result.append(pid)
        stack.extend(children.get(pid, []))
    return result


def kill_pid(pid: int) -> None:
    if os.name != "nt":
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        return
    k32 = _kernel32()
    handle = k32.OpenProcess(PROCESS_ALL_ACCESS, False, int(pid))
    if not handle:
        return
    try:
        k32.TerminateProcess(handle, 1)
    finally:
        k32.CloseHandle(handle)


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    duration_sec: float
    cancelled: bool = False


class CancelledError(RuntimeError):
    pass


def stream_command(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    log: LogFn | None = None,
    cancel_event: threading.Event | None = None,
    timeout: float | None = None,
) -> CommandResult:
    if not argv:
        raise ValueError("命令为空")
    argv_list = [str(x) for x in argv]
    log = log or (lambda _line: None)
    creationflags = 0
    job = None
    if os.name == "nt":
        creationflags = CREATE_NEW_PROCESS_GROUP
        try:
            job = WindowsJob()
        except OSError:
            job = None

    started = time.perf_counter()
    proc = subprocess.Popen(
        argv_list,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        shell=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    if job is not None and proc.pid:
        job.assign(proc.pid)

    cancelled = False

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            log(line.rstrip("\r\n"))

    reader = threading.Thread(target=_pump, name="aerialgs-proc-log", daemon=True)
    reader.start()

    deadline = time.monotonic() + timeout if timeout else None
    while proc.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            _terminate_tree(proc, job)
            break
        if deadline is not None and time.monotonic() > deadline:
            cancelled = True
            _terminate_tree(proc, job)
            break
        time.sleep(0.05)

    reader.join(timeout=2)
    returncode = proc.wait(timeout=5) if proc.poll() is None else proc.returncode
    if job is not None:
        try:
            if cancelled:
                job.terminate(1)
        finally:
            job.close()
    duration = time.perf_counter() - started
    if cancelled:
        raise CancelledError(f"已取消: {argv_list[0]}")
    return CommandResult(argv=argv_list, returncode=int(returncode or 0), duration_sec=duration)


def _terminate_tree(proc: subprocess.Popen, job: WindowsJob | None) -> None:
    if job is not None:
        try:
            job.terminate(1)
            return
        except OSError:
            pass
    pid = proc.pid
    if pid:
        for child in reversed(list_child_pids(pid)):
            kill_pid(child)
        kill_pid(pid)
    try:
        proc.terminate()
    except OSError:
        pass
