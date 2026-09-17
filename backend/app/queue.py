from __future__ import annotations

import asyncio
import threading
from collections import defaultdict, deque
from typing import Any, Callable

from backend.app.pipeline import Pipeline, PipelineError
from backend.app.projects import JobStore, utcnow


class JobQueue:
    def __init__(self, pipeline: Pipeline, jobs: JobStore) -> None:
        self.pipeline = pipeline
        self.jobs = jobs
        self._pending: deque[tuple[str, str | None]] = deque()
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, name="aerialgs-queue", daemon=True)
        self._wakeup = threading.Event()
        self._running_id: str | None = None
        self._subscribers: dict[str, list[Callable[[str], None]]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._worker.start()

    def subscribe(self, job_id: str, callback: Callable[[str], None]) -> Callable[[], None]:
        self._subscribers[job_id].append(callback)

        def _unsub() -> None:
            cbs = self._subscribers.get(job_id, [])
            if callback in cbs:
                cbs.remove(callback)

        return _unsub

    def emit(self, job_id: str, line: str) -> None:
        for callback in list(self._subscribers.get(job_id, [])):
            try:
                callback(line)
            except Exception:
                pass
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass

    def enqueue(self, job_id: str, rerun_stage: str | None = None) -> dict[str, Any]:
        with self._lock:
            self._cancel[job_id] = threading.Event()
            self._pending.append((job_id, rerun_stage))
        self._wakeup.set()
        return {"job_id": job_id, "queued": True}

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            event = self._cancel.get(job_id)
            if event is None:
                event = threading.Event()
                self._cancel[job_id] = event
            event.set()
            self._pending = deque(item for item in self._pending if item[0] != job_id)
        job = self.jobs.get(job_id)
        if job and job["status"] == "queued":
            self.jobs.update(job_id, status="cancelled", finished_at=utcnow(), error="已取消")
        return {"job_id": job_id, "cancel_requested": True}

    def drop(self, job_id: str) -> dict[str, Any]:
        self.cancel(job_id)
        self.jobs.delete(job_id)
        return {"job_id": job_id, "deleted": True}

    def status(self) -> dict[str, Any]:
        with self._lock:
            pending = [item[0] for item in self._pending]
            running = self._running_id
        return {"running": running, "queued": pending, "gpu_slots": 1}

    def _loop(self) -> None:
        while True:
            self._wakeup.wait(timeout=0.5)
            self._wakeup.clear()
            with self._lock:
                if not self._pending:
                    continue
                job_id, rerun_stage = self._pending.popleft()
                self._running_id = job_id
                cancel_event = self._cancel.get(job_id) or threading.Event()
            try:
                self.pipeline.log_sink = self.emit
                if not self.jobs.get(job_id):
                    continue
                self.pipeline.run_job(job_id, cancel_event, rerun_stage=rerun_stage)
            except Exception:
                self.emit(job_id, "本任务失败，自动进入队列中的下一个文件夹")
            finally:
                with self._lock:
                    self._running_id = None
                    self._cancel.pop(job_id, None)
                self._wakeup.set()
