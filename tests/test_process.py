from __future__ import annotations

import sys
import threading
import time

import pytest

from backend.app.process import CancelledError, stream_command


def test_stream_python_no_shell():
    result = stream_command(
        [sys.executable, "-c", "print('hello-aerialgs')"],
        timeout=10,
    )
    assert result.returncode == 0
    assert result.argv[0] == sys.executable


def test_cancel_python_process():
    ev = threading.Event()
    logs = []

    def _run():
        try:
            stream_command(
                [sys.executable, "-c", "import time; time.sleep(30); print('done')"],
                log=logs.append,
                cancel_event=ev,
                timeout=20,
            )
        except CancelledError:
            logs.append("cancelled")

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.4)
    ev.set()
    t.join(timeout=10)
    assert not t.is_alive()
    assert "cancelled" in logs
