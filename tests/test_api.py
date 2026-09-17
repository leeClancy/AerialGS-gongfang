from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.pipeline import FakeRunner
from backend.app.process import CancelledError, CommandResult


class SlowCancelRunner(FakeRunner):
    def run(self, argv, *, cwd=None, env=None, log=None, cancel_event=None):
        log = log or (lambda _line: None)
        log("[slow] waiting for cancel")
        deadline = time.time() + 8
        while time.time() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("slow-cancelled")
            time.sleep(0.05)
        return CommandResult(argv=list(argv), returncode=0, duration_sec=8)


def test_api_create_status_and_fake_job(tmp_runtime, image_dir):
    app = create_app(tmp_runtime, fake_runner=True)
    with TestClient(app) as client:
        diag = client.get("/api/diagnostics")
        assert diag.status_code == 200
        created = client.post(
            "/api/projects",
            json={"name": "demo", "source_dir": str(image_dir), "sort_mode": "filename"},
        )
        assert created.status_code == 200, created.text
        project_id = created.json()["id"]
        preview = client.post(f"/api/projects/{project_id}/preview", json={"sort_mode": "filename"})
        assert preview.status_code == 200
        assert preview.json()["count"] == 3
        job = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"preset": "fast", "run_training": True, "mapper_backend": "colmap_global"},
        )
        assert job.status_code == 200, job.text
        job_id = job.json()["id"]
        body = None
        for _ in range(80):
            body = client.get(f"/api/jobs/{job_id}").json()
            if body["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        assert body["status"] == "succeeded", body
        logs = client.get(f"/api/jobs/{job_id}/logs").json()["logs"]
        assert any("feature_extract" in row["line"] or "GPU" in row["line"] for row in logs)
        results = client.get(f"/api/projects/{project_id}/results")
        assert results.status_code == 200
        assert results.json()["ply"]


def test_api_cancel_running(tmp_runtime, image_dir):
    app = create_app(tmp_runtime, fake_runner=True)
    app.state.pipeline.runner = SlowCancelRunner()
    with TestClient(app) as client:
        created = client.post("/api/projects", json={"name": "c", "source_dir": str(image_dir)})
        project_id = created.json()["id"]
        job = client.post(f"/api/projects/{project_id}/jobs", json={"run_training": False})
        job_id = job.json()["id"]
        for _ in range(50):
            body = client.get(f"/api/jobs/{job_id}").json()
            if body["status"] == "running":
                break
            time.sleep(0.05)
        cancelled = client.post(f"/api/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
        for _ in range(80):
            body = client.get(f"/api/jobs/{job_id}").json()
            if body["status"] in {"cancelled", "failed", "succeeded"}:
                break
            time.sleep(0.1)
        assert body["status"] == "cancelled"
