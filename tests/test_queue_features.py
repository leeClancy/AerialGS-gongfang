from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.cache import find_ply, find_stats, purge_work_cache
from backend.app.main import create_app
from backend.app.native_dialog import parse_uri_list, unique_dirs
from backend.app.pipeline import FakeRunner, _command_verb
from backend.app.process import CommandResult
from backend.app.projects import resume_stage_name


def _write_jpeg(folder: Path, name: str, color: tuple[int, int, int]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 24), color)
    img.save(folder / name, format="JPEG")


def test_parse_uri_list_and_unique_dirs(tmp_path: Path):
    first = tmp_path / "seq_a"
    second = tmp_path / "seq_b"
    nested = first / "shot.jpg"
    first.mkdir()
    second.mkdir()
    nested.write_bytes(b"x")
    text = "\n".join(
        [
            first.resolve().as_uri(),
            "# comment",
            str(nested),
            f'"{second}"',
            str(first),
        ]
    )
    parsed = parse_uri_list(text)
    assert str(first.resolve()) in parsed
    assert str(second.resolve()) in parsed
    assert parsed.count(str(first.resolve())) == 1
    merged = unique_dirs([str(first), str(second), str(first), str(tmp_path / "missing")])
    assert len(merged) == 2


def test_purge_work_cache_keeps_ply(tmp_path: Path):
    work = tmp_path / "job"
    (work / "images").mkdir(parents=True)
    (work / "colmap").mkdir()
    (work / "gsplat_scene").mkdir()
    ply_dir = work / "train" / "ply"
    ply_dir.mkdir(parents=True)
    ply = ply_dir / "point_cloud.ply"
    ply.write_text("ply\n", encoding="utf-8")
    (work / "train" / "stats.json").write_text('{"num_gaussians": 12}', encoding="utf-8")
    (work / "images" / "000001.jpg").write_bytes(b"x")
    info = purge_work_cache(work)
    kept = find_ply(work)
    stats = find_stats(work)
    assert kept is not None
    assert kept.parent == work / "output"
    assert stats is not None
    assert not (work / "images").exists()
    assert not (work / "colmap").exists()
    assert not (work / "train").exists()
    assert any(path.endswith("point_cloud.ply") for path in info["kept"])


def test_batch_skips_empty_and_clears_cache(tmp_runtime, image_dir, tmp_path: Path):
    other = tmp_path / "seq_b"
    _write_jpeg(other, "a.jpg", (10, 20, 30))
    _write_jpeg(other, "b.jpg", (40, 50, 60))
    _write_jpeg(other, "c.jpg", (70, 80, 90))
    empty = tmp_path / "empty"
    empty.mkdir()
    app = create_app(tmp_runtime, fake_runner=True)
    with TestClient(app) as client:
        dropped = client.post(
            "/api/fs/drop",
            json={"text": image_dir.resolve().as_uri(), "paths": [str(other)]},
        )
        assert dropped.status_code == 200
        assert len(dropped.json()["paths"]) == 2
        created = client.post(
            "/api/batch",
            json={
                "source_dirs": [str(image_dir), str(other), str(empty)],
                "preset": "fast",
                "run_training": True,
                "clear_cache": True,
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        assert len(body["jobs"]) == 2
        assert any("没有支持的图片" in item["error"] for item in body["skipped"])
        project_id = body["jobs"][0]["project_id"]
        for item in body["jobs"]:
            done = None
            for _ in range(80):
                done = client.get(f"/api/jobs/{item['id']}").json()
                if done["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(0.1)
            assert done["status"] == "succeeded", done
        results = client.get(f"/api/projects/{project_id}/results")
        assert results.status_code == 200
        assert results.json()["ply"]
        work = Path(results.json()["work_dir"])
        assert find_ply(work) is not None
        assert not (work / "train").exists()
        listed = client.get("/api/jobs").json()["jobs"]
        assert len(listed) == 2


def test_resume_stage_name_maps_nested_gates():
    job = {
        "status": "failed",
        "current_stage": "undistort_scene",
        "stages": [
            {"name": "prepare_images", "status": "succeeded"},
            {"name": "export_scene", "status": "running"},
            {"name": "undistort_scene", "status": "failed"},
        ],
    }
    assert resume_stage_name(job) == "export_scene"
    assert resume_stage_name({"status": "failed", "current_stage": "feature_match", "stages": []}) == "feature_match"
    assert resume_stage_name({"status": "succeeded", "stages": [{"name": "train_gsplat", "status": "succeeded"}]}) is None


class FailMatchOnce(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def run(self, argv, *, cwd=None, env=None, log=None, cancel_event=None):
        verb = _command_verb(argv)
        if verb in {"exhaustive_matcher", "sequential_matcher"} and not self.failed_once:
            self.failed_once = True
            log = log or (lambda _line: None)
            log("[fake] matcher boom")
            return CommandResult(argv=list(argv), returncode=1, duration_sec=0.01)
        return super().run(argv, cwd=cwd, env=env, log=log, cancel_event=cancel_event)


def test_resume_keeps_cache_and_delete_removes_job(tmp_runtime, image_dir):
    app = create_app(tmp_runtime, fake_runner=True)
    app.state.pipeline.runner = FailMatchOnce()
    with TestClient(app) as client:
        created = client.post(
            "/api/batch",
            json={
                "source_dirs": [str(image_dir)],
                "preset": "fast",
                "run_training": True,
                "clear_cache": True,
            },
        )
        assert created.status_code == 200, created.text
        job_id = created.json()["jobs"][0]["id"]
        project_id = created.json()["jobs"][0]["project_id"]
        done = None
        for _ in range(80):
            done = client.get(f"/api/jobs/{job_id}").json()
            if done["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        assert done["status"] == "failed"
        work = Path(client.get(f"/api/projects/{project_id}").json()["work_dir"])
        assert (work / "images").is_dir()
        assert any((work / "images").iterdir())
        resumed = client.post(f"/api/jobs/{job_id}/resume")
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["resume_stage"] == "feature_match"
        finished = None
        for _ in range(80):
            finished = client.get(f"/api/jobs/{job_id}").json()
            if finished["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        assert finished["status"] == "succeeded", finished
        gone = client.post(f"/api/jobs/{job_id}/delete")
        assert gone.status_code == 200
        assert client.get(f"/api/jobs/{job_id}").status_code == 404
        assert client.get("/api/jobs").json()["jobs"] == []


def test_delete_project_removes_jobs(tmp_runtime, image_dir):
    app = create_app(tmp_runtime, fake_runner=True)
    with TestClient(app) as client:
        created = client.post(
            "/api/batch",
            json={"source_dirs": [str(image_dir)], "preset": "fast", "run_training": False, "clear_cache": False},
        )
        assert created.status_code == 200, created.text
        job_id = created.json()["jobs"][0]["id"]
        project_id = created.json()["jobs"][0]["project_id"]
        for _ in range(80):
            done = client.get(f"/api/jobs/{job_id}").json()
            if done["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        deleted = client.post(f"/api/projects/{project_id}/delete")
        assert deleted.status_code == 200, deleted.text
        assert client.get(f"/api/projects/{project_id}").status_code == 404
        assert client.get(f"/api/jobs/{job_id}").status_code == 404
        assert client.get("/api/projects").json()["projects"] == []
