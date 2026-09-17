from __future__ import annotations

import threading
from pathlib import Path

from backend.app.pipeline import FakeRunner, Pipeline
from backend.app.projects import JobStore, ProjectStore, default_job_params
from backend.app.reconstruction import write_text_model
from backend.app.db import Database


def test_fake_pipeline_selects_largest_and_exports(tmp_runtime, image_dir):
    db = Database(tmp_runtime.data_dir / "t.sqlite")
    projects = ProjectStore(db, tmp_runtime)
    jobs = JobStore(db)
    runner = FakeRunner()
    pipeline = Pipeline(tmp_runtime, projects, jobs, runner=runner)
    project = projects.create(name="p", source_dir=str(image_dir), sort_mode="filename")
    params = default_job_params(
        {
            "preset": "fast",
            "run_training": True,
            "allow_incremental_fallback": False,
            "clear_cache": False,
        }
    )
    job = jobs.create(project["id"], params)
    pipeline.run_job(job["id"], threading.Event())
    done = jobs.get(job["id"])
    assert done["status"] == "succeeded"
    assert any(call[1] == "global_mapper" for call in runner.calls)
    assert any(call[1] == "view_graph_calibrator" for call in runner.calls)
    rec = done["result"]["reconstruction"]
    assert rec["selected"]["num_images"] == 3
    scene = Path(done["result"]["scene_dir"])
    assert (scene / "sparse" / "0" / "images.txt").exists()
    assert (scene / "images_4").exists()


def test_mapper_clears_old_sparse_models(tmp_runtime, image_dir):
    db = Database(tmp_runtime.data_dir / "t3.sqlite")
    projects = ProjectStore(db, tmp_runtime)
    jobs = JobStore(db)
    runner = FakeRunner()
    pipeline = Pipeline(tmp_runtime, projects, jobs, runner=runner)
    project = projects.create(name="p", source_dir=str(image_dir), sort_mode="filename")
    leftover = Path(project["work_dir"]) / "colmap" / "sparse" / "99"
    write_text_model(leftover, image_names=[f"{i:06d}.jpg" for i in range(1, 40)])
    params = default_job_params(
        {
            "preset": "fast",
            "run_training": False,
            "allow_incremental_fallback": False,
            "clear_cache": False,
        }
    )
    job = jobs.create(project["id"], params)
    pipeline.run_job(job["id"], threading.Event())
    done = jobs.get(job["id"])
    assert done["status"] == "succeeded"
    assert not leftover.exists()
    assert done["result"]["reconstruction"]["selected"]["num_images"] == 3


def test_glomap_adapter_command(tmp_runtime, image_dir):
    db = Database(tmp_runtime.data_dir / "t2.sqlite")
    projects = ProjectStore(db, tmp_runtime)
    jobs = JobStore(db)
    runner = FakeRunner()
    pipeline = Pipeline(tmp_runtime, projects, jobs, runner=runner)
    project = projects.create(name="p", source_dir=str(image_dir))
    params = default_job_params({"mapper_backend": "glomap", "run_training": False, "clear_cache": False})
    job = jobs.create(project["id"], params)
    pipeline.run_job(job["id"], threading.Event())
    verbs = [" ".join(c[:2]) for c in runner.calls]
    assert any("glomap" in v.lower() and "mapper" in v for v in verbs)
