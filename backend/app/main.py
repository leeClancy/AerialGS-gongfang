from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from io import BytesIO

from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from PIL import Image as PILImage
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.app.cache import find_ply, find_stats
from backend.app.config import PRESETS, RuntimePaths, build_paths
from backend.app.db import Database
from backend.app.diagnostics import collect_diagnostics
from backend.app.native_dialog import parse_uri_list, pick_file, pick_folder, pick_folders, unique_dirs
from backend.app.pipeline import FakeRunner, Pipeline, RealRunner
from backend.app.projects import JobStore, ProjectStore, default_job_params, resume_stage_name
from backend.app.queue import JobQueue
from backend.app.runtime_fetch import RuntimeInstaller, schedule_relaunch


class CreateProjectBody(BaseModel):
    name: str = ""
    source_dir: str
    sort_mode: str = "filename"
    work_dir: str | None = None


class PreviewBody(BaseModel):
    sort_mode: str = "filename"


class CreateJobBody(BaseModel):
    preset: str = "balanced"
    sort_mode: str = "filename"
    camera_model: str = "SIMPLE_RADIAL"
    max_num_features: int = 8192
    match_mode: str = "auto"
    overlap: int = 10
    quadratic_overlap: bool = True
    loop_detection: bool = True
    single_camera: bool = True
    mapper_backend: str = "colmap_global"
    allow_incremental_fallback: bool = True
    min_registration_ratio: float = 0.5
    data_factor: int | None = None
    max_steps: int | None = None
    strategy: str = "default"
    save_ply: bool = True
    run_training: bool = True
    use_gpu: bool = True
    gpu_index: int = 0
    max_splats: int = 4500000
    grow_grad2d: float = 0.0002
    sh_degree: int = 3
    coarse_to_fine: bool = True
    opacity_reset_every: int = 3000
    num_threads: int = 0
    max_image_size: int = 3200
    max_num_matches: int = 32768
    undistort_images: bool = True
    clear_cache: bool = True


class BatchJobBody(CreateJobBody):
    source_dirs: list[str]
    work_dir: str | None = None


class DropPathsBody(BaseModel):
    text: str = ""
    paths: list[str] = Field(default_factory=list)


class RerunBody(BaseModel):
    stage: str = Field(..., min_length=1)


class BrowseQuery(BaseModel):
    path: str
    include_files: bool = False
    suffix: str | None = None


class RuntimeImportBody(BaseModel):
    artifact_id: str
    path: str


def create_app(
    paths: RuntimePaths | None = None,
    *,
    fake_runner: bool | None = None,
) -> FastAPI:
    if fake_runner is None:
        fake_runner = os.environ.get("AERIALGS_FAKE", "0") == "1"
    paths = paths or build_paths()
    db = Database(paths.data_dir / "aerialgs.sqlite")
    projects = ProjectStore(db, paths)
    jobs = JobStore(db)
    recovered = jobs.recover_interrupted()
    runner = FakeRunner() if fake_runner else RealRunner()
    pipeline = Pipeline(paths, projects, jobs, runner=runner)
    queue = JobQueue(pipeline, jobs)
    pipeline.log_sink = queue.emit

    app = FastAPI(title="航拍高斯工坊", version="1.0.0")
    app.state.paths = paths
    app.state.projects = projects
    app.state.jobs = jobs
    app.state.pipeline = pipeline
    app.state.queue = queue
    app.state.fake_runner = fake_runner
    installer = RuntimeInstaller(paths)
    app.state.installer = installer

    @app.on_event("startup")
    async def _startup() -> None:
        queue.bind_loop(asyncio.get_running_loop())
        queue.start()
        if recovered:
            pass
        auto = os.environ.get("AERIALGS_AUTO_FETCH", "1") != "0"
        if not fake_runner and auto and installer.needed():
            installer.start()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "host": "127.0.0.1",
            "portable": paths.portable,
            "fake": fake_runner,
            "runtime_needed": installer.needed() if not fake_runner else False,
        }

    @app.get("/api/runtime/status")
    def runtime_status() -> dict[str, Any]:
        return installer.status()

    @app.post("/api/runtime/install")
    def runtime_install() -> dict[str, Any]:
        return installer.start(force=True)

    @app.post("/api/runtime/cancel")
    def runtime_cancel() -> dict[str, Any]:
        return installer.cancel()

    @app.post("/api/runtime/import")
    def runtime_import(body: RuntimeImportBody) -> dict[str, Any]:
        try:
            return installer.import_file(body.artifact_id, body.path)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/runtime/open-cache")
    def runtime_open_cache() -> dict[str, Any]:
        cache = installer.cache
        cache.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(cache)  # type: ignore[attr-defined]
        except OSError as exc:
            raise HTTPException(500, f"无法打开缓存目录：{exc}") from exc
        return {"ok": True, "path": str(cache)}

    @app.post("/api/runtime/relaunch")
    def runtime_relaunch() -> dict[str, Any]:
        try:
            schedule_relaunch(paths)
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        return {"ok": True, "message": "正在重启并切换到 GPU 运行时"}

    @app.get("/api/diagnostics")
    def diagnostics() -> dict[str, Any]:
        return collect_diagnostics(paths).as_dict()

    @app.get("/api/presets")
    def presets() -> dict[str, Any]:
        return PRESETS

    @app.post("/api/projects")
    def create_project(body: CreateProjectBody) -> dict[str, Any]:
        try:
            return projects.create(
                name=body.name,
                source_dir=body.source_dir,
                sort_mode=body.sort_mode,
                work_dir=body.work_dir,
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/projects")
    def list_projects() -> dict[str, Any]:
        return {"projects": projects.list_projects()}

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        project = projects.get(project_id)
        if not project:
            raise HTTPException(404, "项目不存在")
        project["jobs"] = jobs.list_jobs(project_id)
        return project

    @app.post("/api/projects/{project_id}/preview")
    def preview_project(project_id: str, body: PreviewBody) -> dict[str, Any]:
        try:
            return projects.preview_order(project_id, body.sort_mode)
        except KeyError:
            raise HTTPException(404, "项目不存在")

    @app.post("/api/projects/{project_id}/jobs")
    def create_job(project_id: str, body: CreateJobBody) -> dict[str, Any]:
        if not projects.get(project_id):
            raise HTTPException(404, "项目不存在")
        diag = collect_diagnostics(paths)
        if not fake_runner and not diag.ready_for_tasks:
            raise HTTPException(409, {"message": "硬件或运行时未就绪", "blockers": diag.blockers})
        params = default_job_params(body.model_dump(exclude_none=True))
        job = jobs.create(project_id, params)
        commands = pipeline.preview_commands(project_id, params)
        jobs.update(job["id"], result={"command_preview": commands})
        queue.enqueue(job["id"])
        job = jobs.get(job["id"])
        job["command_preview"] = commands
        job["queue"] = queue.status()
        return job

    @app.post("/api/batch")
    def create_batch(body: BatchJobBody) -> dict[str, Any]:
        diag = collect_diagnostics(paths)
        if not fake_runner and not diag.ready_for_tasks:
            raise HTTPException(409, {"message": "硬件或运行时未就绪", "blockers": diag.blockers})
        payload = body.model_dump(exclude_none=True)
        folders = unique_dirs(list(payload.pop("source_dirs") or []))
        work_base = payload.pop("work_dir", None)
        if not folders:
            raise HTTPException(400, "没有可用的源图文件夹")
        params = default_job_params(payload)
        created: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for folder in folders:
            try:
                job_work = None
                if work_base:
                    slug = Path(folder).name or "job"
                    job_work = str(Path(work_base) / f"{slug}_{uuid.uuid4().hex[:8]}")
                project = projects.create(
                    name="",
                    source_dir=folder,
                    sort_mode=str(params.get("sort_mode") or "filename"),
                    work_dir=job_work,
                )
                job = jobs.create(project["id"], params)
                preview = pipeline.preview_commands(project["id"], params)
                jobs.update(job["id"], result={"command_preview": preview})
                queue.enqueue(job["id"])
                item = jobs.get(job["id"])
                item["source_dir"] = project["source_dir"]
                item["project_name"] = project["name"]
                item["image_count"] = project["image_count"]
                created.append(item)
            except (FileNotFoundError, ValueError, OSError) as exc:
                skipped.append({"source_dir": folder, "error": str(exc)})
        return {"jobs": created, "skipped": skipped, "queue": queue.status()}

    @app.get("/api/jobs")
    def list_jobs(project_id: str | None = None) -> dict[str, Any]:
        items = jobs.list_jobs(project_id)
        for job in items:
            project = projects.get(job["project_id"])
            if project:
                job["source_dir"] = project.get("source_dir")
                job["project_name"] = project.get("name")
        return {"jobs": items, "queue": queue.status()}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        job["queue"] = queue.status()
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        if not jobs.get(job_id):
            raise HTTPException(404, "任务不存在")
        return queue.cancel(job_id)

    @app.post("/api/jobs/{job_id}/rerun")
    def rerun_job(job_id: str, body: RerunBody) -> dict[str, Any]:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        jobs.update(job_id, status="queued", error=None, finished_at=None, current_stage=body.stage)
        queue.enqueue(job_id, rerun_stage=body.stage)
        return jobs.get(job_id)  # type: ignore[return-value]

    @app.post("/api/jobs/{job_id}/resume")
    def resume_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        if job["status"] in {"running", "queued"}:
            raise HTTPException(409, "任务已在队列中")
        stage = resume_stage_name(job)
        jobs.update(job_id, status="queued", error=None, finished_at=None, current_stage=stage)
        queue.enqueue(job_id, rerun_stage=stage)
        job = jobs.get(job_id)
        job["queue"] = queue.status()
        job["resume_stage"] = stage
        return job

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        return queue.drop(job_id)

    @app.get("/api/jobs/{job_id}/logs")
    def job_logs(job_id: str, after: int = 0) -> dict[str, Any]:
        if not jobs.get(job_id):
            raise HTTPException(404, "任务不存在")
        return {"logs": jobs.logs(job_id, after_id=after)}

    @app.get("/api/jobs/{job_id}/logs/stream")
    async def job_logs_sse(job_id: str) -> StreamingResponse:
        if not jobs.get(job_id):
            raise HTTPException(404, "任务不存在")
        loop = asyncio.get_running_loop()
        async_queue: asyncio.Queue[str] = asyncio.Queue()

        def _push(line: str) -> None:
            loop.call_soon_threadsafe(async_queue.put_nowait, line)

        unsub = queue.subscribe(job_id, _push)

        async def _gen():
            try:
                for row in jobs.logs(job_id, after_id=0, limit=2000):
                    yield f"data: {json.dumps(row, ensure_ascii=False)}\n\n"
                while True:
                    line = await async_queue.get()
                    yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
            finally:
                unsub()

        return StreamingResponse(_gen(), media_type="text/event-stream")

    @app.websocket("/api/ws/jobs/{job_id}")
    async def job_ws(websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        if not jobs.get(job_id):
            await websocket.close(code=4404)
            return
        loop = asyncio.get_running_loop()
        async_queue: asyncio.Queue[str] = asyncio.Queue()

        def _push(line: str) -> None:
            loop.call_soon_threadsafe(async_queue.put_nowait, line)

        unsub = queue.subscribe(job_id, _push)
        try:
            for row in jobs.logs(job_id, after_id=0, limit=2000):
                await websocket.send_text(json.dumps(row, ensure_ascii=False))
            while True:
                get_task = asyncio.create_task(async_queue.get())
                recv_task = asyncio.create_task(websocket.receive_text())
                done, pending = await asyncio.wait(
                    {get_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                if get_task in done:
                    await websocket.send_text(json.dumps({"line": get_task.result()}, ensure_ascii=False))
                if recv_task in done:
                    try:
                        recv_task.result()
                    except WebSocketDisconnect:
                        break
        except WebSocketDisconnect:
            pass
        finally:
            unsub()

    @app.get("/api/projects/{project_id}/results")
    def project_results(project_id: str) -> dict[str, Any]:
        project = projects.get(project_id)
        if not project:
            raise HTTPException(404, "项目不存在")
        work = Path(project["work_dir"])
        ply = find_ply(work)
        stats_file = find_stats(work)
        stats = json.loads(stats_file.read_text(encoding="utf-8")) if stats_file else {}
        return {
            "project_id": project_id,
            "work_dir": str(work),
            "stats": stats,
            "ply": str(ply) if ply else None,
            "ply_url": f"/api/projects/{project_id}/ply" if ply else None,
            "viewer_url": f"/viewer.html?src=/api/projects/{project_id}/ply" if ply else None,
            "ply_size": ply.stat().st_size if ply else 0,
        }

    @app.get("/api/projects/{project_id}/thumbs/{index}")
    def project_thumb(project_id: str, index: int) -> Response:
        project = projects.get(project_id)
        if not project:
            raise HTTPException(404, "项目不存在")
        preview_path = Path(project["work_dir"]) / "source_preview.json"
        if not preview_path.is_file():
            raise HTTPException(404, "没有预览清单")
        records = json.loads(preview_path.read_text(encoding="utf-8"))
        if index < 0 or index >= len(records):
            raise HTTPException(404, "序号超出范围")
        source = Path(records[index]["source"])
        if not source.is_file():
            raise HTTPException(404, "原图不存在")
        try:
            with PILImage.open(source) as img:
                frame = img.convert("RGB")
                frame.thumbnail((240, 160))
                buf = BytesIO()
                frame.save(buf, format="JPEG", quality=72)
        except OSError as exc:
            raise HTTPException(400, "无法读取图片") from exc
        return Response(content=buf.getvalue(), media_type="image/jpeg")

    @app.get("/api/projects/{project_id}/ply")
    def project_ply(project_id: str) -> FileResponse:
        project = projects.get(project_id)
        if not project:
            raise HTTPException(404, "项目不存在")
        ply = find_ply(Path(project["work_dir"]))
        if not ply:
            raise HTTPException(404, "还没有 PLY")
        return FileResponse(ply, media_type="application/octet-stream", filename=ply.name)

    @app.post("/api/fs/list")
    def list_fs(body: BrowseQuery) -> dict[str, Any]:
        target = Path(body.path)
        if not target.exists():
            raise HTTPException(404, "路径不存在")
        if target.is_file():
            return {"path": str(target), "type": "file"}
        entries = []
        suffix = (body.suffix or "").lower()
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            is_dir = child.is_dir()
            if not is_dir and not body.include_files:
                continue
            if not is_dir and suffix and child.suffix.lower() != suffix and not child.name.lower().endswith(suffix.lower()):
                continue
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_dir": is_dir,
                }
            )
        return {"path": str(target), "entries": entries}

    @app.post("/api/fs/pick-folders")
    def pick_source_folders() -> dict[str, Any]:
        folders = pick_folders("选择一个或多个源图文件夹")
        return {"paths": folders}

    @app.post("/api/fs/pick-folder")
    def pick_one_folder() -> dict[str, Any]:
        folder = pick_folder("选择文件夹")
        return {"path": folder}

    @app.post("/api/fs/pick-file")
    def pick_one_file() -> dict[str, Any]:
        path = pick_file("选择已下载的文件")
        return {"path": path}

    @app.post("/api/fs/drop")
    def drop_paths(body: DropPathsBody) -> dict[str, Any]:
        merged = list(body.paths or [])
        merged.extend(parse_uri_list(body.text or ""))
        return {"paths": unique_dirs(merged)}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(paths.web_dir / "index.html")

    if paths.web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=paths.web_dir), name="static")

        @app.get("/viewer.html")
        def viewer() -> FileResponse:
            return FileResponse(paths.web_dir / "viewer.html")

        @app.get("/styles.css")
        def css() -> FileResponse:
            return FileResponse(paths.web_dir / "styles.css")

        @app.get("/app.js")
        def js() -> FileResponse:
            return FileResponse(paths.web_dir / "app.js")

        @app.get("/viewer.js")
        def viewer_js() -> FileResponse:
            return FileResponse(paths.web_dir / "viewer.js")

    @app.exception_handler(HTTPException)
    async def http_exc(_, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            return JSONResponse(detail, status_code=exc.status_code)
        return JSONResponse({"message": str(detail)}, status_code=exc.status_code)

    return app
