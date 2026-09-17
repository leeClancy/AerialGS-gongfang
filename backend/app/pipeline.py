from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Protocol

from backend.app import commands
from backend.app.cache import purge_work_cache
from backend.app.config import RuntimePaths
from backend.app.diagnostics import oom_hint
from backend.app.images import downscale_images
from backend.app.process import CancelledError, CommandResult, stream_command
from backend.app.projects import JobStore, ProjectStore, utcnow, RESUME_GATE
from backend.app.reconstruction import copy_model, select_largest_model, write_text_model

STAGE_DEVICES = {
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
}


class CommandRunner(Protocol):
    def run(
        self,
        argv: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        log: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> CommandResult: ...


class RealRunner:
    def run(
        self,
        argv: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        log: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> CommandResult:
        return stream_command(argv, cwd=cwd, env=env, log=log, cancel_event=cancel_event)


class FakeRunner:
    """测试用假进程：构造真实输出目录，不调用 COLMAP / gsplat。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(
        self,
        argv: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        log: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> CommandResult:
        self.calls.append(list(argv))
        log = log or (lambda _line: None)
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("fake-cancelled")
        verb = _command_verb(argv)
        log(f"[fake] {verb}")
        if verb == "feature_extractor":
            db = Path(_option(argv, "--database_path"))
            db.parent.mkdir(parents=True, exist_ok=True)
            db.write_bytes(b"SQLite format 3\x00fake")
        elif verb in {"exhaustive_matcher", "sequential_matcher", "view_graph_calibrator"}:
            db = Path(_option(argv, "--database_path"))
            db.parent.mkdir(parents=True, exist_ok=True)
            db.touch()
        elif verb in {"global_mapper", "mapper", "glomap_mapper"}:
            output = Path(_option(argv, "--output_path"))
            images = Path(_option(argv, "--image_path"))
            names = [p.name for p in images.iterdir() if p.is_file()] if images.is_dir() else ["000001.jpg"]
            write_text_model(output / "0", image_names=names)
            if len(names) > 4:
                write_text_model(output / "1", image_names=names[: max(1, len(names) // 4)])
        elif verb == "image_undistorter":
            output = Path(_option(argv, "--output_path"))
            images = Path(_option(argv, "--image_path"))
            model = Path(_option(argv, "--input_path"))
            (output / "images").mkdir(parents=True, exist_ok=True)
            for src in images.iterdir():
                if src.is_file():
                    shutil.copy2(src, output / "images" / src.name)
            copy_model(model, output / "sparse")
        elif verb == "train.py" or (len(argv) >= 2 and argv[1].endswith("train.py")):
            result_dir = Path(_option(argv, "--result_dir"))
            ply_dir = result_dir / "ply"
            ply_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "ckpts").mkdir(parents=True, exist_ok=True)
            (ply_dir / "point_cloud.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
            (result_dir / "stats.json").write_text(
                json.dumps({"fake": True, "num_gaussians": 0, "note": "test stub, not real training"}),
                encoding="utf-8",
            )
        return CommandResult(argv=list(argv), returncode=0, duration_sec=0.01)


def _command_verb(argv: list[str]) -> str:
    if not argv:
        return ""
    name = Path(argv[0]).name.lower()
    if name.startswith("glomap"):
        return "glomap_mapper" if len(argv) > 1 and argv[1] == "mapper" else name
    if len(argv) > 1 and not argv[1].startswith("-"):
        return Path(argv[1]).name
    return name


def _option(argv: list[str], flag: str) -> str:
    for idx, item in enumerate(argv):
        if item == flag and idx + 1 < len(argv):
            return argv[idx + 1]
    raise KeyError(flag)


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


class PipelineError(RuntimeError):
    def __init__(self, message: str, *, stage: str | None = None) -> None:
        super().__init__(message)
        self.stage = stage


class Pipeline:
    def __init__(
        self,
        paths: RuntimePaths,
        projects: ProjectStore,
        jobs: JobStore,
        runner: CommandRunner | None = None,
        log_sink: Callable[[str, str], None] | None = None,
    ) -> None:
        self.paths = paths
        self.projects = projects
        self.jobs = jobs
        self.runner = runner or RealRunner()
        self.log_sink = log_sink or (lambda _job_id, _line: None)

    def preview_commands(self, project_id: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        project = self.projects.get(project_id)
        if not project:
            raise KeyError(project_id)
        pairs = self._planned_commands(project, params)
        return [
            {
                "stage": name,
                "device": STAGE_DEVICES[name],
                "argv": argv,
                "display": commands.subprocess_display(argv),
            }
            for name, argv in pairs
        ]

    def _planned_commands(self, project: dict[str, Any], params: dict[str, Any]) -> list[tuple[str, list[str]]]:
        work = Path(project["work_dir"])
        images_dir = work / "images"
        colmap_dir = work / "colmap"
        sparse_dir = colmap_dir / "sparse"
        scene_dir = work / "gsplat_scene"
        train_dir = work / "train"
        database = colmap_dir / "database.db"
        database_global = colmap_dir / "database_global.db"
        image_count = 0
        if images_dir.is_dir():
            image_count = len([p for p in images_dir.iterdir() if p.is_file()])
        if image_count <= 0:
            try:
                from backend.app.images import scan_images

                image_count = len(scan_images(project["source_dir"]))
            except Exception:
                image_count = 0
        colmap = self.paths.colmap_exe or "colmap"
        pairs: list[tuple[str, list[str]]] = [
            (
                "feature_extract",
                commands.feature_extractor_cmd(
                    colmap,
                    database,
                    images_dir,
                    camera_model=params.get("camera_model", "SIMPLE_RADIAL"),
                    max_num_features=int(params.get("max_num_features", 8192)),
                    use_gpu=bool(params.get("use_gpu", True)),
                    gpu_index=int(params.get("gpu_index", 0)),
                    single_camera=bool(params.get("single_camera", True)),
                    max_image_size=int(params.get("max_image_size") or 0),
                    num_threads=int(params.get("num_threads") or 0),
                ),
            ),
            (
                "feature_match",
                commands.matcher_cmd(
                    colmap,
                    database,
                    match_mode=params.get("match_mode", "auto"),
                    image_count=image_count,
                    use_gpu=bool(params.get("use_gpu", True)),
                    gpu_index=int(params.get("gpu_index", 0)),
                    overlap=int(params.get("overlap", 10)),
                    quadratic_overlap=bool(params.get("quadratic_overlap", True)),
                    loop_detection=bool(params.get("loop_detection", True)),
                    vocab_tree_path=self.paths.vocab_tree,
                    max_num_matches=int(params.get("max_num_matches") or 0),
                    num_threads=int(params.get("num_threads") or 0),
                ),
            ),
            ("view_graph_calibrator", commands.view_graph_calibrator_cmd(colmap, database_global)),
        ]
        if params.get("mapper_backend") == "glomap":
            glomap = self.paths.glomap_exe or "glomap"
            pairs.append(
                (
                    "glomap_mapper",
                    commands.glomap_mapper_cmd(glomap, database_global, images_dir, sparse_dir),
                )
            )
        else:
            pairs.append(
                (
                    "global_mapper",
                    commands.global_mapper_cmd(
                        colmap,
                        database_global,
                        images_dir,
                        sparse_dir,
                        num_threads=int(params.get("num_threads") or 0),
                    ),
                )
            )
        if params.get("run_training", True):
            pairs.append(
                (
                    "train_gsplat",
                    commands.train_cmd(
                        self.paths.python_exe,
                        self.paths.trainer_dir / "train.py",
                        data_dir=scene_dir,
                        result_dir=train_dir,
                        data_factor=int(params.get("data_factor", 4)),
                        max_steps=int(params.get("max_steps", 15000)),
                        strategy=str(params.get("strategy", "default")),
                        save_steps=list(params.get("save_steps") or [15000]),
                        save_ply=bool(params.get("save_ply", True)),
                        sh_degree=int(params.get("sh_degree", 3)),
                        max_splats=int(params.get("max_splats") or 0),
                        grow_grad2d=float(params.get("grow_grad2d") or 0.0002),
                        coarse_to_fine=bool(params.get("coarse_to_fine", True)),
                        opacity_reset_every=int(params.get("opacity_reset_every") or 0),
                    ),
                )
            )
        return pairs

    def run_job(self, job_id: str, cancel_event: threading.Event, rerun_stage: str | None = None) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        project = self.projects.get(job["project_id"])
        if not project:
            raise KeyError(job["project_id"])
        params = dict(job["params"])
        self.jobs.update(job_id, status="running", started_at=utcnow(), error=None, current_stage=None)
        log = lambda line: self._log(job_id, line)
        work = Path(project["work_dir"])
        try:
            result = self._execute(project, params, job_id, log, cancel_event, rerun_stage=rerun_stage)
            self._purge_if_needed(work, params, log)
            self.jobs.update(
                job_id,
                status="succeeded",
                finished_at=utcnow(),
                current_stage=None,
                result=result,
            )
            log("流水线完成")
        except CancelledError as exc:
            self.jobs.update(job_id, status="cancelled", finished_at=utcnow(), error=str(exc))
            log(str(exc))
            log("工作缓存已保留，可点续跑从中断阶段接着做")
        except Exception as exc:
            stage = getattr(exc, "stage", None)
            self.jobs.update(
                job_id,
                status="failed",
                finished_at=utcnow(),
                error=str(exc),
                current_stage=stage,
            )
            log(f"失败: {exc}")
            log("工作缓存已保留，可点续跑。队列继续下一个文件夹")

    def _purge_if_needed(self, work: Path, params: dict[str, Any], log: Callable[[str], None]) -> None:
        if not params.get("clear_cache", True):
            return
        try:
            info = purge_work_cache(work)
            kept = len(info.get("kept") or [])
            log(f"已清理工作缓存，保留 {kept} 个结果文件，避免占满磁盘")
        except Exception as exc:
            log(f"清理缓存未完成：{exc}")

    def _log(self, job_id: str, line: str) -> None:
        self.jobs.append_log(job_id, line)
        try:
            self.log_sink(job_id, line)
        except Exception:
            pass

    def _execute(
        self,
        project: dict[str, Any],
        params: dict[str, Any],
        job_id: str,
        log: Callable[[str], None],
        cancel_event: threading.Event,
        rerun_stage: str | None = None,
    ) -> dict[str, Any]:
        work = Path(project["work_dir"])
        images_dir = work / "images"
        colmap_dir = work / "colmap"
        sparse_dir = colmap_dir / "sparse"
        scene_dir = work / "gsplat_scene"
        train_dir = work / "train"
        database = colmap_dir / "database.db"
        database_global = colmap_dir / "database_global.db"

        skip_until = RESUME_GATE.get(rerun_stage, rerun_stage) if rerun_stage else None
        if skip_until:
            has_images = images_dir.is_dir() and any(images_dir.iterdir())
            if not has_images:
                log("工作图目录不在，从整理图片重新开始")
                skip_until = None

        def should_run(stage: str) -> bool:
            nonlocal skip_until
            if skip_until is None:
                return True
            if stage == skip_until:
                skip_until = None
                return True
            return False

        if should_run("prepare_images"):
            self._mark(job_id, "prepare_images", "running")
            prepared = self.projects.prepare_workset(project["id"], params.get("sort_mode"))
            log(
                f"已整理 {prepared['image_count']} 张图（硬链接 {prepared['methods']['hardlink']} / 复制 {prepared['methods']['copy']}）"
            )
            self._mark(job_id, "prepare_images", "succeeded", finished=True)
        else:
            prepared = {
                "image_count": len(list(images_dir.glob('*'))) if images_dir.exists() else 0,
                "ascii_names": [p.name for p in images_dir.glob('*')] if images_dir.exists() else [],
            }

        image_count = int(prepared["image_count"])
        colmap = self.paths.colmap_exe
        if colmap is None:
            raise PipelineError("未找到 COLMAP 可执行文件", stage="feature_extract")

        feature_cmd = commands.feature_extractor_cmd(
            colmap,
            database,
            images_dir,
            camera_model=params.get("camera_model", "SIMPLE_RADIAL"),
            max_num_features=int(params.get("max_num_features", 8192)),
            use_gpu=bool(params.get("use_gpu", True)),
            gpu_index=int(params.get("gpu_index", 0)),
            single_camera=bool(params.get("single_camera", True)),
            max_image_size=int(params.get("max_image_size") or 0),
            num_threads=int(params.get("num_threads") or 0),
        )
        match_cmd = commands.matcher_cmd(
            colmap,
            database,
            match_mode=params.get("match_mode", "auto"),
            image_count=image_count,
            use_gpu=bool(params.get("use_gpu", True)),
            gpu_index=int(params.get("gpu_index", 0)),
            overlap=int(params.get("overlap", 10)),
            quadratic_overlap=bool(params.get("quadratic_overlap", True)),
            loop_detection=bool(params.get("loop_detection", True)),
            vocab_tree_path=self.paths.vocab_tree,
            max_num_matches=int(params.get("max_num_matches") or 0),
            num_threads=int(params.get("num_threads") or 0),
        )
        calib_cmd = commands.view_graph_calibrator_cmd(colmap, database_global)

        if should_run("feature_extract"):
            colmap_dir.mkdir(parents=True, exist_ok=True)
            if database.exists():
                database.unlink()
            self._run_stage(job_id, "feature_extract", feature_cmd, log, cancel_event)

        if should_run("feature_match"):
            self._run_stage(job_id, "feature_match", match_cmd, log, cancel_event)

        if should_run("view_graph_calibrator"):
            shutil.copy2(database, database_global)
            log("已保留原始 database.db，并复制为 database_global.db")
            self._run_stage(job_id, "view_graph_calibrator", calib_cmd, log, cancel_event)

        mapper_backend = params.get("mapper_backend", "colmap_global")
        selected = None
        if mapper_backend == "glomap":
            if not self.paths.glomap_exe:
                raise PipelineError("未找到独立 GLOMAP 1.2 可执行文件", stage="glomap_mapper")
            glomap_cmd = commands.glomap_mapper_cmd(
                self.paths.glomap_exe, database_global, images_dir, sparse_dir
            )
            if should_run("glomap_mapper"):
                _reset_output_dir(sparse_dir)
                log("独立 GLOMAP 同样没有百分比进度，主要在 CPU 上算。")
                self._run_stage(job_id, "glomap_mapper", glomap_cmd, log, cancel_event)
                log("GLOMAP 主要求解在 CPU 上运行，不是全 GPU 重建")
        else:
            global_cmd = commands.global_mapper_cmd(
                colmap,
                database_global,
                images_dir,
                sparse_dir,
                num_threads=int(params.get("num_threads") or 0),
            )
            if should_run("global_mapper"):
                _reset_output_dir(sparse_dir)
                log("全局重建是 CPU 非线性优化，COLMAP 不给百分比。日志还在刷就说明还在算，属正常。")
                self._run_stage(job_id, "global_mapper", global_cmd, log, cancel_event)
                log("COLMAP global_mapper 内置 GLOMAP：位姿与全局优化主要使用 CPU")
        if not sparse_dir.exists():
            sparse_dir.mkdir(parents=True, exist_ok=True)

        if should_run("validate_model"):
            self._mark(job_id, "validate_model", "running")
            selected = select_largest_model(
                sparse_dir,
                total_images=image_count,
                min_registration_ratio=float(params.get("min_registration_ratio", 0.5)),
            )
            log(f"模型选择: {json.dumps(selected['selected'], ensure_ascii=False) if selected.get('selected') else '无'}")
            for warning in selected.get("warnings") or []:
                log(f"警告: {warning}")
            used_fallback = False
            if selected.get("block_training") and params.get("allow_incremental_fallback", True):
                log("全局重建不足，回退增量 mapper（主要走 CPU，可能明显更慢）")
                fallback_dir = colmap_dir / "sparse_incremental"
                _reset_output_dir(fallback_dir)
                inc_cmd = commands.incremental_mapper_cmd(
                    colmap, database, images_dir, fallback_dir
                )
                self._run_stage(job_id, "incremental_mapper", inc_cmd, log, cancel_event)
                selected = select_largest_model(
                    fallback_dir,
                    total_images=image_count,
                    min_registration_ratio=float(params.get("min_registration_ratio", 0.5)),
                )
                used_fallback = True
            if selected.get("block_training"):
                self._mark(job_id, "validate_model", "failed", finished=True, error="注册率过低")
                raise PipelineError("模型不完整，已阻止无意义训练", stage="validate_model")
            self._mark(job_id, "validate_model", "succeeded", finished=True)
        else:
            selected = select_largest_model(sparse_dir, total_images=image_count)
            used_fallback = False

        if should_run("export_scene"):
            self._mark(job_id, "export_scene", "running")
            if scene_dir.exists():
                shutil.rmtree(scene_dir)
            src_model = Path(selected["selected"]["path"])
            if params.get("undistort_images", True):
                undistort_cmd = commands.image_undistorter_cmd(
                    colmap,
                    images_dir,
                    src_model,
                    scene_dir,
                )
                self._run_stage(job_id, "undistort_scene", undistort_cmd, log, cancel_event)
            else:
                (scene_dir / "images").mkdir(parents=True, exist_ok=True)
                for src in images_dir.iterdir():
                    if src.is_file():
                        shutil.copy2(src, scene_dir / "images" / src.name)
                copy_model(src_model, scene_dir / "sparse" / "0")
                log("已跳过去畸变，直接复制图像与稀疏模型")
            sparse_root = scene_dir / "sparse"
            sparse_zero = sparse_root / "0"
            if not sparse_zero.is_dir():
                copy_model(sparse_root, sparse_zero)
            factor = int(params.get("data_factor", 4))
            downscale_images(scene_dir / "images", scene_dir / "images_2", 2)
            downscale_images(scene_dir / "images", scene_dir / "images_4", 4)
            if factor not in {1, 2, 4}:
                downscale_images(scene_dir / "images", scene_dir / f"images_{factor}", factor)
            log(f"已去畸变并导出 gsplat 场景，factor={factor}")
            self._mark(job_id, "export_scene", "succeeded", finished=True)

        train_stats: dict[str, Any] = {}
        if params.get("run_training", True) and should_run("train_gsplat"):
            trainer = self.paths.trainer_dir / "train.py"
            train_cmd = commands.train_cmd(
                self.paths.python_exe,
                trainer,
                data_dir=scene_dir,
                result_dir=train_dir,
                data_factor=int(params.get("data_factor", 4)),
                max_steps=int(params.get("max_steps", 15000)),
                strategy=str(params.get("strategy", "default")),
                save_steps=list(params.get("save_steps") or [15000]),
                save_ply=bool(params.get("save_ply", True)),
                sh_degree=int(params.get("sh_degree", 3)),
                max_splats=int(params.get("max_splats") or 0),
                grow_grad2d=float(params.get("grow_grad2d") or 0.0002),
                coarse_to_fine=bool(params.get("coarse_to_fine", True)),
                opacity_reset_every=int(params.get("opacity_reset_every") or 0),
            )
            try:
                self._run_stage(job_id, "train_gsplat", train_cmd, log, cancel_event)
            except PipelineError as exc:
                if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
                    hint = oom_hint()
                    log(hint)
                    raise PipelineError(hint, stage="train_gsplat") from exc
                raise
            stats_file = train_dir / "stats.json"
            if stats_file.is_file():
                train_stats = json.loads(stats_file.read_text(encoding="utf-8"))
            ply_candidates = sorted((train_dir / "ply").glob("*.ply")) if (train_dir / "ply").is_dir() else []
            train_stats["ply"] = str(ply_candidates[-1]) if ply_candidates else None

        return {
            "prepared": prepared,
            "reconstruction": selected,
            "used_incremental_fallback": used_fallback,
            "scene_dir": str(scene_dir),
            "train": train_stats,
        }

    def _run_stage(
        self,
        job_id: str,
        stage: str,
        argv: list[str],
        log: Callable[[str], None],
        cancel_event: threading.Event,
    ) -> None:
        existing = self.jobs.db.query(
            "SELECT id FROM stages WHERE job_id=? AND name=?", (job_id, stage)
        )
        if not existing:
            self.jobs.add_stage(job_id, stage, STAGE_DEVICES[stage], argv, status="running")
        self.jobs.set_stage(job_id, stage, started_at=utcnow(), status="running", error=None)
        self.jobs.update(job_id, current_stage=stage)
        log(f"==> {stage} [{STAGE_DEVICES[stage]}] {commands.subprocess_display(argv)}")
        try:
            result = self.runner.run(argv, log=log, cancel_event=cancel_event)
        except CancelledError:
            self.jobs.set_stage(job_id, stage, status="cancelled", finished_at=utcnow())
            raise
        except Exception as exc:
            self.jobs.set_stage(job_id, stage, status="failed", finished_at=utcnow(), error=str(exc))
            raise PipelineError(str(exc), stage=stage) from exc
        if result.returncode != 0:
            error = f"{stage} 退出码 {result.returncode}"
            self.jobs.set_stage(job_id, stage, status="failed", finished_at=utcnow(), error=error)
            raise PipelineError(error, stage=stage)
        self.jobs.set_stage(job_id, stage, status="succeeded", finished_at=utcnow())

    def _mark(self, job_id: str, stage: str, status: str, finished: bool = False, error: str | None = None) -> None:
        existing = [
            row
            for row in self.jobs.db.query(
                "SELECT id FROM stages WHERE job_id=? AND name=?", (job_id, stage)
            )
        ]
        if not existing:
            self.jobs.add_stage(job_id, stage, STAGE_DEVICES[stage], [], status=status)
        fields: dict[str, Any] = {"status": status}
        if status == "running":
            fields["started_at"] = utcnow()
            self.jobs.update(job_id, current_stage=stage)
        if finished:
            fields["finished_at"] = utcnow()
        if error:
            fields["error"] = error
        self.jobs.set_stage(job_id, stage, **fields)
