from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from backend.app.config import PRESETS, RuntimePaths
from backend.app.db import Database
from backend.app.images import inspect_images, materialize_work_images, sort_records
from backend.app.cache import remove_work_tree


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStore:
    def __init__(self, db: Database, paths: RuntimePaths) -> None:
        self.db = db
        self.paths = paths

    def create(
        self,
        *,
        name: str,
        source_dir: str,
        sort_mode: str = "filename",
        work_dir: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        records = inspect_images(source_dir, sort_mode)
        if not records:
            raise ValueError("源目录中没有支持的图片")
        project_id = uuid.uuid4().hex[:12]
        root = Path(work_dir) if work_dir else (self.paths.work_dir / project_id)
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": project_id,
            "name": name or Path(source_dir).name,
            "source_dir": str(Path(source_dir).resolve()),
            "work_dir": str(root.resolve()),
            "created_at": utcnow(),
            "settings": settings or {"sort_mode": sort_mode},
            "image_count": len(records),
            "preview": [rec.as_dict() for rec in records[:8]],
            "preview_tail": [rec.as_dict() for rec in records[-4:]],
        }
        self.db.execute(
            "INSERT INTO projects(id, name, source_dir, work_dir, created_at, settings_json) VALUES (?,?,?,?,?,?)",
            (
                payload["id"],
                payload["name"],
                payload["source_dir"],
                payload["work_dir"],
                payload["created_at"],
                json.dumps(payload["settings"], ensure_ascii=False),
            ),
        )
        mapping_path = root / "source_preview.json"
        mapping_path.write_text(
            json.dumps([rec.as_dict() for rec in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload

    def _with_image_count(self, item: dict[str, Any]) -> dict[str, Any]:
        work = Path(item.get("work_dir") or "")
        preview_path = work / "source_preview.json"
        if preview_path.is_file():
            try:
                records = json.loads(preview_path.read_text(encoding="utf-8"))
                if isinstance(records, list) and records:
                    item["image_count"] = len(records)
                    return item
            except (OSError, json.JSONDecodeError):
                pass
        images_dir = work / "images"
        if images_dir.is_dir():
            item["image_count"] = sum(1 for path in images_dir.iterdir() if path.is_file())
        else:
            item["image_count"] = int(item.get("image_count") or 0)
        return item

    def list_projects(self) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM projects ORDER BY created_at DESC")
        result = []
        for row in rows:
            item = dict(row)
            item["settings"] = json.loads(item.pop("settings_json") or "{}")
            result.append(self._with_image_count(item))
        return result

    def get(self, project_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM projects WHERE id=?", (project_id,))
        if not row:
            return None
        item = dict(row)
        item["settings"] = json.loads(item.pop("settings_json") or "{}")
        return self._with_image_count(item)

    def preview_order(self, project_id: str, sort_mode: str) -> dict[str, Any]:
        project = self.get(project_id)
        if not project:
            raise KeyError(project_id)
        records = inspect_images(project["source_dir"], sort_mode)
        return {
            "project_id": project_id,
            "sort_mode": sort_mode,
            "count": len(records),
            "images": [rec.as_dict() for rec in records],
        }

    def prepare_workset(self, project_id: str, sort_mode: str | None = None) -> dict[str, Any]:
        project = self.get(project_id)
        if not project:
            raise KeyError(project_id)
        mode = sort_mode or project.get("settings", {}).get("sort_mode") or "filename"
        records = inspect_images(project["source_dir"], mode)
        images_dir = Path(project["work_dir"]) / "images"
        mapping = materialize_work_images(records, images_dir)
        map_path = Path(project["work_dir"]) / "source_map.json"
        map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "image_count": len(mapping),
            "images_dir": str(images_dir),
            "methods": {
                "hardlink": sum(1 for item in mapping if item["method"] == "hardlink"),
                "copy": sum(1 for item in mapping if item["method"] == "copy"),
            },
            "ascii_names": [item["ascii_name"] for item in mapping],
        }

    def delete(self, project_id: str, *, wipe_work: bool = True) -> dict[str, Any] | None:
        project = self.get(project_id)
        if not project:
            return None
        self.db.execute("DELETE FROM projects WHERE id=?", (project_id,))
        if wipe_work:
            project["cache"] = remove_work_tree(project["work_dir"], source_dir=project.get("source_dir"))
        return project


class JobStore:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._lock = threading.Lock()

    def create(self, project_id: str, params: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        now = utcnow()
        self.db.execute(
            "INSERT INTO jobs(id, project_id, status, created_at, params_json, result_json) VALUES (?,?,?,?,?,?)",
            (job_id, project_id, "queued", now, json.dumps(params, ensure_ascii=False), "{}"),
        )
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not row:
            return None
        item = dict(row)
        item["params"] = json.loads(item.pop("params_json") or "{}")
        item["result"] = json.loads(item.pop("result_json") or "{}")
        item["stages"] = self.db.query(
            "SELECT name, device, status, started_at, finished_at, error, command_json FROM stages WHERE job_id=? ORDER BY id",
            (job_id,),
        )
        for stage in item["stages"]:
            stage["command"] = json.loads(stage.pop("command_json") or "[]")
        return item

    def list_jobs(self, project_id: str | None = None) -> list[dict[str, Any]]:
        if project_id:
            rows = self.db.query(
                "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            )
        else:
            rows = self.db.query("SELECT * FROM jobs ORDER BY created_at DESC")
        result = []
        for row in rows:
            item = dict(row)
            item["params"] = json.loads(item.pop("params_json") or "{}")
            item["result"] = json.loads(item.pop("result_json") or "{}")
            result.append(item)
        return result

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = []
        values: list[Any] = []
        for key, value in fields.items():
            if key in {"params", "result"}:
                assignments.append(f"{key}_json=?")
                values.append(json.dumps(value, ensure_ascii=False))
            else:
                assignments.append(f"{key}=?")
                values.append(value)
        values.append(job_id)
        self.db.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id=?", tuple(values))

    def add_stage(
        self,
        job_id: str,
        name: str,
        device: str,
        command: list[str] | None = None,
        status: str = "pending",
    ) -> None:
        self.db.execute(
            "INSERT INTO stages(job_id, name, device, status, command_json) VALUES (?,?,?,?,?)",
            (job_id, name, device, status, json.dumps(command or [], ensure_ascii=False)),
        )

    def set_stage(self, job_id: str, name: str, **fields: Any) -> None:
        assignments = []
        values: list[Any] = []
        for key, value in fields.items():
            assignments.append(f"{key}=?")
            values.append(value)
        values.extend([job_id, name])
        self.db.execute(
            f"UPDATE stages SET {', '.join(assignments)} WHERE job_id=? AND name=?",
            tuple(values),
        )

    def append_log(self, job_id: str, line: str) -> None:
        self.db.execute(
            "INSERT INTO logs(job_id, ts, line) VALUES (?,?,?)",
            (job_id, utcnow(), line),
        )

    def logs(self, job_id: str, after_id: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT id, ts, line FROM logs WHERE job_id=? AND id>? ORDER BY id LIMIT ?",
            (job_id, after_id, limit),
        )

    def recover_interrupted(self) -> int:
        rows = self.db.query("SELECT id FROM jobs WHERE status IN ('running','queued')")
        count = 0
        now = utcnow()
        for row in rows:
            self.update(
                row["id"],
                status="failed",
                finished_at=now,
                error="进程中断，可从失败阶段重跑",
            )
            count += 1
        return count

    def delete(self, job_id: str) -> None:
        self.db.execute("DELETE FROM logs WHERE job_id=?", (job_id,))
        self.db.execute("DELETE FROM stages WHERE job_id=?", (job_id,))
        self.db.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def ids_for_project(self, project_id: str) -> list[str]:
        rows = self.db.query("SELECT id FROM jobs WHERE project_id=?", (project_id,))
        return [str(row["id"]) for row in rows]


RESUME_GATE = {
    "undistort_scene": "export_scene",
    "incremental_mapper": "validate_model",
}


def resume_stage_name(job: dict[str, Any]) -> str | None:
    """First incomplete pipeline gate. None means run from the beginning."""
    stages = job.get("stages") or []
    name = None
    for stage in stages:
        if stage.get("status") in {"failed", "cancelled", "running", "pending"}:
            name = str(stage.get("name") or "") or None
            break
    if name is None and job.get("status") in {"failed", "cancelled"}:
        current = job.get("current_stage")
        if current:
            name = str(current)
    if not name:
        return None
    return RESUME_GATE.get(name, name)


def default_job_params(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    preset_name = (overrides or {}).get("preset", "balanced")
    preset = dict(PRESETS.get(preset_name, PRESETS["balanced"]))
    params = {
        "preset": preset_name,
        "sort_mode": "filename",
        "camera_model": "SIMPLE_RADIAL",
        "max_num_features": int(preset.get("max_num_features", 8192)),
        "match_mode": "auto",
        "overlap": 10,
        "quadratic_overlap": True,
        "loop_detection": True,
        "single_camera": True,
        "mapper_backend": "colmap_global",
        "allow_incremental_fallback": True,
        "min_registration_ratio": 0.5,
        "use_gpu": True,
        "gpu_index": 0,
        "data_factor": preset["data_factor"],
        "max_steps": preset["max_steps"],
        "strategy": preset.get("strategy", "default"),
        "save_steps": preset.get("save_steps", [preset["max_steps"]]),
        "save_ply": True,
        "run_training": True,
        "max_splats": int(preset.get("max_splats", 1500000)),
        "grow_grad2d": float(preset.get("grow_grad2d", 0.0002)),
        "sh_degree": int(preset.get("sh_degree", 3)),
        "coarse_to_fine": bool(preset.get("coarse_to_fine", True)),
        "opacity_reset_every": int(preset.get("opacity_reset_every", 3000)),
        "num_threads": 0,
        "max_image_size": int(preset.get("max_image_size", 3200)),
        "max_num_matches": int(preset.get("max_num_matches", 32768)),
        "undistort_images": True,
        "clear_cache": True,
    }
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                params[key] = value
    return params
