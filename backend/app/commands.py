from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

MatchMode = Literal["auto", "sequential", "exhaustive"]
MapperBackend = Literal["colmap_global", "glomap", "incremental"]


def resolve_match_mode(mode: str, image_count: int, auto_exhaustive_limit: int = 50) -> str:
    normalized = (mode or "auto").strip().lower()
    if normalized in {"exhaustive", "sequential"}:
        return normalized
    if image_count <= auto_exhaustive_limit:
        return "exhaustive"
    return "sequential"


def _flag(value: bool) -> str:
    return "1" if value else "0"


def feature_extractor_cmd(
    colmap: str | Path,
    database_path: str | Path,
    image_path: str | Path,
    *,
    camera_model: str = "SIMPLE_RADIAL",
    max_num_features: int = 8192,
    use_gpu: bool = True,
    gpu_index: int = 0,
    single_camera: bool = True,
    max_image_size: int = 0,
    num_threads: int = 0,
) -> list[str]:
    cmd = [
        str(colmap),
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.single_camera",
        _flag(single_camera),
        "--ImageReader.camera_model",
        str(camera_model),
        "--FeatureExtraction.use_gpu",
        _flag(use_gpu),
        "--FeatureExtraction.gpu_index",
        str(gpu_index),
        "--SiftExtraction.max_num_features",
        str(max_num_features),
    ]
    if max_image_size:
        cmd.extend(["--SiftExtraction.max_image_size", str(int(max_image_size))])
    if num_threads:
        cmd.extend(["--SiftExtraction.num_threads", str(int(num_threads))])
    return cmd


def exhaustive_matcher_cmd(
    colmap: str | Path,
    database_path: str | Path,
    *,
    use_gpu: bool = True,
    gpu_index: int = 0,
    max_num_matches: int = 0,
    num_threads: int = 0,
) -> list[str]:
    cmd = [
        str(colmap),
        "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--FeatureMatching.use_gpu",
        _flag(use_gpu),
        "--FeatureMatching.gpu_index",
        str(gpu_index),
    ]
    if max_num_matches:
        cmd.extend(["--SiftMatching.max_num_matches", str(int(max_num_matches))])
    if num_threads:
        cmd.extend(["--SiftMatching.num_threads", str(int(num_threads))])
    return cmd


def sequential_matcher_cmd(
    colmap: str | Path,
    database_path: str | Path,
    *,
    use_gpu: bool = True,
    gpu_index: int = 0,
    overlap: int = 10,
    quadratic_overlap: bool = True,
    loop_detection: bool = True,
    vocab_tree_path: str | Path | None = None,
    loop_detection_period: int = 10,
    loop_detection_num_images: int = 50,
    max_num_matches: int = 0,
    num_threads: int = 0,
) -> list[str]:
    cmd = [
        str(colmap),
        "sequential_matcher",
        "--database_path",
        str(database_path),
        "--FeatureMatching.use_gpu",
        _flag(use_gpu),
        "--FeatureMatching.gpu_index",
        str(gpu_index),
        "--SequentialMatching.overlap",
        str(int(overlap)),
        "--SequentialMatching.quadratic_overlap",
        _flag(quadratic_overlap),
        "--SequentialMatching.loop_detection",
        _flag(loop_detection),
        "--SequentialMatching.loop_detection_period",
        str(int(loop_detection_period)),
        "--SequentialMatching.loop_detection_num_images",
        str(int(loop_detection_num_images)),
    ]
    if max_num_matches:
        cmd.extend(["--SiftMatching.max_num_matches", str(int(max_num_matches))])
    if num_threads:
        cmd.extend(["--SiftMatching.num_threads", str(int(num_threads))])
    if loop_detection and vocab_tree_path:
        cmd.extend(["--SequentialMatching.vocab_tree_path", str(vocab_tree_path)])
    return cmd


def matcher_cmd(
    colmap: str | Path,
    database_path: str | Path,
    *,
    match_mode: str,
    image_count: int,
    use_gpu: bool = True,
    gpu_index: int = 0,
    overlap: int = 10,
    quadratic_overlap: bool = True,
    loop_detection: bool = True,
    vocab_tree_path: str | Path | None = None,
    auto_exhaustive_limit: int = 50,
    max_num_matches: int = 0,
    num_threads: int = 0,
) -> list[str]:
    resolved = resolve_match_mode(match_mode, image_count, auto_exhaustive_limit)
    if resolved == "exhaustive":
        return exhaustive_matcher_cmd(
            colmap,
            database_path,
            use_gpu=use_gpu,
            gpu_index=gpu_index,
            max_num_matches=max_num_matches,
            num_threads=num_threads,
        )
    return sequential_matcher_cmd(
        colmap,
        database_path,
        use_gpu=use_gpu,
        gpu_index=gpu_index,
        overlap=overlap,
        quadratic_overlap=quadratic_overlap,
        loop_detection=loop_detection,
        vocab_tree_path=vocab_tree_path,
        max_num_matches=max_num_matches,
        num_threads=num_threads,
    )


def view_graph_calibrator_cmd(colmap: str | Path, database_path: str | Path) -> list[str]:
    return [
        str(colmap),
        "view_graph_calibrator",
        "--database_path",
        str(database_path),
    ]


def global_mapper_cmd(
    colmap: str | Path,
    database_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    *,
    num_threads: int = 0,
    multiple_models: bool = True,
) -> list[str]:
    cmd = [
        str(colmap),
        "global_mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
        "--GlobalMapper.multiple_models",
        _flag(multiple_models),
    ]
    if num_threads:
        cmd.extend(["--GlobalMapper.num_threads", str(int(num_threads))])
    return cmd


def incremental_mapper_cmd(
    colmap: str | Path,
    database_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
) -> list[str]:
    return [
        str(colmap),
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
    ]


def image_undistorter_cmd(
    colmap: str | Path,
    image_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
) -> list[str]:
    return [
        str(colmap),
        "image_undistorter",
        "--image_path",
        str(image_path),
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--output_type",
        "COLMAP",
    ]


def glomap_mapper_cmd(
    glomap: str | Path,
    database_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
) -> list[str]:
    return [
        str(glomap),
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
    ]


def train_cmd(
    python_exe: str | Path,
    trainer_script: str | Path,
    *,
    data_dir: str | Path,
    result_dir: str | Path,
    data_factor: int = 4,
    max_steps: int = 15000,
    strategy: str = "default",
    save_steps: list[int] | None = None,
    save_ply: bool = True,
    sh_degree: int = 3,
    max_splats: int = 0,
    grow_grad2d: float = 0.0002,
    coarse_to_fine: bool = True,
    opacity_reset_every: int = 3000,
) -> list[str]:
    save_steps = save_steps or [max_steps]
    cmd = [
        str(python_exe),
        str(trainer_script),
        "--data_dir",
        str(data_dir),
        "--result_dir",
        str(result_dir),
        "--data_factor",
        str(int(data_factor)),
        "--max_steps",
        str(int(max_steps)),
        "--strategy",
        str(strategy),
        "--save_steps",
        ",".join(str(int(s)) for s in save_steps),
        "--sh_degree",
        str(int(sh_degree)),
        "--max_splats",
        str(int(max_splats)),
        "--grow_grad2d",
        str(float(grow_grad2d)),
        "--opacity_reset_every",
        str(int(opacity_reset_every)),
    ]
    if coarse_to_fine:
        cmd.append("--coarse_to_fine")
    if save_ply:
        cmd.append("--save_ply")
    return cmd


def command_preview(argv: list[str]) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "display": subprocess_display(argv),
    }


def subprocess_display(argv: list[str]) -> str:
    parts = []
    for item in argv:
        if any(ch.isspace() for ch in item) or any(ord(ch) > 127 for ch in item):
            parts.append('"' + item.replace('"', '\\"') + '"')
        else:
            parts.append(item)
    return " ".join(parts)
