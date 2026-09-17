from pathlib import Path

from backend.app.commands import (
    exhaustive_matcher_cmd,
    feature_extractor_cmd,
    glomap_mapper_cmd,
    global_mapper_cmd,
    image_undistorter_cmd,
    incremental_mapper_cmd,
    matcher_cmd,
    resolve_match_mode,
    sequential_matcher_cmd,
    train_cmd,
    view_graph_calibrator_cmd,
)


def test_feature_extractor_single_camera_gpu():
    cmd = feature_extractor_cmd(
        r"C:\runtime\colmap.exe",
        r"C:\work\database.db",
        r"C:\work\images",
        camera_model="SIMPLE_RADIAL",
        max_num_features=4096,
        use_gpu=True,
        gpu_index=0,
    )
    assert cmd[0].endswith("colmap.exe")
    assert cmd[1] == "feature_extractor"
    assert "--ImageReader.single_camera" in cmd
    assert cmd[cmd.index("--ImageReader.single_camera") + 1] == "1"
    assert cmd[cmd.index("--FeatureExtraction.use_gpu") + 1] == "1"
    assert cmd[cmd.index("--FeatureExtraction.gpu_index") + 1] == "0"
    assert cmd[cmd.index("--SiftExtraction.max_num_features") + 1] == "4096"
    sized = feature_extractor_cmd("c", "d", "i", max_image_size=1600, num_threads=8)
    assert sized[sized.index("--SiftExtraction.max_image_size") + 1] == "1600"
    assert sized[sized.index("--SiftExtraction.num_threads") + 1] == "8"
    assert "--SiftExtraction.use_gpu" not in cmd
    assert "--SiftMatching.use_gpu" not in cmd
    assert all(not item.startswith("cmd.exe") for item in cmd)


def test_auto_match_small_uses_exhaustive():
    assert resolve_match_mode("auto", 12) == "exhaustive"
    cmd = matcher_cmd("colmap", "db.db", match_mode="auto", image_count=12)
    assert cmd[1] == "exhaustive_matcher"
    assert "--FeatureMatching.use_gpu" in cmd
    assert "--SiftMatching.use_gpu" not in cmd


def test_auto_match_large_uses_sequential():
    assert resolve_match_mode("auto", 80) == "sequential"
    cmd = sequential_matcher_cmd(
        "colmap",
        "db.db",
        overlap=15,
        quadratic_overlap=True,
        loop_detection=True,
        vocab_tree_path="vocab.bin",
    )
    assert cmd[1] == "sequential_matcher"
    assert cmd[cmd.index("--SequentialMatching.overlap") + 1] == "15"
    assert cmd[cmd.index("--SequentialMatching.quadratic_overlap") + 1] == "1"
    assert cmd[cmd.index("--SequentialMatching.loop_detection") + 1] == "1"
    assert cmd[cmd.index("--SequentialMatching.vocab_tree_path") + 1] == "vocab.bin"
    assert cmd[cmd.index("--FeatureMatching.use_gpu") + 1] == "1"
    assert "--SiftMatching.use_gpu" not in cmd


def test_exhaustive_and_forced_modes():
    assert matcher_cmd("c", "d", match_mode="exhaustive", image_count=999)[1] == "exhaustive_matcher"
    assert matcher_cmd("c", "d", match_mode="sequential", image_count=3)[1] == "sequential_matcher"


def test_view_graph_and_mappers():
    calib = view_graph_calibrator_cmd("colmap", "database_global.db")
    assert calib[1] == "view_graph_calibrator"
    glob = global_mapper_cmd("colmap", "db", "images", "sparse")
    assert glob[1] == "global_mapper"
    assert "--GlobalMapper.multiple_models" in glob
    inc = incremental_mapper_cmd("colmap", "db", "images", "sparse")
    assert inc[1] == "mapper"
    glo = glomap_mapper_cmd("glomap.exe", "db", "images", "sparse")
    assert Path(glo[0]).name.startswith("glomap")
    assert glo[1] == "mapper"
    undistort = image_undistorter_cmd("colmap", "images", "sparse/0", "scene")
    assert undistort[1] == "image_undistorter"
    assert undistort[undistort.index("--output_type") + 1] == "COLMAP"


def test_train_cmd_and_no_shell_string():
    cmd = train_cmd(
        "python.exe",
        r"F:\app\trainer\train.py",
        data_dir=r"C:\work\scene",
        result_dir=r"C:\work\train",
        data_factor=4,
        max_steps=15000,
        strategy="mcmc",
        save_steps=[7000, 15000],
        sh_degree=3,
        max_splats=4500000,
        grow_grad2d=0.0002,
        coarse_to_fine=True,
    )
    assert cmd[cmd.index("--sh_degree") + 1] == "3"
    assert cmd[cmd.index("--max_splats") + 1] == "4500000"
    assert "--coarse_to_fine" in cmd
    assert "--data_factor" in cmd
    assert cmd[cmd.index("--data_factor") + 1] == "4"
    assert cmd[cmd.index("--max_steps") + 1] == "15000"
    assert cmd[cmd.index("--strategy") + 1] == "mcmc"
    assert "--save_ply" in cmd
    assert not any(" && " in x for x in cmd)
