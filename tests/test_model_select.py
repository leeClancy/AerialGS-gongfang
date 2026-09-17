from pathlib import Path

from backend.app.reconstruction import select_largest_model, write_text_model


def test_select_max_registered_model(tmp_path: Path):
    sparse = tmp_path / "sparse"
    write_text_model(sparse / "0", image_names=[f"{i:06d}.jpg" for i in range(1, 4)])
    write_text_model(sparse / "1", image_names=[f"{i:06d}.jpg" for i in range(1, 9)])
    write_text_model(sparse / "2", image_names=["000001.jpg"])
    result = select_largest_model(sparse, total_images=8, min_registration_ratio=0.5)
    assert result["ok"] is True
    assert result["selected"]["num_images"] == 8
    assert Path(result["selected"]["path"]).name == "1"
    assert result["registration_ratio"] == 1.0
    assert len(result["models"]) == 3


def test_block_training_when_ratio_low(tmp_path: Path):
    sparse = tmp_path / "sparse"
    write_text_model(sparse / "0", image_names=["a.jpg", "b.jpg"])
    result = select_largest_model(sparse, total_images=20, min_registration_ratio=0.5)
    assert result["block_training"] is True
    assert result["ok"] is False
    assert any("注册率" in w for w in result["warnings"])


def test_empty_sparse(tmp_path: Path):
    result = select_largest_model(tmp_path / "missing")
    assert result["ok"] is False
    assert result["selected"] is None
