from pathlib import Path

from openi_baselines.artifacts import collect_artifacts


def test_collect_artifacts_classifies_checkpoint_prediction_and_result(tmp_path):
    source = tmp_path / "native"
    source.mkdir()
    (source / "checkpoint.pth").write_bytes(b"weights")
    (source / "pred.npy").write_bytes(b"prediction")
    (source / "result.txt").write_text("mse:0.4, mae:0.3", encoding="utf-8")

    counts = collect_artifacts(source, tmp_path / "normalized")

    assert counts == {
        "checkpoints": 1,
        "predictions": 1,
        "native_results": 1,
    }
    assert (tmp_path / "normalized" / "checkpoints" / "checkpoint.pth").is_file()
    assert (tmp_path / "normalized" / "predictions" / "pred.npy").is_file()
    assert (tmp_path / "normalized" / "native_results" / "result.txt").is_file()


def test_collect_artifacts_preserves_relative_paths_and_source_files(tmp_path):
    source = tmp_path / "native"
    nested = source / "results" / "setting"
    nested.mkdir(parents=True)
    prediction = nested / "pred.npy"
    prediction.write_bytes(b"prediction")

    collect_artifacts(source, tmp_path / "normalized")

    copied = tmp_path / "normalized" / "predictions" / "results" / "setting" / "pred.npy"
    assert copied.is_file()
    assert prediction.is_file()


def test_collect_artifacts_does_not_recurse_into_destination(tmp_path):
    source = tmp_path / "task"
    source.mkdir()
    destination = source / "artifacts"
    (source / "pred.npy").write_bytes(b"prediction")

    first = collect_artifacts(source, destination)
    second = collect_artifacts(source, destination)

    assert first["predictions"] == 1
    assert second["predictions"] == 1
    assert len(list(destination.rglob("pred.npy"))) == 1
