import json

import pytest

from openi_baselines.metrics import (
    MetricsNotFoundError,
    find_metrics,
    parse_text_metrics,
    write_metrics,
)


def test_parse_text_metrics_uses_last_complete_evaluation():
    text = "epoch mse:0.8, mae:0.7\nfinal mse:0.42, mae:0.31, rse:0.5"

    metrics = parse_text_metrics(text)

    assert metrics == {"mse": 0.42, "mae": 0.31, "rse": 0.5}


def test_parse_text_metrics_accepts_scientific_notation_and_correlation():
    text = "mse:1.2e-03, mae:4E-2, rse:0.8, corr:-2.5e-1"

    metrics = parse_text_metrics(text)

    assert metrics["mse"] == pytest.approx(0.0012)
    assert metrics["mae"] == pytest.approx(0.04)
    assert metrics["corr"] == pytest.approx(-0.25)


def test_find_metrics_prefers_result_file_over_training_log(tmp_path):
    (tmp_path / "train.log").write_text("mse:0.9, mae:0.8", encoding="utf-8")
    result = tmp_path / "nested" / "result.txt"
    result.parent.mkdir()
    result.write_text("mse:0.3, mae:0.2", encoding="utf-8")

    metrics, source = find_metrics(tmp_path)

    assert metrics["mse"] == 0.3
    assert source == result


def test_missing_complete_metrics_raise_clear_error():
    with pytest.raises(MetricsNotFoundError, match="MSE and MAE"):
        parse_text_metrics("training loss: 0.1")


def test_write_metrics_creates_normalized_json(tmp_path):
    target = tmp_path / "metrics.json"

    write_metrics(
        target,
        model="DLinear",
        dataset="ETTh1",
        pred_len=96,
        metrics={"mse": 0.4, "mae": 0.3},
        source=tmp_path / "result.txt",
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["model"] == "DLinear"
    assert payload["pred_len"] == 96
    assert payload["metrics"]["mae"] == 0.3
