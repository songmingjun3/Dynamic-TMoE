from pathlib import Path

import pytest

from openi_baselines.paths import resolve_dataset_file, resolve_repo


def test_resolve_repo_accepts_nested_openi_checkout(tmp_path):
    repo = tmp_path / "code" / "Dynamic-TMoE"
    (repo / "baselines" / "Dlinear").mkdir(parents=True)

    assert resolve_repo(tmp_path / "code") == repo


def test_resolve_repo_accepts_repository_root(tmp_path):
    (tmp_path / "baselines").mkdir()

    assert resolve_repo(tmp_path) == tmp_path


@pytest.mark.parametrize(
    ("dataset", "relative_file"),
    [
        ("ETTh1", "ETT-small/ETTh1.csv"),
        ("ETTm2", "ETT-small/ETTm2.csv"),
        ("Electricity", "electricity/electricity.csv"),
        ("Exchange", "exchange_rate/exchange_rate.csv"),
        ("ILI", "illness/national_illness.csv"),
        ("Traffic", "traffic/traffic.csv"),
        ("Weather", "weather/weather.csv"),
    ],
)
def test_resolve_dataset_finds_selected_ts_mount(tmp_path, dataset, relative_file):
    csv_file = tmp_path / "dataset" / "TS" / Path(relative_file)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    csv_file.touch()

    location = resolve_dataset_file(tmp_path / "dataset", dataset)

    assert location.file == csv_file
    assert location.root_path == csv_file.parent


def test_missing_dataset_reports_expected_filename(tmp_path):
    with pytest.raises(FileNotFoundError, match="ETTh1.csv"):
        resolve_dataset_file(tmp_path, "ETTh1")
