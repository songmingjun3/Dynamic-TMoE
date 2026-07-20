import pytest

from openi_baselines.registry import (
    UnsupportedTaskError,
    get_task,
    iter_tasks,
    parse_batch_sizes,
    parse_pred_lengths,
)


def test_registry_contains_expected_supported_matrix():
    tasks = {(task.model, task.dataset) for task in iter_tasks()}

    assert len(tasks) == 79
    assert ("TimeMixer", "Exchange") not in tasks
    assert ("TimeMixer", "ILI") not in tasks
    assert ("DLinear", "Exchange") in tasks
    assert ("ST-MTM", "Weather") in tasks


def test_names_are_case_insensitive_and_alias_aware():
    assert get_task("dlinear", "electricity").model == "DLinear"
    assert get_task("st-mtm", "ecl").dataset == "Electricity"
    assert get_task("timesnet", "illness").dataset == "ILI"
    assert get_task("fits", "exchange_rate").dataset == "Exchange"


def test_prediction_length_selection_supports_all_single_and_subset():
    task = get_task("PatchTST", "ETTh1")

    assert parse_pred_lengths(task, None) == (96, 192, 336, 720)
    assert parse_pred_lengths(task, "all") == (96, 192, 336, 720)
    assert parse_pred_lengths(task, "336") == (336,)
    assert parse_pred_lengths(task, "96,720") == (96, 720)


def test_ili_uses_short_horizons():
    task = get_task("RAFT", "ILI")

    assert task.horizons == (24, 36, 48, 60)


def test_duplicate_prediction_lengths_are_removed_in_request_order():
    task = get_task("DLinear", "ETTh1")

    assert parse_pred_lengths(task, "336,96,336") == (336, 96)


def test_unsupported_model_dataset_pair_reports_supported_datasets():
    with pytest.raises(UnsupportedTaskError, match="TimeMixer.*Exchange"):
        get_task("TimeMixer", "Exchange")


def test_invalid_prediction_length_reports_allowed_values():
    task = get_task("DLinear", "ETTh1")

    with pytest.raises(ValueError, match="96, 192, 336, 720"):
        parse_pred_lengths(task, "48")


def test_batch_size_mapping_matches_selected_horizons():
    assert parse_batch_sizes((96, 192), "192:64,96:128") == {
        192: 64,
        96: 128,
    }
    assert parse_batch_sizes((96, 192), None) == {}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("96:128", "Missing batch size for prediction lengths: 192"),
        (
            "96:128,192:64,336:32",
            "Batch sizes provided for unselected prediction lengths: 336",
        ),
        (
            "96:128,96:64,192:32",
            "Duplicate batch size mapping for prediction length 96",
        ),
        ("96=128,192:64", "Expected PRED_LEN:BATCH_SIZE"),
        ("96:0,192:64", "Batch size must be a positive integer"),
        ("96:large,192:64", "Invalid batch size"),
    ],
)
def test_batch_size_mapping_rejects_invalid_values(value, message):
    with pytest.raises(ValueError, match=message):
        parse_batch_sizes((96, 192), value)
