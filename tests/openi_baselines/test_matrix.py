import pytest

from openi_baselines.matrix import plan_matrix
from openi_baselines.registry import parse_task_matrix


def test_plan_matrix_uses_explicit_horizons_and_global_batches():
    plans = plan_matrix(
        parse_task_matrix("DLinear,PatchTST", "ETTh1"),
        pred_len_value="96,192",
        batch_size_value="192:64,96:128",
    )

    assert [
        (plan.task.model, plan.pred_lengths, plan.batch_sizes)
        for plan in plans
    ] == [
        ("DLinear", (96, 192), {96: 128, 192: 64}),
        ("PatchTST", (96, 192), {96: 128, 192: 64}),
    ]


def test_plan_matrix_all_uses_horizon_union_for_global_batch_validation():
    plans = plan_matrix(
        parse_task_matrix("DLinear", "ETTh1,ILI"),
        pred_len_value="all",
        batch_size_value=(
            "24:8,36:8,48:8,60:8,96:32,192:16,336:8,720:4"
        ),
    )

    assert plans[0].pred_lengths == (96, 192, 336, 720)
    assert plans[0].batch_sizes == {
        96: 32,
        192: 16,
        336: 8,
        720: 4,
    }
    assert plans[1].pred_lengths == (24, 36, 48, 60)
    assert plans[1].batch_sizes == {24: 8, 36: 8, 48: 8, 60: 8}


def test_plan_matrix_rejects_explicit_horizon_invalid_for_any_task():
    with pytest.raises(ValueError, match="DLinear/ILI"):
        plan_matrix(
            parse_task_matrix("DLinear", "ETTh1,ILI"),
            pred_len_value="96",
            batch_size_value=None,
        )
