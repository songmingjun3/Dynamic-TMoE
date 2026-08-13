from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from .paths import DatasetLocation
from .types import AccelerationOptions, ProcessSpec, TaskSpec


@dataclass(frozen=True)
class CommandLayout:
    repo_root: Path
    data: DatasetLocation
    task_output: Path


DATASET_PROFILES = {
    "ETTh1": {"data": "ETTh1", "id": "ETTh1", "seq": 96, "label": 48, "channels": 7},
    "ETTh2": {"data": "ETTh2", "id": "ETTh2", "seq": 96, "label": 48, "channels": 7},
    "ETTm1": {"data": "ETTm1", "id": "ETTm1", "seq": 96, "label": 48, "channels": 7},
    "ETTm2": {"data": "ETTm2", "id": "ETTm2", "seq": 96, "label": 48, "channels": 7},
    "Electricity": {"data": "custom", "id": "Electricity", "seq": 96, "label": 48, "channels": 321},
    "Exchange": {"data": "custom", "id": "Exchange", "seq": 96, "label": 48, "channels": 8},
    "ILI": {"data": "custom", "id": "ILI", "seq": 36, "label": 18, "channels": 7},
    "Traffic": {"data": "custom", "id": "Traffic", "seq": 96, "label": 48, "channels": 862},
    "Weather": {"data": "custom", "id": "Weather", "seq": 96, "label": 48, "channels": 21},
}


def _extend(argv: list[str], **options: object) -> None:
    for name, value in options.items():
        if value is None:
            continue
        argv.extend((f"--{name}", str(value)))


def _base_process(
    task: TaskSpec,
    layout: CommandLayout,
    *,
    stage: str,
    argv: list[str],
) -> ProcessSpec:
    cwd = layout.task_output / "native_work" / stage
    cwd.mkdir(parents=True, exist_ok=True)
    entrypoint = layout.repo_root / task.entrypoint
    python_path = str(layout.repo_root)
    inherited_python_path = os.environ.get("PYTHONPATH")
    if inherited_python_path:
        python_path = os.pathsep.join((python_path, inherited_python_path))
    return ProcessSpec(
        stage=stage,
        argv=(sys.executable, "-u", str(entrypoint), *argv),
        cwd=cwd,
        env={
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": python_path,
        },
    )


def _common(task: TaskSpec, pred_len: int, layout: CommandLayout) -> tuple[dict, list[str]]:
    profile = dict(DATASET_PROFILES[task.dataset])
    checkpoint_dir = layout.task_output / "checkpoints"
    argv: list[str] = []
    return profile, argv


def _dlinear(task: TaskSpec, pred_len: int, layout: CommandLayout) -> ProcessSpec:
    profile, argv = _common(task, pred_len, layout)
    batches = {"ETTm1": 8, "Electricity": 16, "Traffic": 16, "Weather": 16}
    rates = {
        "ETTh1": 0.005,
        "ETTh2": 0.05,
        "ETTm1": 0.0001,
        "ETTm2": 0.001 if pred_len <= 192 else 0.01,
        "Electricity": 0.001,
        "Exchange": 0.0005,
        "ILI": 0.01,
        "Traffic": 0.05,
    }
    _extend(
        argv,
        is_training=1,
        root_path=layout.data.root_path,
        data_path=layout.data.file.name,
        model_id=f"{profile['id']}_{profile['seq']}_{pred_len}",
        model=task.native_model,
        data=profile["data"],
        features="M",
        seq_len=profile["seq"],
        label_len=profile["label"],
        pred_len=pred_len,
        enc_in=profile["channels"],
        des="Exp",
        itr=1,
        batch_size=batches.get(task.dataset, 32),
        learning_rate=rates.get(task.dataset),
        gpu=0,
        checkpoints="./checkpoints",
    )
    return _base_process(task, layout, stage="train", argv=argv)


def _fedformer(task: TaskSpec, pred_len: int, layout: CommandLayout) -> ProcessSpec:
    profile, argv = _common(task, pred_len, layout)
    _extend(
        argv,
        is_training=1,
        root_path=layout.data.root_path,
        data_path=layout.data.file.name,
        task_id=profile["id"],
        model=task.native_model,
        data=profile["data"],
        features="M",
        seq_len=profile["seq"],
        label_len=profile["label"],
        pred_len=pred_len,
        e_layers=2,
        d_layers=1,
        enc_in=profile["channels"],
        dec_in=profile["channels"],
        c_out=profile["channels"],
        d_model=512,
        des="Exp",
        itr=1,
        gpu=0,
        checkpoints="./checkpoints",
    )
    return _base_process(task, layout, stage="train", argv=argv)


def _fits(task: TaskSpec, pred_len: int, layout: CommandLayout) -> ProcessSpec:
    profile, argv = _common(task, pred_len, layout)
    orders = {
        "ETTh1": 6,
        "ETTh2": 6,
        "ETTm1": 14,
        "ETTm2": 14,
        "Electricity": 10,
        "Exchange": 6,
        "ILI": 6,
        "Traffic": 10,
        "Weather": 10,
    }
    batches = {"ILI": 16, "Exchange": 32, "Weather": 32}
    _extend(
        argv,
        is_training=1,
        root_path=layout.data.root_path,
        data_path=layout.data.file.name,
        model_id=f"{profile['id']}_{profile['seq']}_{pred_len}",
        model=task.native_model,
        data=profile["data"],
        features="M",
        seq_len=profile["seq"],
        pred_len=pred_len,
        enc_in=profile["channels"],
        des="Exp",
        train_mode=2 if task.dataset == "ETTm2" else 1,
        H_order=orders[task.dataset],
        seed=2021,
        patience=20,
        itr=1,
        batch_size=batches.get(task.dataset, 64),
        learning_rate=0.0005,
        gpu=0,
        checkpoints="./checkpoints",
    )
    return _base_process(task, layout, stage="train", argv=argv)


PATCH_CONFIG = {
    "ETTh1": (336, 4, 16, 128, 0.3, 128, 0.0001),
    "ETTh2": (96, 4, 16, 128, 0.3, 128, 0.0001),
    "ETTm1": (336, 16, 128, 256, 0.2, 128, 0.0001),
    "ETTm2": (336, 16, 128, 256, 0.2, 128, 0.0001),
    "Electricity": (336, 16, 128, 256, 0.2, 32, 0.0001),
    "Exchange": (96, 4, 128, 256, 0.2, 32, 0.0001),
    "ILI": (36, 4, 16, 128, 0.3, 16, 0.0025),
    "Traffic": (96, 16, 128, 256, 0.2, 24, 0.0001),
    "Weather": (96, 16, 128, 256, 0.2, 128, 0.0001),
}


def _patchtst(task: TaskSpec, pred_len: int, layout: CommandLayout) -> ProcessSpec:
    profile, argv = _common(task, pred_len, layout)
    seq_len, heads, d_model, d_ff, dropout, batch, rate = PATCH_CONFIG[task.dataset]
    _extend(
        argv,
        random_seed=2021,
        is_training=1,
        root_path=layout.data.root_path,
        data_path=layout.data.file.name,
        model_id=f"{profile['id']}_{seq_len}_{pred_len}",
        model=task.native_model,
        data=profile["data"],
        features="M",
        seq_len=seq_len,
        label_len=profile["label"],
        pred_len=pred_len,
        enc_in=profile["channels"],
        e_layers=3,
        n_heads=heads,
        d_model=d_model,
        d_ff=d_ff,
        dropout=dropout,
        fc_dropout=dropout,
        head_dropout=0,
        patch_len=24 if task.dataset == "ILI" else 16,
        stride=2 if task.dataset == "ILI" else 8,
        des="Exp",
        train_epochs=100,
        itr=1,
        batch_size=batch,
        learning_rate=rate,
        gpu=0,
        checkpoints="./checkpoints",
    )
    return _base_process(task, layout, stage="train", argv=argv)


def _raft(task: TaskSpec, pred_len: int, layout: CommandLayout) -> ProcessSpec:
    profile, argv = _common(task, pred_len, layout)
    _extend(
        argv,
        task_name="long_term_forecast",
        is_training=1,
        model_id=f"{profile['id']}_{profile['seq']}_{pred_len}",
        model=task.native_model,
        data=profile["data"],
        root_path=layout.data.root_path,
        data_path=layout.data.file.name,
        features="M",
        seq_len=profile["seq"],
        label_len=profile["label"],
        pred_len=pred_len,
        enc_in=profile["channels"],
        dec_in=profile["channels"],
        c_out=profile["channels"],
        gpu=0,
        checkpoints="./checkpoints",
    )
    return _base_process(task, layout, stage="train", argv=argv)


def _stmtm(task: TaskSpec, pred_len: int, layout: CommandLayout) -> tuple[ProcessSpec, ...]:
    profile, _ = _common(task, pred_len, layout)
    pretrain_dir = layout.task_output / "pretrain_checkpoints"
    checkpoint_dir = layout.task_output / "checkpoints"
    architecture = {
        "e_layers": 1,
        "enc_in": profile["channels"],
        "dec_in": profile["channels"],
        "c_out": profile["channels"],
        "d_model": 16,
        "d_ff": 64,
        "n_heads": 8,
        "kernel_size": 200,
        "seg_len": 25,
        "p_tmask": 0.2,
        "topk": 3,
    }
    common = {
        "root_path": layout.data.root_path,
        "data_path": layout.data.file.name,
        "model_id": "STMTM",
        "model": "STMTM",
        "data": profile["data"] if profile["data"] != "custom" else profile["id"],
        "features": "M",
        "seq_len": profile["seq"],
        **architecture,
        "gpu": 0,
        "pretrain_checkpoints": pretrain_dir,
    }
    pretrain_argv: list[str] = []
    _extend(
        pretrain_argv,
        task_name="pretrain",
        **common,
        learning_rate=0.001,
        batch_size=16 if task.dataset.startswith("ETT") else 32,
        train_epochs=10 if task.dataset in {"Electricity", "Traffic"} else 50,
    )
    finetune_argv: list[str] = []
    _extend(
        finetune_argv,
        task_name="finetune",
        is_training=1,
        **common,
        pred_len=pred_len,
        learning_rate=0.0001,
        dropout=0.2,
        batch_size=128,
        checkpoints=checkpoint_dir,
    )
    return (
        _base_process(task, layout, stage="pretrain", argv=pretrain_argv),
        _base_process(task, layout, stage="finetune", argv=finetune_argv),
    )


def _tfps(task: TaskSpec, pred_len: int, layout: CommandLayout) -> ProcessSpec:
    profile, argv = _common(task, pred_len, layout)
    expert_count = {96: 16, 192: 4, 336: 4, 720: 8, 24: 4, 36: 4, 48: 4, 60: 4}[pred_len]
    learning_rate = 0.005 if pred_len == 336 else 0.0005
    _extend(
        argv,
        random_seed=2021,
        is_training=1,
        root_path=layout.data.root_path,
        data_path=layout.data.file.name,
        model_id=f"{profile['id']}_{profile['seq']}_{pred_len}",
        model=task.native_model,
        data=profile["data"],
        features="M",
        seq_len=profile["seq"],
        label_len=profile["label"],
        pred_len=pred_len,
        enc_in=profile["channels"],
        e_layers=3,
        n_heads=4,
        d_model=128,
        d_ff=128,
        dropout=0.3,
        fc_dropout=0.3,
        head_dropout=0,
        patch_len=16,
        stride=8,
        T_num_expert=expert_count,
        T_top_k=1,
        F_num_expert=expert_count,
        F_top_k=1,
        beta=0.1,
        des="Exp",
        train_epochs=10,
        itr=1,
        batch_size=128,
        learning_rate=learning_rate,
        gpu=0,
        checkpoints="./checkpoints",
    )
    return _base_process(task, layout, stage="train", argv=argv)


def _timemixer(task: TaskSpec, pred_len: int, layout: CommandLayout) -> ProcessSpec:
    profile, argv = _common(task, pred_len, layout)
    _extend(
        argv,
        task_name="long_term_forecast",
        is_training=1,
        root_path=layout.data.root_path,
        data_path=layout.data.file.name,
        model_id=f"{profile['id']}_96_{pred_len}",
        model=task.native_model,
        data=profile["data"],
        features="M",
        seq_len=96,
        label_len=0,
        pred_len=pred_len,
        e_layers=2,
        enc_in=profile["channels"],
        dec_in=profile["channels"],
        c_out=profile["channels"],
        des="Exp",
        itr=1,
        d_model=16,
        d_ff=32,
        learning_rate=0.01,
        train_epochs=10,
        patience=10,
        batch_size=128,
        down_sampling_layers=3,
        down_sampling_method="avg",
        down_sampling_window=2,
        gpu=0,
        checkpoints="./checkpoints",
    )
    return _base_process(task, layout, stage="train", argv=argv)


TIMESNET_DIMENSIONS = {
    "ETTh1": (16, 32),
    "ETTh2": (32, 32),
    "ETTm1": (64, 64),
    "ETTm2": (32, 32),
    "Electricity": (256, 512),
    "Exchange": (64, 64),
    "ILI": (768, 768),
    "Traffic": (512, 512),
    "Weather": (32, 32),
}


def _timesnet(task: TaskSpec, pred_len: int, layout: CommandLayout) -> ProcessSpec:
    profile, argv = _common(task, pred_len, layout)
    d_model, d_ff = TIMESNET_DIMENSIONS[task.dataset]
    _extend(
        argv,
        task_name="long_term_forecast",
        is_training=1,
        root_path=layout.data.root_path,
        data_path=layout.data.file.name,
        model_id=f"{profile['id']}_{profile['seq']}_{pred_len}",
        model=task.native_model,
        data=profile["data"],
        features="M",
        seq_len=profile["seq"],
        label_len=profile["label"],
        pred_len=pred_len,
        e_layers=2,
        d_layers=1,
        factor=3,
        enc_in=profile["channels"],
        dec_in=profile["channels"],
        c_out=profile["channels"],
        d_model=d_model,
        d_ff=d_ff,
        top_k=5,
        des="Exp",
        itr=1,
        gpu=0,
        checkpoints="./checkpoints",
    )
    return _base_process(task, layout, stage="train", argv=argv)


BUILDERS = {
    "DLinear": _dlinear,
    "FEDformer": _fedformer,
    "FITS": _fits,
    "PatchTST": _patchtst,
    "RAFT": _raft,
    "ST-MTM": _stmtm,
    "TFPS": _tfps,
    "TimeMixer": _timemixer,
    "TimesNet": _timesnet,
}


def _with_batch_size(process: ProcessSpec, batch_size: int) -> ProcessSpec:
    argv = list(process.argv)
    option = "--batch_size"
    if option in argv:
        argv[argv.index(option) + 1] = str(batch_size)
    else:
        argv.extend((option, str(batch_size)))
    return replace(process, argv=tuple(argv))


def _with_num_workers(process: ProcessSpec, num_workers: int) -> ProcessSpec:
    argv = list(process.argv)
    option = "--num_workers"
    if option in argv:
        argv[argv.index(option) + 1] = str(num_workers)
    else:
        argv.extend((option, str(num_workers)))
    return replace(process, argv=tuple(argv))


def _with_value_option(
    process: ProcessSpec, option: str, value: object
) -> ProcessSpec:
    argv = list(process.argv)
    if option in argv:
        argv[argv.index(option) + 1] = str(value)
    else:
        argv.extend((option, str(value)))
    return replace(process, argv=tuple(argv))


def _with_flag(process: ProcessSpec, option: str) -> ProcessSpec:
    if option in process.argv:
        return process
    return replace(process, argv=(*process.argv, option))


def _with_acceleration(
    process: ProcessSpec, options: AccelerationOptions
) -> ProcessSpec:
    if options.use_amp:
        process = _with_flag(process, "--use_amp")
    if options.patience is not None:
        process = _with_value_option(
            process, "--patience", options.patience
        )

    env = dict(process.env)
    environment_options = {
        "OPENI_PIN_MEMORY": options.pin_memory,
        "OPENI_PERSISTENT_WORKERS": options.persistent_workers,
        "OPENI_PREFETCH_FACTOR": options.prefetch_factor,
        "OPENI_CUDNN_BENCHMARK": options.cudnn_benchmark,
    }
    for name, value in environment_options.items():
        if value is not None:
            env[name] = (
                "1" if isinstance(value, bool) and value else
                "0" if isinstance(value, bool) else str(value)
            )
    if options.cpu_threads is not None:
        env["OMP_NUM_THREADS"] = str(options.cpu_threads)
        env["MKL_NUM_THREADS"] = str(options.cpu_threads)
    return replace(process, env=env)


def build_processes(
    task: TaskSpec,
    pred_len: int,
    layout: CommandLayout,
    batch_size: int | None = None,
    num_workers: int | None = None,
    acceleration: AccelerationOptions | None = None,
) -> tuple[ProcessSpec, ...]:
    if pred_len not in task.horizons:
        raise ValueError(f"Unsupported prediction length {pred_len} for {task.model}/{task.dataset}")
    result = BUILDERS[task.model](task, pred_len, layout)
    if isinstance(result, ProcessSpec):
        processes = (result,)
    else:
        processes = result
    if batch_size is not None:
        processes = tuple(
            _with_batch_size(process, batch_size) for process in processes
        )
    if num_workers is not None:
        processes = tuple(
            _with_num_workers(process, num_workers) for process in processes
        )
    if acceleration is not None:
        processes = tuple(
            _with_acceleration(process, acceleration)
            for process in processes
        )
    return processes
