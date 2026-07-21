# V100 32GB Baseline 训练加速参数矩阵

## 使用边界

本文参数针对单张 Tesla V100S 32GB、8 核 CPU、50GB 内存、PyTorch 2.6、CUDA 12.4。表中的 batch size 是基于当前仓库模型规模、输入长度和数据集通道数给出的“吞吐优先起始值”，不是实测最大值，也不保证每个运行环境都能达到同样显存占用。

优化目标是最大化 samples/s 并缩短单轮耗时，不是把显存强行占满。建议首次运行观察 50～100 个 step；若显存低于 70% 且吞吐仍随 batch 增加，则继续按 2 倍放大。发生 OOM 时减半。最终值建议保持在最大稳定 batch 的 75%～90%。

## 统一入口加速参数

`train_openi.py` 支持：

| 参数 | 作用 | 推荐起点 |
| --- | --- | --- |
| `--num-workers N` | DataLoader 进程数 | 通用 `4`；DLinear/FITS 可用 `6`；ILI 用 `2` |
| `--use-amp` | 向原生模型传入 AMP 开关 | 仅用于推荐开启的模型 |
| `--patience N` | 覆盖原生 early stopping patience | `3` 或 `5` |
| `--pin-memory` / `--no-pin-memory` | 锁页内存 | 开启 |
| `--persistent-workers` / `--no-persistent-workers` | epoch 间保留 worker | 开启 |
| `--prefetch-factor N` | 每个 worker 预取 batch 数 | `2` |
| `--cudnn-benchmark` / `--no-cudnn-benchmark` | 固定形状下选择更快 cuDNN 算法 | 仅卷积型模型开启 |
| `--cpu-threads N` | 同时设置 OMP/MKL 线程数 | `1` |

不传这些参数时保留各 baseline 原有行为。`--prefetch-factor` 只在 worker 数大于 0 时生效。统一入口会把 DataLoader 配置通过环境传到原生 baseline，并记录在每个任务的 `command.json` 中。

## 参数调整依据

已知资源和监控证据：

| 证据 | 观测 | 解释 |
| --- | --- | --- |
| CPU 配额 | 8 核 | 可以使用 4～6 个 DataLoader worker，同时给训练主进程和系统保留核心 |
| 内存 | 50GB | 足以启用锁页内存、persistent workers 和 `prefetch_factor=2` |
| 任务监控采样 | CPU 约 `199%` | 当时实际只使用约两个逻辑核心，不表示服务器只有两个核心 |
| 任务监控采样 | GPU 利用率 `0%`、显存约 `1.259%` | 该采样点 GPU 在等待或模型计算过轻，应优先增加 batch 和数据供给能力 |

`cpu_threads=1` 仍作为默认值，因为它限制的是每个进程内部的 OMP/MKL 线程。采用 4～6 个 worker 时，如果每个 worker 再启动多个 BLAS 线程，会超过 8 核并产生上下文切换。只有实测 `4 workers × 2 threads` 更快时才改为 `cpu_threads=2`。

## 数据集分组

| 分组 | 数据集 | 通道数 |
| --- | --- | ---: |
| L（低维） | ETTh1、ETTh2、ETTm1、ETTm2、Exchange、Weather | 7～21 |
| E（高维） | Electricity | 321 |
| T（超高维） | Traffic | 862 |
| I（短序列） | ILI | 7 |

长序列 batch 映射顺序为 `96,192,336,720`；ILI 映射顺序为 `24,36,48,60`。

## 每个模型推荐矩阵

| 模型 | Workers | AMP | Patience | cuDNN benchmark | L：96/192/336/720 | E：96/192/336/720 | T：96/192/336/720 | I：24/36/48/60 |
| --- | ---: | --- | ---: | --- | --- | --- | --- | --- |
| DLinear | `6`（ILI `2`） | 关闭 | 3 | 关闭 | `512/256/128/64` | `128/64/32/16` | `64/32/16/8` | `128/128/64/64` |
| FITS | `6`（ILI `2`） | 关闭 | 3 | 关闭 | `256/128/64/32` | `64/32/16/8` | `32/16/8/4` | `64/64/32/32` |
| FEDformer | `4`（ILI `2`） | 开启 | 3 | 关闭 | `64/32/16/8` | `16/8/4/2` | `8/4/2/1` | `32/32/16/16` |
| PatchTST | `4`（ILI `2`） | 开启 | 5 | 关闭 | `128/64/32/16` | `32/16/8/4` | `24/12/6/3` | `32/32/16/16` |
| RAFT | `4`（ILI `2`） | 开启 | 5 | 开启 | `64/32/16/8` | `16/8/4/2` | `8/4/2/1` | `32/32/16/16` |
| ST-MTM | `4`（ILI `2`） | 关闭 | 3 | 开启 | `64/32/16/8` | `16/8/4/2` | `8/4/2/1` | `32/32/16/16` |
| TFPS | `4`（ILI `2`） | 开启 | 5 | 关闭 | `64/32/16/8` | `16/8/4/2` | `8/4/2/1` | `32/32/16/16` |
| TimeMixer | `4` | 开启 | 5 | 关闭 | `256/128/64/32` | `64/32/16/8` | `32/16/8/4` | 不支持 |
| TimesNet | `4`（ILI `2`） | 开启 | 3 | 开启 | `128/64/32/16` | `16/8/4/2` | `4/2/1/1` | `8/8/4/4` |

共同推荐：`--pin-memory --persistent-workers --prefetch-factor 2 --cpu-threads 1`。worker 数按上表选择；Traffic 保持 `4`，不要因为内存充足就直接提高到 `8`。

AMP 说明：FITS 和 ST-MTM 虽然原生解析器接受 `--use_amp`，当前训练循环没有完整的 AMP 计算路径，因此矩阵中保持关闭。DLinear 计算量太小，AMP 转换开销可能抵消收益，也保持关闭。

ST-MTM 说明：统一 batch size 会同时覆盖预训练和微调阶段。表中采用预训练也能承受的保守值，避免只按微调阶段放大导致预训练 OOM。

TimesNet 说明：Electricity、Traffic 和 ILI 使用的 `d_model` 明显大于 ETT/Weather，不能沿用低维数据集的大 batch；尤其 Traffic 建议从表中值开始，不要直接翻倍。

## 推荐提交形式

当前 `--batch-size` 映射按 horizon 全局应用到本次任务中的所有模型和数据集。要使用上述矩阵，不要把九个模型与全部数据集放进同一个笛卡尔积任务；至少按“单模型 + 单数据集分组”拆分，追求最优时使用“单模型 + 单数据集”。

PatchTST 在低维数据集上的示例：

```text
--model PatchTST --dataset ETTh1,ETTh2,ETTm1,ETTm2,Exchange,Weather --pred-len all --batch-size 96:128,192:64,336:32,720:16 --num-workers 4 --use-amp --patience 5 --pin-memory --persistent-workers --prefetch-factor 2 --cpu-threads 1
```

TimesNet 在 Traffic 上的示例：

```text
--model TimesNet --dataset Traffic --pred-len all --batch-size 96:4,192:2,336:1,720:1 --num-workers 4 --use-amp --patience 3 --pin-memory --persistent-workers --prefetch-factor 2 --cudnn-benchmark --cpu-threads 1
```

DLinear 在低维数据集上的示例：

```text
--model DLinear --dataset ETTh1,ETTh2,ETTm1,ETTm2,Exchange,Weather --pred-len all --batch-size 96:512,192:256,336:128,720:64 --num-workers 6 --patience 3 --pin-memory --persistent-workers --prefetch-factor 2 --cpu-threads 1
```

## 调优判定规则

每次只改变一个因素，记录前 100 个稳定 step 的平均耗时：

1. 重型模型先固定 `num_workers=4`，DLinear/FITS 固定为 `6`，再逐级调整 batch size。
2. 对支持模型比较 AMP 开启与关闭的 samples/s 和验证指标。
3. 比较 `4×1`、`6×1` 和 `4×2`（workers × CPU threads）；以平均 epoch 时间最低者为准。ILI 先比较 `2×1` 与 `4×1`。
4. GPU 利用率持续低而 CPU 接近配额上限时，优先处理 DataLoader，不再盲目增加 batch。
5. batch 放大后先保持原学习率；若收敛变慢，再单独进行学习率实验。不要在吞吐测试阶段同时改变学习率。
6. 开启 cuDNN benchmark 后若要求严格确定性，应另跑关闭版本确认结果差异。

模型精度复现报告必须记录 batch size、AMP、patience 和 DataLoader 参数，因为这些设置可能改变优化轨迹与早停轮次。
