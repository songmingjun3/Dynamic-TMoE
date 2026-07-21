# 启智平台全部 Baseline 实验运行参数

本文给出两组可直接复制到启智训练任务“参数”栏的配置，用两个大型矩阵任务覆盖仓库支持的全部 baseline 实验。

## 适用环境

| 配置项 | 值 |
| --- | --- |
| 代码分支 | `codex/openi-baseline-runner` |
| 启动文件 | `train_openi.py` |
| 数据集 | `TS` |
| 镜像 | 已安装项目 baseline 依赖和 `c2net` |
| GPU | 单卡 Tesla V100 32GB |
| 执行方式 | 单 GPU 串行笛卡尔积 |

本方案覆盖 9 个 baseline、79 个受支持的模型/数据集组合和 316 次 horizon 训练。仓库不支持的 `TimeMixer × Exchange` 与 `TimeMixer × ILI` 不包含在实验矩阵中。

## 覆盖总览

| 任务 | 模型数 | 数据集数 | 组合数 | Horizon 训练数 |
| --- | ---: | ---: | ---: | ---: |
| A：共同数据集 | 9 | 7 | 63 | 252 |
| B：Exchange 与 ILI | 8 | 2 | 16 | 64 |
| 合计 | 9 | 9 | 79 | 316 |

两个启智任务彼此独立。每个任务内部按模型优先顺序执行笛卡尔积，同一时间只运行一个模型/数据集/horizon。

## 任务 A：全部模型 × 共同数据集

建议任务名称：

```text
all-baselines-common-datasets
```

模型：

```text
DLinear,FEDformer,FITS,PatchTST,RAFT,ST-MTM,TFPS,TimeMixer,TimesNet
```

数据集：

```text
ETTh1,ETTh2,ETTm1,ETTm2,Electricity,Traffic,Weather
```

启智页面选择启动文件 `train_openi.py`，参数栏完整复制以下单行内容：

```text
--model DLinear,FEDformer,FITS,PatchTST,RAFT,ST-MTM,TFPS,TimeMixer,TimesNet --dataset ETTh1,ETTh2,ETTm1,ETTm2,Electricity,Traffic,Weather --pred-len all --batch-size 96:32,192:16,336:8,720:8
```

该任务生成 `9 × 7 = 63` 个模型/数据集组合。每个组合依次运行 `96,192,336,720`，共 `63 × 4 = 252` 次 horizon 训练。

## 任务 B：除 TimeMixer 外全部模型 × Exchange/ILI

建议任务名称：

```text
all-baselines-exchange-ili
```

模型：

```text
DLinear,FEDformer,FITS,PatchTST,RAFT,ST-MTM,TFPS,TimesNet
```

数据集：

```text
Exchange,ILI
```

启智页面选择启动文件 `train_openi.py`，参数栏完整复制以下单行内容：

```text
--model DLinear,FEDformer,FITS,PatchTST,RAFT,ST-MTM,TFPS,TimesNet --dataset Exchange,ILI --pred-len all --batch-size 24:16,36:16,48:8,60:8,96:32,192:16,336:8,720:8
```

该任务生成 `8 × 2 = 16` 个模型/数据集组合。Exchange 运行 `96,192,336,720`，ILI 运行 `24,36,48,60`，共 `16 × 4 = 64` 次 horizon 训练。

## Batch Size 说明

上述 `--batch-size` 是针对 V100 32GB 的全局保守配置，会覆盖所有 baseline 的原始 batch size：

| Horizon | Batch Size |
| ---: | ---: |
| 24 | 16 |
| 36 | 16 |
| 48 | 8 |
| 60 | 8 |
| 96 | 32 |
| 192 | 16 |
| 336 | 8 |
| 720 | 8 |

这套配置用于降低大型模型和高维数据集的 OOM 风险，但不等同于每篇论文的原始训练超参数。由于 batch size 会影响优化过程，使用该配置获得的指标应标注为“V100 统一 batch 配置”。

如果目标是严格保留仓库中的原始 batch size，请从两个任务参数中完整删除 `--batch-size ...`，不要只删除部分 horizon 映射。未传该参数时，每个 baseline 会使用仓库内置配置。

如果仍发生 CUDA OOM，需要为当前矩阵中的每个 horizon 提供完整的新映射。例如将普通 horizon 进一步降低为：

```text
--batch-size 96:16,192:8,336:4,720:4
```

任务 B 同时包含普通 horizon 和 ILI horizon，修改时必须继续覆盖全部 8 个长度。

## 执行顺序与失败处理

- 两个矩阵任务都在单 GPU 上串行执行，不会同时把多个 baseline 放入显存。
- 一个 horizon 失败后，继续该模型/数据集的其他 horizon。
- 一个模型/数据集组合失败后，继续矩阵中的后续组合。
- 只要有任意实验失败，矩阵任务最终返回非零退出码。
- ST-MTM 的每个 horizon 会依次运行预训练和微调，统一使用该 horizon 对应的 batch size。
- `status=succeeded` 的 horizon 只有在再次运行时仍能访问同一输出目录，才会自动跳过。
- 在参数末尾添加 `--force` 会强制重跑矩阵中的全部实验。

## 平台时限不足时的拆分方法

任务 A 包含 252 次训练，任务 B 包含 64 次训练，实际运行时间可能超过平台单任务时限。遇到超时或需要缩短单次占卡时间时，只拆分 `--model` 数组，其他参数保持不变。

建议模型分组：

```text
DLinear,FEDformer,FITS
PatchTST,RAFT,ST-MTM
TFPS,TimeMixer,TimesNet
```

拆分任务 A 时，将原命令中的 `--model` 分别替换为以上三组。

拆分任务 B 时使用：

```text
DLinear,FEDformer,FITS
PatchTST,RAFT,ST-MTM
TFPS,TimesNet
```

任务 B 的第三组删除 TimeMixer，因为仓库不支持 TimeMixer 在 Exchange 和 ILI 上的实验。拆分不会改变总覆盖范围，但不同启智任务的产物位于各自输出目录，需要在下载后合并汇总。

## 输出产物

每个 horizon 的主要产物位于：

```text
<output>/<model>/<dataset>/<pred_len>/
├── checkpoints/
├── predictions/
├── native_results/
├── logs/
├── command.json
├── environment.json
├── metrics.json
└── status.json
```

每个大型矩阵任务的输出根目录还应包含：

```text
multi_task_summary.json
multi_task_summary.csv
```

## 实验完成检查清单

- [ ] 任务 A 的 `multi_task_summary` 包含 252 行 horizon 状态。
- [ ] 任务 B 的 `multi_task_summary` 包含 64 行 horizon 状态。
- [ ] 两个任务合计收集到 316 行状态。
- [ ] 每个成功行对应的目录存在 `status.json` 和 `metrics.json`。
- [ ] `metrics.json` 至少包含 MSE 和 MAE。
- [ ] 需要保存模型的实验已产生 checkpoint。
- [ ] 预测文件、训练日志和原生结果已经下载。
- [ ] 汇总中没有未处理的 `failed` 状态。

若存在失败行，先查看对应 `<model>/<dataset>/<pred_len>/logs/error.log` 或 `matrix_error.log`，调整 batch size 或依赖后，仅重新提交失败范围，避免无条件使用 `--force` 重跑全部 316 次实验。

## 提交前冒烟验证

正式提交两个大型矩阵前，先运行一个小任务确认镜像、数据集和 GPU：

```text
--model DLinear --dataset ETTh1 --pred-len 96 --batch-size 96:32
```

日志应出现 `Use GPU: cuda:0`，任务输出应包含状态为 `succeeded` 的 `status.json` 和含 MSE/MAE 的 `metrics.json`。
