# OpenI 全部 Baseline 实验运行参数文档设计

## 目标

新增一份可直接用于启智训练任务提交的运行参数文档，以两个大型矩阵任务覆盖仓库支持的全部 baseline 实验。文档面向单卡 Tesla V100 32GB，提供明确的启动文件、参数、覆盖数量、全局 batch size、续跑和产物检查说明。

## 文档位置

最终文档保存为：

```text
docs/openi-all-baseline-run-parameters.md
```

现有 `docs/openi-baseline-reproduction.md` 增加指向该完整参数清单的链接，不重复完整命令。

## 平台配置

文档明确列出启智页面字段：

- 代码分支：`codex/openi-baseline-runner`；
- 启动文件：`train_openi.py`；
- 数据集：`TS`；
- 镜像：已安装项目 baseline 依赖和 `c2net`；
- 资源：单卡 Tesla V100 32GB；
- 参数：复制对应矩阵任务的单行参数。

## 实验覆盖

矩阵任务 A 使用全部 9 个模型：

```text
DLinear,FEDformer,FITS,PatchTST,RAFT,ST-MTM,TFPS,TimeMixer,TimesNet
```

以及全部模型共同支持的 7 个数据集：

```text
ETTh1,ETTh2,ETTm1,ETTm2,Electricity,Traffic,Weather
```

任务 A 产生 `9 × 7 = 63` 个模型/数据集组合，每个组合运行 4 个标准 horizon，共 `252` 次训练。

矩阵任务 B 使用除 TimeMixer 外的 8 个模型：

```text
DLinear,FEDformer,FITS,PatchTST,RAFT,ST-MTM,TFPS,TimesNet
```

以及：

```text
Exchange,ILI
```

任务 B 产生 `8 × 2 = 16` 个模型/数据集组合。Exchange 运行 `96,192,336,720`，ILI 运行 `24,36,48,60`，共 `64` 次训练。

两个任务合计覆盖注册表中的 `79` 个模型/数据集组合和 `316` 次 horizon 训练，不包含仓库未支持的 `TimeMixer × Exchange` 与 `TimeMixer × ILI`。

## Batch Size 配置

任务 A 使用普通 horizon 的平衡型 V100 32GB 映射：

```text
96:32,192:16,336:8,720:8
```

任务 B 的全局映射必须覆盖 Exchange 和 ILI horizon 的并集：

```text
24:16,36:16,48:8,60:8,96:32,192:16,336:8,720:8
```

文档明确提示：`--batch-size` 会覆盖 baseline 原始 batch size。该配置以降低大型模型和高维数据集在 V100 32GB 上的 OOM 风险为目标，不是论文原始 batch 超参数；若需要严格保持原始设置，应删除整个 `--batch-size` 参数。

## 运行行为

两个矩阵任务均使用 `--pred-len all`，在单 GPU 上按模型优先的笛卡尔积顺序串行执行。

文档说明：

- 单个 horizon 失败后继续该模型/数据集的其他 horizon；
- 单个模型/数据集失败后继续矩阵中的剩余任务；
- 任意失败会使矩阵最终返回非零退出码；
- 再次提交到同一输出上下文时，`status=succeeded` 的 horizon 自动跳过；
- `--force` 会强制重跑矩阵中的全部任务；
- 两个大型矩阵可能受平台最长运行时间限制。

## 超时拆分方案

文档保留两个矩阵作为主要推荐，同时提供按模型数组拆分的恢复方法。拆分只改变 `--model` 列表，不改变数据集、horizon 或 batch 映射，并通过各任务独立输出继续覆盖原矩阵。

建议拆分为三组：

1. `DLinear,FEDformer,FITS`
2. `PatchTST,RAFT,ST-MTM`
3. `TFPS,TimeMixer,TimesNet`（任务 B 删除 TimeMixer）

文档不要求默认提交拆分任务；仅在平台超时或需要缩短单次占卡时间时使用。

## 产物检查

文档列出完成标准：

- 输出根目录存在 `multi_task_summary.json` 和 `multi_task_summary.csv`；
- 每个 `<model>/<dataset>/<pred_len>/` 包含 `status.json`；
- 成功任务包含 `metrics.json`，至少有 MSE 和 MAE；
- checkpoint、预测文件、日志和原生结果已归一化保存；
- 汇总中任务 A 应有 252 行，任务 B 应有 64 行；
- 最终确认 316 行 horizon 状态均已收集。

## 验证与交付

生成最终文档后执行两个矩阵参数的本地 `--dry-run`：

- 验证任务 A 产生 63 个任务和 252 行状态；
- 验证任务 B 产生 16 个任务和 64 行状态；
- 验证所有原生命令包含对应 horizon 的全局 `--batch_size`；
- 验证 `python -m pytest tests/openi_baselines -q` 仍通过；
- 验证 Markdown 和 Git 差异无空白错误。

文档提交到 `codex/openi-baseline-runner` 后，同步推送 GitHub `origin` 和启智 `openi` 两个远程仓库。

## 非目标

- 不修改训练入口、runner 或 baseline 模型代码；
- 不新增第三个默认矩阵任务；
- 不自动估算任务时长；
- 不声称全局 batch 映射与论文原始超参数完全一致；
- 不自动提交或启动启智训练任务。
