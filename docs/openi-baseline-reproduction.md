# 启智平台 Baseline 复现任务指南

## 支持范围

统一入口 `train_openi.py` 支持 DLinear、FEDformer、FITS、PatchTST、RAFT、ST-MTM、TFPS、TimeMixer 和 TimesNet。

DLinear、FEDformer、FITS、PatchTST、RAFT、ST-MTM、TFPS 和 TimesNet 支持 ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Exchange、ILI、Traffic 和 Weather。TimeMixer 按仓库现有实验脚本支持 ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Traffic 和 Weather。

常规数据集的标准预测长度为 `96,192,336,720`，ILI 为 `24,36,48,60`。ST-MTM 自动依次运行预训练和微调。

需要一次提交全部受支持实验时，请使用[全部 Baseline 实验运行参数](openi-all-baseline-run-parameters.md)。

## 构建镜像要求

镜像需要包含项目 baseline 依赖，并额外包含启智平台 SDK：

```bash
pip install c2net==0.2.0
```

提交前可在调试环境检查：

```bash
python -c "import c2net, torch; print(torch.__version__, torch.cuda.is_available())"
```

训练任务运行期间不会自动安装或降级大型机器学习依赖。

## 本地启动辅助脚本

以下 Shell 脚本用于本地调试或生成等价参数；启智训练任务必须按下一节选择 `.py` 启动文件。

运行一个模型/数据集的全部标准预测长度：

```bash
bash scripts/openi/PatchTST/ETTh1/all.sh
```

运行单个预测长度：

```bash
bash scripts/openi/PatchTST/ETTh1/pred_336.sh
```

通过统一入口指定预测长度子集：

```bash
python -u train_openi.py \
  --model PatchTST \
  --dataset ETTh1 \
  --pred-len 96,192
```

## 启智页面配置

创建训练任务时选择：

- 代码分支：包含本功能的分支；
- 数据集：`TS`；
- 镜像：已经安装 baseline 依赖和 `c2net` 的构建镜像；
- 资源：单卡 GPU；
- 启动文件：`train_openi.py`；
- 参数：按下方格式填写模型、数据集、预测长度和可选 batch size 映射。

单个预测长度参数：

```text
--model DLinear --dataset ETTh1 --pred-len 96 --batch-size 96:128
```

多个预测长度使用逗号分隔，并为每个长度提供对应 batch size：

```text
--model DLinear --dataset ETTh1 --pred-len 96,192 --batch-size 96:128,192:64
```

多个模型和数据集同样使用逗号分隔，并按模型优先的笛卡尔积顺序执行：

```text
--model DLinear,PatchTST --dataset ETTh1,Weather --pred-len 96,192 --batch-size 96:128,192:64
```

上例依次执行 `DLinear×ETTh1`、`DLinear×Weather`、`PatchTST×ETTh1` 和 `PatchTST×Weather`。所有任务在单 GPU 上串行运行；任意组合或显式预测长度不受支持时，整个矩阵会在训练前拒绝，不会静默跳过。

`--batch-size` 使用 `pred_len:batch_size` 映射格式，映射顺序不影响预测长度的执行顺序。传入该参数时，每个选中的预测长度都必须有且仅有一个映射；存在缺失、额外、重复、非整数或非正数时，任务会在启动 baseline 训练前退出。不传 `--batch-size` 时保留每个 baseline 的原始默认配置。

`--num-workers` 可选，用于统一覆盖所有选中 baseline 的 DataLoader worker 数量，必须是正整数。例如 `--num-workers 2`。不传时保留各 baseline 的原始默认值；ST-MTM 会同时应用到预训练和微调阶段。

ST-MTM 的某个预测长度设置 batch size 后，该值会同时用于对应的预训练和微调阶段。多个预测长度仍然顺序执行，不会同时占用 GPU 显存。

不传 `--pred-len` 或传入 `all` 时，每个数据集使用自身标准预测长度。矩阵同时包含普通数据集和 ILI 且需要覆盖 batch size 时，映射必须包含实际 horizon 的并集：

```text
--batch-size 24:64,36:64,48:32,60:32,96:128,192:64,336:32,720:16
```

任务入口调用 `c2net.context.prepare()` 获取实际代码、数据集和输出路径，不需要把 `/tmp/code`、`/tmp/dataset` 或 `/tmp/output` 写死在任务参数中。

## 输出目录

产物直接写入并上传自 `c2net_context.output_path`：

```text
<output_path>/<model>/<dataset>/<pred_len>/
├── checkpoints/
├── predictions/
├── native_results/
├── native_work/
├── logs/
│   ├── train.log
│   └── error.log
├── command.json
├── environment.json
├── metrics.json
└── status.json
```

数据集级目录还包含 `metrics_summary.csv`、`metrics_summary.json` 和 `status_summary.json`。

多任务输出根目录还包含 `multi_task_summary.json` 和 `multi_task_summary.csv`，按模型、数据集和预测长度记录状态及任务退出码。

`metrics.json` 至少包含 MSE 和 MAE；原模型提供的 RMSE、MAPE、MSPE、RSE 和相关系数会一并保留。`command.json` 保存实际 argv、工作目录和阶段信息，`environment.json` 保存 Python、PyTorch、CUDA 和可见 GPU 信息。

## 失败、续跑和强制重跑

一个预测长度失败后，任务继续运行其余预测长度。已有 checkpoint、日志和其他部分产物仍会上传；整个任务最终以非零退出码结束，失败原因记录在 `status.json` 和 `logs/error.log`。

多任务矩阵中的一个模型/数据集任务失败后，其余任务继续执行；只要有任意任务失败，矩阵最终返回非零退出码，并在根目录汇总中保留各任务状态。

再次提交相同任务时，已有 `status=succeeded` 的预测长度会跳过。强制重跑使用：

```text
--model DLinear --dataset ETTh1 --pred-len 96 --force
```

生成的 Shell 脚本会把额外参数原样传给统一入口。

## 提交前验证

不启动训练的本地命令生成检查：

```bash
python train_openi.py \
  --local \
  --code-root . \
  --dataset-root dataset \
  --output-root output/openi-dry-run \
  --model DLinear \
  --dataset ETTh1 \
  --pred-len 96 \
  --dry-run
```

验证生成脚本没有漂移并运行自动化测试：

```bash
python tools/generate_openi_scripts.py --check
python -m pytest tests/openi_baselines -v
```

正式提交完整矩阵前，先提交单卡冒烟任务：

```text
启动文件：train_openi.py
参数：--model DLinear --dataset ETTh1 --pred-len 96 --batch-size 96:128
```

冒烟任务通过标准：任务输出可下载，且包含 checkpoint、预测文件、含 MSE/MAE 的 `metrics.json` 和状态为 `succeeded` 的 `status.json`。
