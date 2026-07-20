# 启智平台 Baseline 复现任务指南

## 支持范围

统一入口 `train_openi.py` 支持 DLinear、FEDformer、FITS、PatchTST、RAFT、ST-MTM、TFPS、TimeMixer 和 TimesNet。

DLinear、FEDformer、FITS、PatchTST、RAFT、ST-MTM、TFPS 和 TimesNet 支持 ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Exchange、ILI、Traffic 和 Weather。TimeMixer 按仓库现有实验脚本支持 ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Traffic 和 Weather。

常规数据集的标准预测长度为 `96,192,336,720`，ILI 为 `24,36,48,60`。ST-MTM 自动依次运行预训练和微调。

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

## 任务提交粒度

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
- 启动命令：上述 `bash scripts/openi/...` 命令之一。

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

`metrics.json` 至少包含 MSE 和 MAE；原模型提供的 RMSE、MAPE、MSPE、RSE 和相关系数会一并保留。`command.json` 保存实际 argv、工作目录和阶段信息，`environment.json` 保存 Python、PyTorch、CUDA 和可见 GPU 信息。

## 失败、续跑和强制重跑

一个预测长度失败后，任务继续运行其余预测长度。已有 checkpoint、日志和其他部分产物仍会上传；整个任务最终以非零退出码结束，失败原因记录在 `status.json` 和 `logs/error.log`。

再次提交相同任务时，已有 `status=succeeded` 的预测长度会跳过。强制重跑使用：

```bash
bash scripts/openi/DLinear/ETTh1/pred_96.sh --force
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

```bash
bash scripts/openi/DLinear/ETTh1/pred_96.sh
```

冒烟任务通过标准：任务输出可下载，且包含 checkpoint、预测文件、含 MSE/MAE 的 `metrics.json` 和状态为 `succeeded` 的 `status.json`。
