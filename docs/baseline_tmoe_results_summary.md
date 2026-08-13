# Dynamic-TMoE Baseline / TMoE Results Summary

## Material Passport

- Origin Skill: `experiment-agent`
- Origin Mode: `validate`
- Origin Date: `2026-08-04`
- Verification Status: `ANALYZED`
- Version Label: `baseline_tmoe_results_v1`

## 使用规则

- 本清单只整理已有文件，不启动、不重试失败任务；失败结果只存档。
- `complete_numeric` 可在注明环境、协议和来源后引用。
- `checkpoint_recovered_test` 不是完整独立训练，只能作为检查点恢复测试引用。
- `no_numeric_metric`、`failed`、`smoke_not_for_paper` 不进入论文主结果表。
- 不同环境、数据划分、训练预算和协议的数值不得直接混合比较；优先使用同一环境和同一协议的行。

## 环境注册表

| ID | 适用结果 | Python / Torch / CUDA | GPU / 平台 | 证据 |
|---|---|---|---|---|
| `baseline_openi_v1` | DLinear/FEDformer/FITS/TFPS/TimeMixer/TimesNet structured baseline artifacts | 3.11.11 / 2.6.0+cu124 / 12.4 | Tesla V100S-PCIE-32GB / Linux-4.15.0-156-generic-x86_64-with-glibc2.35 or Linux-5.15.0-112-generic-x86_64-with-glibc2.35 | `baselines/*/test_results/**/environment.json` |
| `patchtst_local_cuda` | PatchTST paper matrix | 3.8.20 / 2.4.1 / 11.8 | NVIDIA GeForce RTX 3060 / Windows local audit | `tools/run_patchtst_paper_matrix.ps1; baselines/PatchTST/requirements.txt; live environment observation on 2026-08-04` |
| `dynamic_tmoe_local_cuda` | Dynamic_TMoE reproduction queue and traffic accumulation runs | 3.10.20 / 2.4.1+cu124 / 12.4 | NVIDIA GeForce RTX 3060 / Windows local audit | `tools/run_reproduction_queue.ps1; output/traffic_accum_summary.csv; live environment observation on 2026-08-04` |

baseline_openi_v1 的 `requirements.txt` 仅是各仓库声明依赖，实际运行环境以结果目录中的 `environment.json` 为准。两个 local CUDA 环境的版本来自 2026-08-04 现场复核；结果文件的时间和运行配置仍以各自来源文件为准。

## 协议登记

- **PatchTST paper matrix**：`random_seed=2021`；数据集输入长度为 ETTh1=336、ETTh2=96、ETTm1/ETTm2=336、Weather/Exchange=96、ILI=36、Electricity=336、Traffic=96；全局 `e_layers=3, n_heads=16, d_model=128, d_ff=256, patch_len=16, stride=8, train_epochs=100, num_workers=0`；批量与学习率按构建器脚本中的数据集设置。
- **Dynamic_TMoE reproduction queue**：`num_workers=0`、`finetune_epochs=3`；ETTm2/Weather/Electricity/Traffic 有批量上限，ETTm2/Electricity/Traffic 还有缩短的 epoch/patience 预算，因此必须标记为 `reproduction_queue_modified_budget`。
- **Dynamic_TMoE Traffic gradient accumulation**：Traffic 862 通道，`seq_len=96`，micro batch=1，梯度累积 32，有效 batch=32，60 epochs，patience=10，finetune=3；这是独立于 reproduction queue 的协议。

## 覆盖概览

| 模型 | 数值记录数 | 数据集 | 环境 | 协议 |
|---|---:|---|---|---|
| DLinear | 28 | ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Weather | baseline_openi_v1 | baseline_test_results |
| Dynamic_TMoE | 40 | ETTh1, ETTh2, ETTm1, ETTm2, Exchange, electricity, national_illness, traffic, weather | dynamic_tmoe_local_cuda | reproduction_queue_modified_budget, traffic_gradient_accumulation |
| FEDformer | 14 | ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather | baseline_openi_v1 | baseline_test_results |
| PatchTST | 36 | ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Exchange, ILI, traffic, weather | patchtst_local_cuda | patchtst_paper_matrix |
| TFPS | 19 | ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather | baseline_openi_v1 | baseline_test_results |
| TimeMixer | 24 | ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather | baseline_openi_v1 | baseline_test_results |
| TimesNet | 24 | ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather | baseline_openi_v1 | baseline_test_results |

## 数值结果

字段为 `dataset / seq_len / pred_len / MSE / MAE / RSE / status / source`。完整逐条 JSON 记录见 `output/baseline_tmoe_results_inventory.json`。

### DLinear

| Dataset | Seq | Pred | MSE | MAE | RSE | Status | Protocol |
|---|---:|---:|---:|---:|---:|---|---|
| ETTh1 | 96 | 96 | 0.382911742 | 0.395928621 | 0.587769687 | `complete_numeric` | `baseline_test_results` |
| ETTh1 | 96 | 192 | 0.440913171 | 0.433454722 | 0.630570829 | `complete_numeric` | `baseline_test_results` |
| ETTh1 | 96 | 336 | 0.49979949 | 0.47589311 | 0.673054039 | `complete_numeric` | `baseline_test_results` |
| ETTh1 | 96 | 720 | 0.527022421 | 0.518599212 | 0.694970727 | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 96 | 0.32907626 | 0.380414486 | 0.462306052 | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 192 | 0.44108364 | 0.449600846 | 0.532600224 | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 336 | 0.477351725 | 0.473398834 | 0.55240649 | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 720 | 0.626816928 | 0.559174299 | 0.63281399 | `complete_numeric` | `baseline_test_results` |
| ETTm1 | 96 | 96 | 0.34940654 | 0.375577748 | 0.562467456 | `complete_numeric` | `baseline_test_results` |
| ETTm1 | 96 | 192 | 0.383798301 | 0.393663287 | 0.589729786 | `complete_numeric` | `baseline_test_results` |
| ETTm1 | 96 | 336 | 0.416551024 | 0.417553127 | 0.614161253 | `complete_numeric` | `baseline_test_results` |
| ETTm1 | 96 | 720 | 0.479412466 | 0.456780791 | 0.658757269 | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 96 | 0.186951473 | 0.281195968 | 0.350558579 | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 192 | 0.250772566 | 0.323109776 | 0.405354619 | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 336 | 0.40129596 | 0.435289562 | 0.511673689 | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 720 | 0.494874716 | 0.486261129 | 0.565447152 | `complete_numeric` | `baseline_test_results` |
| Electricity | 96 | 96 | 0.195204958 | 0.278439015 | 0.439126134 | `complete_numeric` | `baseline_test_results` |
| Electricity | 96 | 192 | 0.193951979 | 0.280406207 | 0.437874317 | `complete_numeric` | `baseline_test_results` |
| Electricity | 96 | 336 | 0.207294196 | 0.295283288 | 0.453142554 | `complete_numeric` | `baseline_test_results` |
| Electricity | 96 | 720 | 0.24219197 | 0.328852326 | 0.490916073 | `complete_numeric` | `baseline_test_results` |
| Traffic | 96 | 96 | 0.64905709 | 0.39651832 | 0.667106092 | `complete_numeric` | `baseline_test_results` |
| Traffic | 96 | 192 | 0.599535942 | 0.371907949 | 0.639052689 | `complete_numeric` | `baseline_test_results` |
| Traffic | 96 | 336 | 0.6063537 | 0.374099076 | 0.639973938 | `complete_numeric` | `baseline_test_results` |
| Traffic | 96 | 720 | 0.646616042 | 0.395827949 | 0.657542825 | `complete_numeric` | `baseline_test_results` |
| Weather | 96 | 96 | 0.198749557 | 0.257924646 | 0.587496519 | `complete_numeric` | `baseline_test_results` |
| Weather | 96 | 192 | 0.239210084 | 0.2971479 | 0.643810272 | `complete_numeric` | `baseline_test_results` |
| Weather | 96 | 336 | 0.286494434 | 0.338904947 | 0.702981055 | `complete_numeric` | `baseline_test_results` |
| Weather | 96 | 720 | 0.348310143 | 0.385389835 | 0.776631713 | `complete_numeric` | `baseline_test_results` |

### Dynamic_TMoE

| Dataset | Seq | Pred | MSE | MAE | RSE | Status | Protocol |
|---|---:|---:|---:|---:|---:|---|---|
| ETTh1 | 96 | 96 | 0.381354183 | 0.400731355 | 0.61753881 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTh1 | 96 | 192 | 0.434078246 | 0.428753555 | 0.65884614 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTh1 | 96 | 336 | 0.481827319 | 0.44770503 | 0.694137812 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTh1 | 96 | 720 | 0.477198333 | 0.466021627 | 0.690795422 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTh2 | 96 | 96 | 0.294069827 | 0.342836201 | 0.542282045 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTh2 | 96 | 192 | 0.392613083 | 0.403941691 | 0.626588464 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTh2 | 96 | 336 | 0.43548739 | 0.439690322 | 0.659914672 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTh2 | 96 | 720 | 0.426676035 | 0.44385314 | 0.653204441 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTm1 | 96 | 96 | 0.319312304 | 0.357771933 | 0.565077245 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTm1 | 96 | 192 | 0.363857597 | 0.384766638 | 0.603206098 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTm1 | 96 | 336 | 0.392926067 | 0.409869939 | 0.626838148 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTm1 | 96 | 720 | 0.461553097 | 0.449737251 | 0.679377019 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTm2 | 96 | 96 | 0.178275943 | 0.262427896 | 0.422227353 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTm2 | 96 | 192 | 0.240108564 | 0.30082795 | 0.490008742 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTm2 | 96 | 336 | 0.300191939 | 0.340272814 | 0.547897756 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| ETTm2 | 96 | 720 | 0.397246242 | 0.398943573 | 0.630274713 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| Exchange | 96 | 96 | 0.0839693695 | 0.203914315 | 0.289774686 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| Exchange | 96 | 192 | 0.1969385 | 0.315801889 | 0.443777531 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| Exchange | 96 | 336 | 0.344432682 | 0.423562527 | 0.586883903 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| Exchange | 96 | 720 | 0.865855455 | 0.701039135 | 0.930513561 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| electricity | 96 | 96 | 0.155671403 | 0.255736858 | 0.394552141 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| electricity | 96 | 192 | 0.168997104 | 0.266963094 | 0.411092574 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| electricity | 96 | 336 | 0.194515313 | 0.293700542 | 0.441038902 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| electricity | 96 | 720 | 0.213309696 | 0.306998304 | 0.461854626 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| national_illness | 36 | 24 | 1.89657819 | 0.88933152 |  | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| national_illness | 36 | 36 | 2.02615595 | 0.917231977 |  | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| national_illness | 36 | 48 | 2.30870485 | 0.939215541 |  | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| national_illness | 36 | 60 | 2.07521105 | 0.921801031 |  | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| traffic | 96 | 96 | 0.670008192 | 0.439278795 | 0.682713917 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| traffic | 96 | 96 | 0.466098292 | 0.298182384 |  | `complete_traffic_gradient_accumulation` | `traffic_gradient_accumulation` |
| traffic | 96 | 192 | 0.563761562 | 0.393285092 | 0.694306769 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| traffic | 96 | 192 | 0.48206189 | 0.306756049 |  | `complete_traffic_gradient_accumulation` | `traffic_gradient_accumulation` |
| traffic | 96 | 336 | 0.56396048 | 0.38015215 | 0.713834867 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| traffic | 96 | 336 | 0.509560218 | 0.297864033 |  | `complete_traffic_gradient_accumulation` | `traffic_gradient_accumulation` |
| traffic | 96 | 720 | 0.630705126 | 0.418450402 | 0.761378394 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| traffic | 96 | 720 | 0.579697059 | 0.335320332 |  | `complete_traffic_gradient_accumulation` | `traffic_gradient_accumulation` |
| weather | 96 | 96 | 0.15417704 | 0.200826243 | 0.392653853 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| weather | 96 | 192 | 0.206564188 | 0.249868989 | 0.454493344 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| weather | 96 | 336 | 0.261101514 | 0.289442122 | 0.510980904 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |
| weather | 96 | 720 | 0.347190827 | 0.345308691 | 0.589228988 | `complete_reproduction_queue_modified_budget` | `reproduction_queue_modified_budget` |

### FEDformer

| Dataset | Seq | Pred | MSE | MAE | RSE | Status | Protocol |
|---|---:|---:|---:|---:|---:|---|---|
| ETTh1 | 96 | 96 | 0.372811139 | 0.413488239 |  | `complete_numeric` | `baseline_test_results` |
| ETTh1 | 96 | 336 | 0.450942218 | 0.463361233 |  | `complete_numeric` | `baseline_test_results` |
| ETTh1 | 96 | 720 | 0.480381995 | 0.495394856 |  | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 96 | 0.348056406 | 0.388806552 |  | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 192 | 0.419748783 | 0.432137847 |  | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 336 | 0.478757113 | 0.477237552 |  | `complete_numeric` | `baseline_test_results` |
| ETTm1 | 96 | 720 | 0.732952833 | 0.571850181 |  | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 336 | 0.327279419 | 0.365226209 |  | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 720 | 0.438770026 | 0.430687636 |  | `complete_numeric` | `baseline_test_results` |
| Exchange | 96 | 192 | 0.27832076 | 0.386263967 |  | `complete_numeric` | `baseline_test_results` |
| Exchange | 96 | 336 | 0.4568578 | 0.4974657 |  | `complete_numeric` | `baseline_test_results` |
| Exchange | 96 | 720 | 1.15575492 | 0.822742522 |  | `complete_numeric` | `baseline_test_results` |
| Weather | 96 | 192 | 0.286468238 | 0.347703874 |  | `complete_numeric` | `baseline_test_results` |
| Weather | 96 | 720 | 0.404107541 | 0.414778441 |  | `complete_numeric` | `baseline_test_results` |

### PatchTST

| Dataset | Seq | Pred | MSE | MAE | RSE | Status | Protocol |
|---|---:|---:|---:|---:|---:|---|---|
| ETTh1 | 336 | 96 | 0.37498337 | 0.399385303 | 0.580659389 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTh1 | 336 | 192 | 0.413630813 | 0.421032131 | 0.610742509 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTh1 | 336 | 336 | 0.43137157 | 0.43569985 | 0.62784481 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTh1 | 336 | 720 | 0.449759245 | 0.466083139 | 0.644132376 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTh2 | 96 | 96 | 0.293964386 | 0.342816859 | 0.43312934 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTh2 | 96 | 192 | 0.376802266 | 0.392914593 | 0.492218941 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTh2 | 96 | 336 | 0.380636752 | 0.408982933 | 0.492881775 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTh2 | 96 | 720 | 0.411622733 | 0.433358431 | 0.514371514 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTm1 | 336 | 96 | 0.291486919 | 0.34231925 | 0.513566852 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTm1 | 336 | 192 | 0.330046505 | 0.368325204 | 0.546426833 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTm1 | 336 | 336 | 0.365893334 | 0.392120004 | 0.575537443 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTm1 | 336 | 720 | 0.41847533 | 0.423890322 | 0.61534977 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTm2 | 336 | 96 | 0.164926156 | 0.254976392 | 0.328995854 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTm2 | 336 | 192 | 0.220376849 | 0.291990101 | 0.379394889 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTm2 | 336 | 336 | 0.277708322 | 0.32931304 | 0.425165981 | `complete_numeric` | `patchtst_paper_matrix` |
| ETTm2 | 336 | 720 | 0.367481977 | 0.384874821 | 0.486707509 | `complete_numeric` | `patchtst_paper_matrix` |
| Electricity | 336 | 96 | 0.129892319 | 0.222457319 | 0.358298689 | `complete_numeric` | `patchtst_paper_matrix` |
| Electricity | 336 | 192 | 0.148991257 | 0.241320372 | 0.383946568 | `complete_numeric` | `patchtst_paper_matrix` |
| Electricity | 336 | 336 | 0.165787265 | 0.260384142 | 0.405555725 | `complete_numeric` | `patchtst_paper_matrix` |
| Electricity | 336 | 720 | 0.209554851 | 0.298031896 | 0.456868619 | `complete_numeric` | `patchtst_paper_matrix` |
| Exchange | 96 | 96 | 0.0836079344 | 0.199958488 | 0.220227063 | `complete_numeric` | `patchtst_paper_matrix` |
| Exchange | 96 | 192 | 0.175946251 | 0.297369838 | 0.323255628 | `complete_numeric` | `patchtst_paper_matrix` |
| Exchange | 96 | 336 | 0.337819874 | 0.418357432 | 0.452834547 | `complete_numeric` | `patchtst_paper_matrix` |
| Exchange | 96 | 720 | 0.907837033 | 0.71189487 | 0.751481354 | `complete_numeric` | `patchtst_paper_matrix` |
| ILI | 36 | 24 | 1.66889751 | 0.81019181 | 0.623428702 | `complete_numeric` | `patchtst_paper_matrix` |
| ILI | 36 | 36 | 1.27187514 | 0.727503181 | 0.540889144 | `complete_numeric` | `patchtst_paper_matrix` |
| ILI | 36 | 48 | 1.964589 | 0.847444057 | 0.670486212 | `complete_numeric` | `patchtst_paper_matrix` |
| ILI | 36 | 60 | 1.69559848 | 0.82697922 | 0.622223556 | `complete_numeric` | `patchtst_paper_matrix` |
| traffic | 96 | 96 | 0.446312308 | 0.283365816 | 0.553012788 | `complete_numeric` | `patchtst_paper_matrix` |
| traffic | 96 | 192 | 0.452957183 | 0.285568088 | 0.555280745 | `complete_numeric` | `patchtst_paper_matrix` |
| traffic | 96 | 336 | 0.467488766 | 0.291406035 | 0.561797798 | `complete_numeric` | `patchtst_paper_matrix` |
| traffic | 96 | 720 | 0.500667691 | 0.309526622 | 0.578517795 | `checkpoint_recovered_test` | `patchtst_paper_matrix` |
| weather | 96 | 96 | 0.177579671 | 0.219185248 | 0.555145323 | `complete_numeric` | `patchtst_paper_matrix` |
| weather | 96 | 192 | 0.224982873 | 0.259153903 | 0.624008536 | `complete_numeric` | `patchtst_paper_matrix` |
| weather | 96 | 336 | 0.278233081 | 0.29764539 | 0.692893505 | `complete_numeric` | `patchtst_paper_matrix` |
| weather | 96 | 720 | 0.350473821 | 0.345856905 | 0.780989766 | `complete_numeric` | `patchtst_paper_matrix` |

### TFPS

| Dataset | Seq | Pred | MSE | MAE | RSE | Status | Protocol |
|---|---:|---:|---:|---:|---:|---|---|
| ETTh1 | 96 | 96 | 0.423811495 | 0.433167607 | 0.617307842 | `complete_numeric` | `baseline_test_results` |
| ETTh1 | 96 | 192 | 0.473281771 | 0.461945593 | 0.653298378 | `complete_numeric` | `baseline_test_results` |
| ETTh1 | 96 | 336 | 0.519673467 | 0.471751422 | 0.68642211 | `complete_numeric` | `baseline_test_results` |
| ETTh1 | 96 | 720 | 0.517669201 | 0.487133116 | 0.688797712 | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 96 | 0.360050052 | 0.380817652 | 0.479348868 | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 192 | 0.42190975 | 0.42129755 | 0.520848393 | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 720 | 0.462459356 | 0.462165356 | 0.543559253 | `complete_numeric` | `baseline_test_results` |
| ETTm1 | 96 | 96 | 0.341788977 | 0.378925115 | 0.556117415 | `complete_numeric` | `baseline_test_results` |
| ETTm1 | 96 | 192 | 0.386715531 | 0.405614644 | 0.59196049 | `complete_numeric` | `baseline_test_results` |
| ETTm1 | 96 | 720 | 0.488254488 | 0.453584313 | 0.664801121 | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 96 | 0.177755862 | 0.26071927 | 0.341552675 | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 192 | 0.240186229 | 0.300134361 | 0.396696597 | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 720 | 0.403945595 | 0.39894408 | 0.510852456 | `complete_numeric` | `baseline_test_results` |
| Exchange | 96 | 96 | 0.105395168 | 0.227340519 | 0.247261956 | `complete_numeric` | `baseline_test_results` |
| Exchange | 96 | 192 | 0.213149205 | 0.329799622 | 0.357054234 | `complete_numeric` | `baseline_test_results` |
| Exchange | 96 | 336 | 0.394410014 | 0.462241143 | 0.489295214 | `complete_numeric` | `baseline_test_results` |
| Exchange | 96 | 720 | 1.21161151 | 0.832897544 | 0.866950631 | `complete_numeric` | `baseline_test_results` |
| Weather | 96 | 192 | 0.206755221 | 0.248841032 | 0.598353803 | `complete_numeric` | `baseline_test_results` |
| Weather | 96 | 720 | 0.34462899 | 0.343443125 | 0.772754371 | `complete_numeric` | `baseline_test_results` |

### TimeMixer

| Dataset | Seq | Pred | MSE | MAE | RSE | Status | Protocol |
|---|---:|---:|---:|---:|---:|---|---|
| ETTh1 |  | 96 | 0.372733235 | 0.3987602 |  | `complete_numeric` | `baseline_test_results` |
| ETTh1 |  | 192 | 0.433981091 | 0.433118492 |  | `complete_numeric` | `baseline_test_results` |
| ETTh1 |  | 336 | 0.502019644 | 0.460331291 |  | `complete_numeric` | `baseline_test_results` |
| ETTh1 |  | 720 | 0.488831788 | 0.473793805 |  | `complete_numeric` | `baseline_test_results` |
| ETTh2 |  | 96 | 0.289928555 | 0.343379796 |  | `complete_numeric` | `baseline_test_results` |
| ETTh2 |  | 192 | 0.39133209 | 0.408306539 |  | `complete_numeric` | `baseline_test_results` |
| ETTh2 |  | 336 | 0.405846536 | 0.427479804 |  | `complete_numeric` | `baseline_test_results` |
| ETTh2 |  | 720 | 0.42999813 | 0.442095011 |  | `complete_numeric` | `baseline_test_results` |
| ETTm1 |  | 96 | 0.34608829 | 0.379992455 |  | `complete_numeric` | `baseline_test_results` |
| ETTm1 |  | 192 | 0.372327268 | 0.387567759 |  | `complete_numeric` | `baseline_test_results` |
| ETTm1 |  | 336 | 0.394427896 | 0.406882405 |  | `complete_numeric` | `baseline_test_results` |
| ETTm1 |  | 720 | 0.461395264 | 0.448275775 |  | `complete_numeric` | `baseline_test_results` |
| ETTm2 |  | 96 | 0.180683896 | 0.263632476 |  | `complete_numeric` | `baseline_test_results` |
| ETTm2 |  | 192 | 0.240344092 | 0.30116114 |  | `complete_numeric` | `baseline_test_results` |
| ETTm2 |  | 336 | 0.300459236 | 0.339035213 |  | `complete_numeric` | `baseline_test_results` |
| ETTm2 |  | 720 | 0.396230608 | 0.401892036 |  | `complete_numeric` | `baseline_test_results` |
| Exchange |  | 96 | 0.0880035236 | 0.206758648 |  | `complete_numeric` | `baseline_test_results` |
| Exchange |  | 192 | 0.185793117 | 0.305697411 |  | `complete_numeric` | `baseline_test_results` |
| Exchange |  | 336 | 0.357508302 | 0.431721836 |  | `complete_numeric` | `baseline_test_results` |
| Exchange |  | 720 | 0.960202575 | 0.737041473 |  | `complete_numeric` | `baseline_test_results` |
| Weather |  | 96 | 0.164345384 | 0.208981946 |  | `complete_numeric` | `baseline_test_results` |
| Weather |  | 192 | 0.207783043 | 0.252374053 |  | `complete_numeric` | `baseline_test_results` |
| Weather |  | 336 | 0.26579684 | 0.292463839 |  | `complete_numeric` | `baseline_test_results` |
| Weather |  | 720 | 0.34283939 | 0.343689799 |  | `complete_numeric` | `baseline_test_results` |

### TimesNet

| Dataset | Seq | Pred | MSE | MAE | RSE | Status | Protocol |
|---|---:|---:|---:|---:|---:|---|---|
| ETTh1 | 96 | 96 | 0.497327566 | 0.480677724 |  | `complete_numeric` | `baseline_test_results` |
| ETTh1 | 96 | 192 | 0.491210401 | 0.475622654 |  | `complete_numeric` | `baseline_test_results` |
| ETTh1 | 96 | 336 | 0.505783796 | 0.477034211 |  | `complete_numeric` | `baseline_test_results` |
| ETTh1 | 96 | 720 | 0.521891236 | 0.494620591 |  | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 96 | 0.383071005 | 0.403741181 |  | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 192 | 0.398589492 | 0.411324352 |  | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 336 | 0.454434603 | 0.454641372 |  | `complete_numeric` | `baseline_test_results` |
| ETTh2 | 96 | 720 | 0.435490161 | 0.448353648 |  | `complete_numeric` | `baseline_test_results` |
| ETTm1 | 96 | 96 | 0.334737182 | 0.376352608 |  | `complete_numeric` | `baseline_test_results` |
| ETTm1 | 96 | 192 | 0.383281857 | 0.400164396 |  | `complete_numeric` | `baseline_test_results` |
| ETTm1 | 96 | 336 | 0.419849128 | 0.422424883 |  | `complete_numeric` | `baseline_test_results` |
| ETTm1 | 96 | 720 | 0.503886163 | 0.461132884 |  | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 96 | 0.188000813 | 0.268213719 |  | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 192 | 0.251110703 | 0.306258768 |  | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 336 | 0.320041507 | 0.348310471 |  | `complete_numeric` | `baseline_test_results` |
| ETTm2 | 96 | 720 | 0.43328011 | 0.414328188 |  | `complete_numeric` | `baseline_test_results` |
| Exchange | 96 | 96 | 0.11455407 | 0.243923783 |  | `complete_numeric` | `baseline_test_results` |
| Exchange | 96 | 192 | 0.214175075 | 0.336279213 |  | `complete_numeric` | `baseline_test_results` |
| Exchange | 96 | 336 | 0.346809179 | 0.435163766 |  | `complete_numeric` | `baseline_test_results` |
| Exchange | 96 | 720 | 1.01217318 | 0.770253181 |  | `complete_numeric` | `baseline_test_results` |
| Weather | 96 | 96 | 0.182479188 | 0.234082073 |  | `complete_numeric` | `baseline_test_results` |
| Weather | 96 | 192 | 0.225247517 | 0.265144706 |  | `complete_numeric` | `baseline_test_results` |
| Weather | 96 | 336 | 0.27833581 | 0.301567167 |  | `complete_numeric` | `baseline_test_results` |
| Weather | 96 | 720 | 0.354702443 | 0.350769103 |  | `complete_numeric` | `baseline_test_results` |

## 失败与不可用记录

以下项目只记录，不重试，也不作为 complete 结果使用。每条明细在 JSON 的 `failures` 中保留。

| 模型 / 数据集 | 数量或状态 | 分类 | 退出码 | 说明 | 来源 |
|---|---:|---|---:|---|---|
| Dynamic_TMoE / traffic | 6 | `superseded_failed_attempt` |  | early traffic accumulated-gradient attempt failed; a later complete row exists; no retry is requested | `output/traffic_accum_summary.csv` |
| FITS / ETTh1 | 4 | `final_artifact_failure_after_metric` | 1 | metric was printed, then test failed at np.save with ValueError for an inhomogeneous array | `baselines/FITS/test_results/multi_task_summary.csv` |
| FITS / ETTh2 | 4 | `final_artifact_failure_after_metric` | 1 | metric was printed, then test failed at np.save with ValueError for an inhomogeneous array | `baselines/FITS/test_results/multi_task_summary.csv` |
| FITS / ETTm1 | 4 | `final_artifact_failure_after_metric` | 1 | metric was printed, then test failed at np.save with ValueError for an inhomogeneous array | `baselines/FITS/test_results/multi_task_summary.csv` |
| FITS / ETTm2 | 4 | `final_artifact_failure_after_metric` | 1 | metric was printed, then test failed at np.save with ValueError for an inhomogeneous array | `baselines/FITS/test_results/multi_task_summary.csv` |
| FITS / Exchange | 4 | `final_artifact_failure_after_metric` | 1 | metric was printed, then test failed at np.save with ValueError for an inhomogeneous array | `baselines/FITS/test_results/multi_task_summary.csv` |
| FITS / Weather | 4 | `final_artifact_failure_after_metric` | 1 | metric was printed, then test failed at np.save with ValueError for an inhomogeneous array | `baselines/FITS/test_results/multi_task_summary.csv` |
| PatchTST / traffic | 1 | `full_training_failed_checkpoint_recovered_test` | -1073741819 | full training stopped at epoch 89/iter 300; stderr contains no OOM message; later test-only used the saved best checkpoint | `output/patchtst_paper_matrix_state.csv` |
| TFPS / Weather | 1 | `task_failed` | 1 | task status is failed; no finite metric row was produced | `baselines/TFPS/test_results/multi_task_summary.csv` |

### 无数值证据

| 模型 | 条目数 | 说明 | 来源 |
|---|---:|---|---|
| FEDformer | 16 | 状态文件可能为 succeeded，但没有有限 MSE/MAE；不可引用为数值结果 | `baselines/FEDformer/test_results/FEDformer/ETTh1/metrics_summary.csv`, `baselines/FEDformer/test_results/FEDformer/ETTh2/metrics_summary.csv`, `baselines/FEDformer/test_results/FEDformer/ETTm1/metrics_summary.csv`, `baselines/FEDformer/test_results/FEDformer/ETTm2/metrics_summary.csv`, `baselines/FEDformer/test_results/FEDformer/Exchange/metrics_summary.csv`, `baselines/FEDformer/test_results/FEDformer/Weather/metrics_summary.csv`, `baselines/FEDformer/test_results/重跑/FEDformer/ETTm1/metrics_summary.csv`, `baselines/FEDformer/test_results/重跑/FEDformer/ETTm2/metrics_summary.csv`, `baselines/FEDformer/test_results/重跑/FEDformer/Exchange/metrics_summary.csv`, `baselines/FEDformer/test_results/重跑/FEDformer/Weather/metrics_summary.csv` |
| TFPS | 4 | 状态文件可能为 succeeded，但没有有限 MSE/MAE；不可引用为数值结果 | `baselines/TFPS/test_results/TFPS/ETTh2/metrics_summary.csv`, `baselines/TFPS/test_results/TFPS/ETTm1/metrics_summary.csv`, `baselines/TFPS/test_results/TFPS/ETTm2/metrics_summary.csv`, `baselines/TFPS/test_results/TFPS/Weather/metrics_summary.csv` |
| RAFT | - | 当前仓库内未发现可追溯结构化数值证据 | `baselines/RAFT` |
| st-mtm | - | 当前仓库内未发现可追溯结构化数值证据 | `baselines/st-mtm` |

FITS 的 24 个任务均为 `exit_code=1`；训练日志曾输出 MSE/MAE，但测试阶段随后在 `np.save` 因不规则数组触发 `ValueError`，因此只能列为失败记录。PatchTST Traffic-720 的完整训练以 `-1073741819` 失败，后来使用已有 best checkpoint 完成 test-only，主表明确标记为 `checkpoint_recovered_test`。

## Smoke 结果（不用于论文主表）

| 模型 | 数据集 | Pred | MSE | MAE | RSE | 来源 |
|---|---|---:|---:|---:|---:|---|
| Dynamic_TMoE | ETTh1 | 96 | 0.387918085 | 0.402789265 | 0.622830689 | `results/long_term_forecast_smoke_cuda_ETTh1_96_96_Dynamic_TMoE_ETTh1_ftM_sl96_ll48_pl96_dm16_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_smoke_cuda_0/metrics.npy` |
| Dynamic_TMoE | ETTh1 | 96 | 0.389069349 | 0.402872801 | 0.623754263 | `results/long_term_forecast_smoke_ETTh1_96_96_Dynamic_TMoE_ETTh1_ftM_sl96_ll48_pl96_dm16_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_smoke_0/metrics.npy` |

## 来源清单

- `output/reproduction_summary_latest.csv`
- `output/traffic_accum_summary.csv`
- `output/patchtst_paper_matrix_state.csv`
- `output/baseline_queue_state.csv`
- `baselines/PatchTST/result.txt`
- `results/*/metrics.npy`
- `baselines/*/test_results/**/metrics_summary.csv`
- `baselines/*/test_results/**/multi_task_summary.csv`
- `baselines/*/test_results/**/environment.json`
- `tools/run_patchtst_paper_matrix.ps1`
- `tools/run_reproduction_queue.py`
- `tools/run_reproduction_queue.ps1`
- `requirements.txt`

## 推荐引用写法

引用某一行时同时写明：模型、数据集、输入长度、预测长度、环境 ID、协议、MSE/MAE/RSE 和来源文件。不要把 `reproduction_queue_modified_budget`、`traffic_gradient_accumulation`、PatchTST paper matrix 和 baseline_openi_v1 的行合并计算平均值或宣称同协议 SOTA。
