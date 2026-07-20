# Dynamic-TMoE 最值得复现的 12 篇论文

检索与核验日期：2026-07-18  
筛选母集：[近三年 50 篇文献](./literature_with_code_50.md)  
引用量口径：OpenAlex `cited_by_count`，为查询日快照；不同于 Google Scholar 或 Semantic Scholar，数值不可直接横向混用。

## 筛选方法

综合评分考虑：与 Dynamic-TMoE 的直接相关性（40%）、代码与实验脚本可用性（25%）、可组成互补消融/基线的价值（20%）、学术影响与复用价值（15%）。最终 12 篇覆盖动态融合、非平稳性、patch/周期建模、外生变量、基础模型、轻量模型与公平评测。

## 推荐清单（按复现优先级）

| 优先级 | 论文与作者 | Venue | DOI | OpenAlex 引用量 | 论文 / 源码 |
|---:|---|---|---|---:|---|
| 1 | **Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting (TimeFuse)** — Zhining Liu, Ze Yang, Lin Xiao, Ruizhong Qiu, Tianxin Wei, Yada Zhu, Hendrik F. Hamann, Jingrui He, Hanghang Tong | ICML 2025 | [10.48550/arXiv.2505.18442](https://doi.org/10.48550/arxiv.2505.18442)¹ | 0 | [论文](https://arxiv.org/abs/2505.18442) · [源码](https://github.com/ZhiningLiu1998/TimeFuse) |
| 2 | **TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting** — Peiyuan Liu, Beiliang Wu, Yifan Hu, Naiqi Li, Tao Dai, Jigang Bao, Shu-Tao Xia | ICML 2025 | [10.48550/arXiv.2410.04442](https://doi.org/10.48550/arxiv.2410.04442)¹ | 4 | [论文](https://arxiv.org/abs/2410.04442) · [源码](https://github.com/Hank0626/TimeBridge) |
| 3 | **Frequency Adaptive Normalization for Non-stationary Time Series Forecasting (FAN)** — Songgaojun Deng, Ning Gui, Weiwei Ye, Ning Gui | NeurIPS 2024 | [10.52202/079017-0985](https://doi.org/10.52202/079017-0985) | 16 | [论文](https://doi.org/10.52202/079017-0985) · [源码](https://github.com/wayne155/FAN) |
| 4 | **TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting** — Yifan Hu, Guibin Zhang, Peiyuan Liu, Disen Lan, Naiqi Li, Dawei Cheng, Tao Dai, Shu-Tao Xia, Shirui Pan | ICML 2025 | [10.48550/arXiv.2501.13041](https://doi.org/10.48550/arxiv.2501.13041)¹ | 1 | [论文](https://arxiv.org/abs/2501.13041) · [源码](https://github.com/TROUBADOUR000/TimeFilter) |
| 5 | **CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns** — Xinyi Hu, Shengsheng Lin, Weiwei Lin, Ruichao Mo, Wentai Wu, Hao Zhong | NeurIPS 2024 Spotlight | [10.52202/079017-3373](https://doi.org/10.52202/079017-3373) | 17 | [论文](https://doi.org/10.52202/079017-3373) · [源码](https://github.com/ACAT-SCUT/CycleNet) |
| 6 | **TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables** — Jiaxiang Dong, Yongxin Liu, Mingsheng Long, Qin Guo, Yunzhong Qiu, Jianmin Wang, Yuxuan Wang, Haixu Wu, Haoran Zhang | NeurIPS 2024 | [10.52202/079017-0015](https://doi.org/10.52202/079017-0015) | 86 | [论文](https://doi.org/10.52202/079017-0015) · [源码](https://github.com/thuml/TimeXer) |
| 7 | **Heterogeneity-Informed Meta-Parameter Learning for Spatiotemporal Time Series Forecasting (HimNet)** — Zheng Dong, Renhe Jiang, Haotian Gao, Hangchen Liu, Jinliang Deng, Qingsong Wen, Xuan Song | KDD 2024 | [10.1145/3637528.3671961](https://doi.org/10.1145/3637528.3671961) | 48 | [论文](https://doi.org/10.1145/3637528.3671961) · [源码](https://github.com/XDZhelheim/HimNet) |
| 8 | **Unified Training of Universal Time Series Forecasting Transformers (Moirai)** — Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, Doyen Sahoo | ICML 2024 | [10.48550/arXiv.2402.02592](https://doi.org/10.48550/arxiv.2402.02592)¹ | 32 | [论文](https://arxiv.org/abs/2402.02592) · [源码](https://github.com/SalesforceAIResearch/uni2ts) |
| 9 | **A Decoder-Only Foundation Model for Time-Series Forecasting (TimesFM)** — Abhimanyu Das, Weihao Kong, Rajat Sen, Yichen Zhou | ICML 2024 | [10.48550/arXiv.2310.10688](https://doi.org/10.48550/arxiv.2310.10688)¹ | 43 | [正式论文](https://proceedings.mlr.press/v235/das24c.html) · [源码](https://github.com/google-research/timesfm) |
| 10 | **SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters** — Shengsheng Lin, Weiwei Lin, Wentai Wu, Haojun Chen, Junjie Yang | ICML 2024 Oral | [10.48550/arXiv.2405.00946](https://doi.org/10.48550/arxiv.2405.00946)¹ | 9 | [论文](https://arxiv.org/abs/2405.00946) · [源码](https://github.com/lss-1138/SparseTSF) |
| 11 | **TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods** — Xiangfei Qiu, Jilin Hu, Lekui Zhou, Xingjian Wu, Junyang Du, Bin Zhang, Chenjuan Guo, Aoying Zhou, Christian S. Jensen, Zhenli Sheng, Bin Yang | PVLDB 2024 | [10.14778/3665844.3665863](https://doi.org/10.14778/3665844.3665863) | 124 | [论文](https://doi.org/10.14778/3665844.3665863) · [源码](https://github.com/decisionintelligence/TFB) |
| 12 | **Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting** — Jiawen Zhu, Shuhan Liu, Di Weng, Yingcai Wu | ICML 2026 | 未分配 | 0 | [论文](https://arxiv.org/abs/2605.20678) · [源码](https://github.com/andone-07/Dynamic-TMoE) |

¹ 该 DOI 是 DataCite 注册的 arXiv DOI，并非会议论文集 DOI；相应会议论文目前未检索到独立 Crossref DOI。

## 研究内容与复现价值

### 1. TimeFuse

研究样本级自适应模型融合：不是固定选择单一预测器，而是利用时间序列的元特征判断不同基础模型在当前样本上的相对适用性，再动态融合输出。它与 Dynamic-TMoE 的门控思想最接近，但专家是现成预测模型，适合验证“动态专家选择是否优于单一主干”。

**建议复现**：先复现其元特征提取、基础模型池和融合器；再把 Dynamic-TMoE 的记忆路由器替换为 TimeFuse 融合器，统一比较静态平均、soft gate、top-k gate 与动态专家池。

### 2. TimeBridge

针对多变量长时预测中的非平稳性，区分短期局部依赖与长期协整关系：用分块降低局部非平稳干扰，同时保留跨变量的长期稳定联系。它提供了与 MMD 漂移检测不同的非平稳建模视角。

**建议复现**：重点复现 Integrated Attention 与 Cointegrated Attention，并比较其与 Dynamic-TMoE 漂移触发机制能否互补。

### 3. FAN

从频域识别并消除随时间变化的非平稳频率成分，预测后再恢复这些成分。它可作为 RevIN 类归一化方法的频域扩展，计算成本相对低，适合成为所有主干模型的统一插件基线。

**建议复现**：在 Dynamic-TMoE 输入端加入 FAN，与 RevIN、无归一化、MMD 漂移检测四种设置进行交叉消融。

### 4. TimeFilter

将时间片视为节点，针对每个 patch 学习特异的时空图并过滤无效关系，从而避免使用固定或全局共享的变量依赖。其 patch-specific 思路与“不同局部模式路由到不同专家”高度互补。

**建议复现**：复现图构造和 filtration 模块；测试将过滤后的 patch 表征送入 Dynamic-TMoE 是否减少错误路由。

### 5. CycleNet

显式建模时间序列的周期基线，并在去除周期后预测残差，最终恢复周期成分。模型结构简洁，适合检验复杂 MoE 的收益是否只是来自对周期模式的更好拟合。

**建议复现**：作为低成本强基线；额外比较固定周期专家与 Dynamic-TMoE 自动生成专家的差异。

### 6. TimeXer

面向带外生变量的预测任务，通过全局 token 和交叉注意力连接目标序列与外生序列，避免简单拼接造成的信息污染。它可把 Dynamic-TMoE 从常规多变量预测扩展到更真实的 covariate-informed forecasting。

**建议复现**：先复现 Weather/Electricity 等外生预测设置，再研究路由器是否应同时读取目标序列和外生变量。

### 7. HimNet

针对时空节点异质性，通过元参数学习为不同节点生成自适应参数，使模型不再假设所有传感器共享同一动态规律。其核心是“参数按实例/节点适配”，与“专家按分布适配”形成直接对照。

**建议复现**：在交通数据集上比较 meta-parameter、固定专家、动态专家三种容量分配方式。

### 8. Moirai

使用多分布输出头、任意变量数适配和大规模统一训练，构建通用概率预测 Transformer。它代表“单一大模型通过预训练获得泛化”的路线，可与 Dynamic-TMoE 的“动态扩展专门化专家”路线比较。

**建议复现**：以官方预训练权重做零样本/少样本评测，不建议首先从头预训练；统一使用 CRPS、MSE、MAE 和推理成本。

### 9. TimesFM

采用 decoder-only patch Transformer，在大规模真实与合成时间序列上预训练，关注零样本预测和可变预测长度。它是评估 Dynamic-TMoE 是否能超越基础模型迁移能力的重要外部基线。

**建议复现**：直接运行官方 checkpoint；重点测试发生人为分布漂移前后的性能退化和在线适应成本。

### 10. SparseTSF

用极少参数建模长期预测，强调跨周期稀疏采样与轻量线性映射。它能帮助判断复杂动态专家机制的增益是否值得参数量、显存和训练时间开销。

**建议复现**：除 MSE/MAE 外必须报告参数量、训练时长、峰值显存和单样本延迟。

### 11. TFB

构建覆盖多数据集、多预测器和统一评测协议的时间序列预测基准，重点解决数据划分、超参数搜索和指标实现不一致导致的不公平比较。它不是新预测结构，但对可信复现最重要。

**建议复现**：将 Dynamic-TMoE、TimeFuse、FAN、CycleNet、SparseTSF 纳入相同 TFB pipeline；固定数据划分、搜索预算、随机种子和早停规则。

### 12. Dynamic TMoE

利用 MMD 检测分布变化，动态扩张或剪枝异质专家池，并通过时间记忆路由器保持专家选择稳定。它把非平稳预测从“输入归一化/固定模型适配”推进到“模型容量随漂移变化”。

**建议复现**：作为主实验对象，应优先核验 MMD 阈值、专家新增/剪枝条件、路由记忆长度和专家异质性四组核心消融，并保存专家生命周期与路由轨迹。

## 推荐复现顺序

1. **评测底座**：TFB、SparseTSF、CycleNet。
2. **非平稳插件**：FAN、TimeBridge。
3. **动态结构对照**：TimeFilter、HimNet、TimeFuse。
4. **主模型**：Dynamic TMoE。
5. **外部泛化**：TimeXer、Moirai、TimesFM。

若算力有限，最小可行组合是 **TFB + SparseTSF + FAN + TimeFuse + Dynamic TMoE**；这五项已经覆盖公平评测、轻量基线、非平稳处理、动态融合和动态专家池。

## 核验说明

- 12 篇题名、作者、DOI/论文链接和引用量均通过 OpenAlex 精确题名匹配核验；会议归属同时参考会议论文页或作者官方仓库。
- OpenAlex 对若干 2025 新论文曾返回错误的高相似结果；这些错误匹配已排除，没有把错误 DOI 或引用量写入本表。
- 新论文引用量很低属于正常现象，不应被解释为论文质量低；本筛选没有单纯按引用量排序。
- 引用量可通过表中的论文或 DOI 链接复核，但未来会变化。
