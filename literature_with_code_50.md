# Dynamic-TMoE 近三年相关文献（50 篇，仅含公开源码）

检索截止：2026-07-18。这里将“近三年”定义为 **2024–2026 三个自然年度**。范围集中于时间序列预测、非平稳/分布漂移、动态专家/模型融合、时序基础模型及直接相关的鲁棒性与基准研究。

排序规则：**CCF-A / 公认顶会与顶刊**（ICML、NeurIPS、KDD、IJCAI、AAAI、PVLDB）优先，随后是 **ICLR**。同级内按年份从新到旧、与 Dynamic-TMoE 的相关性排序。所有条目均已定位到公开代码仓库；匿名审稿仓库会特别标注。

## 第一档：CCF-A / 公认顶会与顶刊

| # | 年份 | 论文 | Venue | 相关方向 | 源码 |
|---:|---:|---|---|---|---|
| 1 | 2026 | Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting | ICML 2026 | 动态 MoE、漂移检测 | [官方](https://github.com/andone-07/Dynamic-TMoE) |
| 2 | 2025 | Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift (TFPS) | NeurIPS 2025 | **专家路由、patch 级漂移** | [官方](https://github.com/syrGitHub/TFPS) |
| 3 | 2025 | How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning | NeurIPS 2025 | 时空分布偏移 | [官方](https://github.com/Jimmy-7664/ST-SSDL) |
| 4 | 2025 | xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories | NeurIPS 2025 | 记忆与混合机制 | [官方](https://github.com/mauricekraus/xLSTM-Mixer) |
| 5 | 2025 | TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning | NeurIPS 2025 | 零样本基础模型 | [官方](https://github.com/NX-AI/tirex) |
| 6 | 2025 | TransDF: Time-Series Forecasting Needs Transformed Label Alignment | NeurIPS 2025 | 非平稳标签对齐 | [匿名代码](https://anonymous.4open.science/r/TransDF-88CF) |
| 7 | 2025 | OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain | NeurIPS 2025 | 变换域预测 | [匿名代码](https://anonymous.4open.science/r/OLinear) |
| 8 | 2025 | Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction | NeurIPS 2025 | **非平稳跳变动力学** | [官方](https://github.com/DSL-Lab/neural-MJD) |
| 9 | 2025 | Selective Learning for Deep Time Series Forecasting | NeurIPS 2025 | 动态样本选择 | [官方](https://github.com/JonasBrusokas/Time-Energy-Model) |
| 10 | 2025 | Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting (TimeFuse) | ICML 2025 | **动态模型融合/专家集成** | [官方](https://github.com/ZhiningLiu1998/TimeFuse) |
| 11 | 2025 | TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting | ICML 2025 | patch 特异时空过滤 | [官方](https://github.com/TROUBADOUR000/TimeFilter) |
| 12 | 2025 | K²VAE: A Koopman-Kalman Enhanced Variational AutoEncoder for Probabilistic Time Series Forecasting | ICML 2025 Spotlight | 非平稳动力学、概率预测 | [官方](https://github.com/decisionintelligence/K2VAE) |
| 13 | 2025 | CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise Spatial Correlations | ICML 2025 | chunk 级变量关系 | [官方](https://github.com/CSTCloudOps/CMoS) |
| 14 | 2025 | LightGTS: A Lightweight General Time Series Forecasting Model | ICML 2025 | 轻量通用预测 | [官方](https://github.com/decisionintelligence/LightGTS) |
| 15 | 2025 | Conditional Diffusion Model with Nonlinear Data Transformation for Time Series Forecasting | ICML 2025 | 非线性变换、概率预测 | [官方](https://github.com/quest-lab-iisc/CNDiff) |
| 16 | 2025 | Retrieval Augmented Time Series Forecasting | ICML 2025 | 检索增强、相似模式记忆 | [官方](https://github.com/kutaytire/Retrieval-Augmented-Time-Series-Forecasting) |
| 17 | 2025 | Non-stationary Diffusion for Probabilistic Time Series Forecasting | ICML 2025 | **非平稳扩散预测** | [官方](https://github.com/wwy155/NsDiff) |
| 18 | 2025 | WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting | ICML 2025 | **动态门控** | [匿名代码](https://anonymous.4open.science/r/ARMA-attention-3437) |
| 19 | 2025 | Patch-wise Structural Loss for Time Series Forecasting | ICML 2025 | patch 结构损失 | [官方](https://github.com/Dilfiraa/PS_Loss) |
| 20 | 2025 | TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting | ICML 2025 | **非平稳变量依赖** | [官方](https://github.com/Hank0626/TimeBridge) |
| 21 | 2025 | Winner-takes-all for Multivariate Probabilistic Time Series Forecasting | ICML 2025 | **多专家/WTA 分配** | [官方](https://github.com/Victorletzelter/timeMCL) |
| 22 | 2025 | Frequency Enhanced Transformer for Multivariate Time Series Forecasting (FreEformer) | IJCAI 2025 | 频域增强 | [匿名代码](https://anonymous.4open.science/r/FreEformer) |
| 23 | 2025 | Conditional Information Bottleneck-Based Multivariate Time Series Forecasting | IJCAI 2025 | 变量依赖压缩 | [官方](https://github.com/Xinhui-Lee/CIB-MTSF) |
| 24 | 2025 | A Dynamic Stiefel Graph Neural Network for Efficient Spatio-Temporal Time Series Forecasting | IJCAI 2025 | 动态图结构 | [官方](https://github.com/komorebi424/DST-SGNN) |
| 25 | 2025 | Non-collective Calibrating Strategy for Time Series Forecasting | IJCAI 2025 | 预测校准 | [官方](https://github.com/hanyuki23/SoP) |
| 26 | 2025 | CASA: CNN Autoencoder-based Score Attention for Efficient Multivariate Long-term Time-series Forecasting | IJCAI 2025 | 注意力评分、长预测 | [官方](https://github.com/lmh9507/CASA) |
| 27 | 2025 | CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables | KDD 2025 | 外生变量、互相关 | [官方](https://github.com/mumiao2000/CrossLinear) |
| 28 | 2025 | Performative Time-Series Forecasting | KDD 2025 | 分布受预测行为影响 | [官方](https://github.com/AdityaLab/pets) |
| 29 | 2025 | SDE: A Simplified and Disentangled Dependency Encoding Framework for State Space Models in Time Series Forecasting | KDD 2025 | 解耦依赖、状态空间 | [官方](https://github.com/YukinoAsuna/SAMBA) |
| 30 | 2025 | Beyond Fixed Variables: Expanding-variate Time Series Forecasting via Flat Scheme and Spatio-temporal Focal Learning | KDD 2025 | **动态变量集合** | [官方](https://github.com/mb-Ma/STEV) |
| 31 | 2025 | Merlin: Multi-View Representation Learning for Robust Multivariate Time Series Forecasting with Unfixed Missing Rates | KDD 2025 | 鲁棒预测、缺失率漂移 | [官方](https://github.com/ChengqingYu/Merlin) |
| 32 | 2025 | TimeCapsule: Solving the Jigsaw Puzzle of Long-Term Time Series Forecasting with Compressed Predictive Representations | KDD 2025 | 压缩预测表征 | [官方](https://github.com/Luoauoa/TimeCapsule) |
| 33 | 2024 | Retrieval-Augmented Diffusion Models for Time Series Forecasting | NeurIPS 2024 | 检索增强扩散 | [官方](https://github.com/AdityaLab/FOIL) |
| 34 | 2024 | Frequency Adaptive Normalization for Non-stationary Time Series Forecasting (FAN) | NeurIPS 2024 | **非平稳频率自适应归一化** | [官方](https://github.com/wayne155/FAN) |
| 35 | 2024 | Rethinking the Power of Timestamps for Robust Time Series Forecasting: A Global-Local Fusion Perspective | NeurIPS 2024 | 全局/局部鲁棒融合 | [官方](https://github.com/ForestsKing/GLAFF) |
| 36 | 2024 | AutoTimes: Autoregressive Time Series Forecasters via Large Language Models | NeurIPS 2024 | 大模型自回归预测 | [官方](https://github.com/thuml/AutoTimes) |
| 37 | 2024 | Are Language Models Actually Useful for Time Series Forecasting? | NeurIPS 2024 | LLM 基线辨析 | [官方](https://github.com/BennyTMT/LLMsForTimeSeries) |
| 38 | 2024 | SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion | NeurIPS 2024 | 变量核心融合 | [官方](https://github.com/Secilia-Cxy/SOFTS) |
| 39 | 2024 | CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns | NeurIPS 2024 Spotlight | 周期模式建模 | [官方](https://github.com/ACAT-SCUT/CycleNet) |
| 40 | 2024 | CondTSF: One-line Plugin of Dataset Condensation for Time Series Forecasting | NeurIPS 2024 | 数据凝缩 | [官方](https://github.com/RafaDD/CondTSF) |
| 41 | 2024 | Scaling Law for Time Series Forecasting | NeurIPS 2024 | 模型/数据规模规律 | [官方](https://github.com/JingzheShi/ScalingLawForTimeSeriesForecasting) |
| 42 | 2024 | TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables | NeurIPS 2024 | 外生变量预测 | [官方](https://github.com/thuml/TimeXer) |
| 43 | 2024 | Are Self-Attentions Effective for Time Series Forecasting? | NeurIPS 2024 | 自注意力有效性 | [官方](https://github.com/dongbeank/CATS) |
| 44 | 2024 | SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters | ICML 2024 Oral | 极轻量稀疏预测 | [官方](https://github.com/lss-1138/SparseTSF) |
| 45 | 2024 | CATS: Enhancing Multivariate Time Series Forecasting by Constructing Auxiliary Time Series as Exogenous Variables | ICML 2024 | 辅助序列、外生变量 | [官方](https://github.com/LJC-FVNR/CATS) |
| 46 | 2024 | Unified Training of Universal Time Series Forecasting Transformers (Moirai) | ICML 2024 | 通用时序基础模型 | [官方](https://github.com/SalesforceAIResearch/uni2ts) |
| 47 | 2024 | A Decoder-Only Foundation Model for Time-Series Forecasting (TimesFM) | ICML 2024 | 零样本时序基础模型 | [官方](https://github.com/google-research/timesfm) |
| 48 | 2024 | Heterogeneity-Informed Meta-Parameter Learning for Spatiotemporal Time Series Forecasting | KDD 2024 | **异质性、自适应参数** | [官方](https://github.com/XDZhelheim/HimNet) |
| 49 | 2024 | UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction | KDD 2024 | 通用时空预测 | [官方](https://github.com/tsinghua-fib-lab/UniST) |
| 50 | 2024 | TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods | PVLDB 2024 | 公平评测与复现基准 | [官方](https://github.com/decisionintelligence/TFB) |

## 最值得优先精读的 12 篇

按与 Dynamic-TMoE 的直接关联排序：TFPS、TimeFuse、WAVE、TimeBridge、FAN、NsDiff、Neural MJD、Winner-takes-all、Performative TS Forecasting、HimNet、Moirai、TFB。

## 核验边界

- 表中“公开源码”表示仓库可访问且与题名对应，不代表无需修补即可完整复现实验。
- 4 个匿名审稿仓库可能在会后迁移或失效，建议尽快镜像并记录 commit。
- 2026 年目前只列入已确认的 ICML 2026 Dynamic-TMoE；没有用尚未正式确认 venue 的预印本凑数。
- 会议与代码映射主要由各会议论文页、作者仓库及公开的 [Awesome Time Series Papers](https://github.com/TSCenter/awesome-time-series-papers) 交叉核验。
