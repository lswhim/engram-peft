# Collision-Aware Semantic Engram：论文与完整实验规划

> 版本：v3（2026-08-18）
> 目标：ICLR 2027 方法论文；标准 benchmark、完整规模、可证伪因果对照
> 资源：4 × NVIDIA A100 40GB
> 当前状态：10K 四方法 × 三 seed 已完成；50K scale-development 正在运行；官方八时间步主协议待运行

## 1. 一句话故事

Semantic hash 不只是把相似 n-gram 放到相近地址；它还产生了**具有不同碰撞负载的多个共享候选**。现有
Engram 把所有候选无差别读取，因此有益语义共享与过载碰撞同时进入模型。我们利用地址表自身可离线统计的
collision specificity，优先读取更可靠的 semantic buckets，在不增加可训练参数、不读取测试标签的条件下改善
知识写入后的泛化。

暂定标题：

> **Trust the Address, Not Every Collision: Collision-Aware Semantic Memory for Language Models**

## 2. Motivation：为什么不能只做 Semantic RQ

原始 Engram 的 arithmetic hash 只负责稳定寻址，collision 基本没有语义。Semantic RQ 将冻结 embedding 的
局部几何投射到离散地址，使相关表达有机会复用 memory rows；但它同时带来一个此前不存在的问题：

```text
语义相似 -> 地址部分共享 -> 可能迁移已写知识
高频/多义 bucket -> 大量无关 n-gram 共享 -> memory interference
```

RQ 的 8 个 residual codes 是 8 个并行地址候选，并不天然构成“关系到实体”的可解释语义层级。WikiBigEdit-50K
地址审计已经否定了这种过强假设：Semantic 的中间 prefix overlap 高于 shuffled，但 propagation 和 locality
都提高，不能把 RQ depth 直接解释为任务相关的 coarse-to-fine hierarchy。

因此真正的问题不是“semantic embedding 是否更强”，而是：

> **当 semantic addressing 主动制造共享时，模型应该信任哪些 collision？**

## 3. 方法：Collision-Aware Semantic Engram

### 3.1 动态语义地址

对运行时出现的每个 2/3-gram：

```text
n-gram -> frozen Qwen3-Embedding -> frozen FAISS RQ -> M discrete codes
```

首次出现时执行 encoder/RQ forward，并写入 persistent cache；后续为确定性查表。训练和推理使用相同冻结
encoder/codebook。不能把词典外 n-gram 回退为 arithmetic hash，否则 semantic generalization 在真正的新表达上
会消失。

### 3.2 地址可靠性

只用训练地址表，统计第 `j` 个 codebook 中 bucket `c` 所包含的 distinct n-gram 数：

\[
L_{j,c}=|\{g:q_j(g)=c\}|,\qquad
S_{j,c}=-\log(1+L_{j,c}).
\]

`L` 越大，bucket 越容易混入无关模式；`S` 越高，地址越 specific。该统计：

- 不使用下游测试 query、答案或 evaluation label；
- 不增加可训练参数；
- 可随 RQ 表离线构建并与 memory 一起 offload；
- 对运行时动态编码得到的 code 同样可查询。

### 3.3 参数匹配的 head selection

把原本 flatten 后的一次投影按输入 block 精确分解：

\[
W^V[e_1;\ldots;e_H]=\sum_j W^V_j e_j.
\]

按当前 token 实际访问 bucket 的 `S_{j,c}` 选择 top-k heads：

\[
\mathcal I_t=\operatorname{TopK}_j S_{j,c_{t,j}},\qquad
m_t=\sum_{j\in\mathcal I_t}a_{t,j}W^V_je_{t,j}.
\]

其中 `a` 仍由原始 context gate 决定，specificity 只决定读哪些 heads。为排除“少读 head 导致注入幅值变小”
的混淆，selected gate mass 被重标定为与 dense flatten 相同。正式配置冻结为 `k=4`。

方法的核心不是额外 router，而是把 semantic address 中已经存在、却被 flatten 丢弃的**地址可靠性信号**用于
memory readout。

### 3.4 为什么规模越大越可能显出优势：collision 的 bias–variance 视角

对某个 memory bucket `b`，训练时所有命中它的 n-gram 都会更新同一 value。令 `u_g` 表示 key `g` 对该 value
产生的归一化更新方向，则查询 `q` 从共享中得到的是：

\[
\Delta_b(q) \propto \sum_{g\in B_b} w_g\langle u_q,u_g\rangle.
\]

Arithmetic hash 使 `B_b` 与语义关系近似独立；随着 writes 增多，更多无关更新进入同一 bucket，alignment 的均值
接近全局背景而方差/抵消增大。Semantic RQ 则提高 paraphrase、同 relation 或相近实体表达落入部分共同 heads 的
概率，使其中一部分 collision 成为有益参数共享。但 semantic bucket 仍可能因高频、多义或量化边界而混杂，因此
不是“semantic collision 都好”，而是形成了多个 reliability 不同的候选共享路径。

当前 specificity `-log(1+L)` 是对干扰风险的保守、train-only proxy：它不声称直接测得 gradient alignment，只利用
在其他条件相同时，更大的 distinct-key load 提供更多无关碰撞机会。RQ 的多 head 结构允许查询在不增加参数的情况下
选择低风险路径。这一解释给出三个可证伪预测：

1. Semantic-flat 相对 Arithmetic 的优势应随 cumulative writes / collision pressure 增大；
2. 在相同 access-weighted load 下，Semantic 应优于 code-vector ownership 被置换的 control；
3. 在 Semantic 地址内，selected low-load heads 的 update-gradient agreement / target entropy 应优于 bottom-load heads，
   且该差异能预测 query-level generalization gain。

前两个预测由 scaling 与 load-matched 2×2 主表检验；第三个必须通过 gradient/target-conflict probe 与
bottom-specific intervention 检验。若第三条不成立，论文只能保留“load-aware regularization”结论，不能声称找到了
semantic collision reliability 的机制。

## 4. 已有证据与当前结论

### 4.1 WikiBigEdit-10K 完整三 seed 结果

Qwen3-1.7B-Base、Qwen3-Embedding-4B、RQ `M=8/K=16`，chronological 10K writes。结果为 mean ± sample std：

| Address / readout | Efficacy | Generalization | Locality | Multi-hop |
|---|---:|---:|---:|---:|
| Semantic + collision-aware | **56.241 ± 0.194** | **54.089 ± 0.075** | 44.616 ± 0.044 | 32.514 ± 0.286 |
| Semantic + flatten | 54.478 ± 0.035 | 52.437 ± 0.107 | **44.737 ± 0.102** | **32.784 ± 0.093** |
| Runtime-randomized RQ + collision-aware | 54.170 ± 0.116 | 51.848 ± 0.103 | 44.657 ± 0.333 | 33.663 ± 0.488 |
| Runtime-randomized RQ + flatten | 52.825 ± 0.098 | 51.639 ± 0.108 | 45.289 ± 0.186 | 34.115 ± 0.874 |

配对差值：

- Collision-aware − Semantic flatten：Efficacy `+1.764 ± 0.160`，Generalization `+1.652 ± 0.181`；
  Locality `−0.121 ± 0.078`，Multi-hop `−0.270 ± 0.357`。
- Semantic collision-aware − Shuffled collision-aware：Efficacy `+2.072 ± 0.229`，Generalization
  `+2.241 ± 0.143`。
- Generalization 的 interaction：
  `(Semantic aware − Semantic flatten) − (Shuffled aware − Shuffled flatten)` = **`+1.444 ± 0.311`**。

三个 seed 上 Semantic aware 相对 Semantic flatten 的 generalization 增益分别为 `+1.855/+1.508/+1.595`
pp，方向一致。

### 4.2 WikiBigEdit-50K trajectory：Semantic-flatten 三 seed

同一条 50K chronological trajectory 在 1K/5K/10K/50K 保存 adapter，并在固定历史 cohorts 上完整评测。
Semantic-RQ + flatten 的 mean ± sample std（%）为：

| Writes | Efficacy | Generalization | Locality | Multi-hop |
|---:|---:|---:|---:|---:|
| 1K | 35.254 ± 0.166 | 33.526 ± 0.254 | 33.786 ± 0.134 | 28.250 ± 0.000 |
| 5K | 50.718 ± 0.028 | 49.861 ± 0.081 | 44.858 ± 0.085 | 37.664 ± 0.172 |
| 10K | 55.554 ± 0.174 | 54.402 ± 0.009 | 45.102 ± 0.043 | 38.525 ± 0.187 |
| 50K | **57.235 ± 0.233** | **55.063 ± 0.156** | **47.538 ± 0.151** | **39.537 ± 0.480** |

Arithmetic 三 seeds 已完成。50K Arithmetic 的 Efficacy/Generalization/Locality/Multi-hop 为
`54.420±0.179 / 52.115±0.187 / 47.055±0.030 / 37.867±0.231`。Semantic-flatten − Arithmetic 的三 seed
配对效应为：

| Axis | Delta (pp) | Hierarchical paired case-bootstrap 95% CI |
|---|---:|---:|
| Efficacy | **+2.815** | **[+2.282, +3.358]** |
| Generalization | **+2.948** | **[+2.437, +3.451]** |
| Locality | **+0.483** | **[+0.035, +0.927]** |
| Multi-hop | **+1.670** | **[+0.604, +2.767]** |

四轴在 50K 的 CI 均不跨零。Scaling 并非严格单调：Semantic 的 Efficacy/Generalization 优势在
1K/5K/10K/50K 分别为 `+1.221/+2.310`、`+1.647/+1.673`、`+1.677/+2.064`、
`+2.815/+2.948` pp；但最大规模的效应明显大于 1K，支持 collision pressure 下 semantic sharing 更有价值的
趋势性预测。严格语义几何归因仍需等待 load-matched control，不能只凭 Arithmetic 对比完成。

补充的 Runtime-randomized RQ + flatten 三 seeds 也已完成。其 50K Efficacy/Generalization/Locality/Multi-hop 为
`55.871±0.046 / 53.179±0.090 / 47.189±0.150 / 39.432±0.332`。Semantic-flat − Runtime-randomized-flat 的
配对效应为：Efficacy `+1.364` pp，CI `[+0.903,+1.832]`；Generalization `+1.884` pp，CI
`[+1.420,+2.327]`；Locality `+0.348` pp，CI `[-0.128,+0.794]`；Multi-hop `+0.105` pp，CI
`[-0.957,+1.216]`。这支持 semantic ownership 对写入与改写泛化有贡献，但该随机化会改变 per-head marginal
bucket loads，因此仍是 supporting control，不能替代 strict load-matched 归因。

Runtime-randomized RQ + collision-aware 三 seeds 也已完成，50K 四轴为
`57.278±0.030 / 53.959±0.101 / 46.142±0.168 / 39.913±0.732`。在随机地址内部，
collision-aware − flatten 的逐 case hierarchical bootstrap 效应为：Efficacy `+1.407` pp，CI
`[+1.006,+1.826]`；Generalization `+0.779` pp，CI `[+0.352,+1.212]`；Locality `−1.048` pp，
CI `[−1.497,−0.595]`；Multi-hop `+0.480` pp，CI `[−1.241,+2.351]`。这表明低负载 head 选择本身
能提高写入与改写泛化，但会显著损害 locality；因此论文不能把 routing 的全部收益归因于 semantic geometry，
必须用四路 difference-in-differences 检验 semantic ownership 是否带来额外收益或缓和该 trade-off。

### 4.3 离线表覆盖与动态 OOV 审计

对同一 50K chronological train stream 和固定 2,000-case evaluation manifest，按实际 tokenizer、`max_length=128`
与 prompt 口径重新统计候选 n-gram：

| Source | 2-gram offline coverage | 3-gram offline coverage |
|---|---:|---:|
| 50K training stream | 100.00% | 98.29% |
| Evaluation queries | 90.80% | 76.70% |

因此不能把当前系统描述为“评测时纯离线查表”：evaluation candidate accesses 中约 `9.20%` 的 2-gram 和
`23.30%` 的 3-gram 首次需要 frozen Qwen3-Embedding→RQ，并随后写入 persistent cache。正式实验必须同时报告：

1. offline-hit、dynamic-OOV 与 aggregate 三个切片的任务指标；
2. 冷路径 encoder+RQ latency、首次 cache 构建成本、warm-cache lookup latency；
3. dynamic cache 大小、host-memory 占用和重复 query hit rate；
4. 禁止 OOV 回退 arithmetic，并用 trace 证明训练/评测的 code 生成规则一致。

这一审计不削弱 semantic generalization 的定义，反而使其可检验：若提升只存在于 offline-hit query，方法证明的是
离线词典共享；只有 dynamic-OOV query 也获得提升，才能支持“新表面形式可由语义地址泛化”的更强 claim。

50K 三 seed post-hoc 配对切片已经完成。对每个 query，若任意有效 2/3-gram context
不在冻结 RQ 表中，则标为 `dynamic_any`；否则标为 `offline_all`。Semantic-flatten − Arithmetic（pp）为：

| Axis / address slice | Three-seed delta (pp) | 95% CI | Queries / seed |
|---|---:|---:|---:|
| Generalization / dynamic-any | **+2.971** | **[+2.445,+3.499]** | 1,797 |
| Generalization / offline-all | **+2.750** | **[+1.233,+4.294]** | 203 |
| Locality / dynamic-any | **+0.484** | **[+0.022,+0.915]** | 1,996 |
| Multi-hop / dynamic-any | **+1.564** | **[+0.527,+2.665]** | 110 |
| Efficacy / offline-all | **+2.876** | **[+2.337,+3.429]** | 1,967 |
| Efficacy / dynamic-any | -0.850 | [-2.828,+1.052] | 33 |

最重要的是，`89.85%` 的 generalization queries 属于 dynamic-any，三 seed aggregate 增益约 `+2.97 pp`；因此总体
generalization 增益没有在首次出现的新 n-gram 上消失。Efficacy 的 dynamic-any 只有 33 条，不能稳定外推。

固定边界的 dynamic-OOV access-ratio dose-response 已完成三 seeds。Generalization 的配对增益在
`0% / (0,10%] / (10,25%] / (25,50%] / (50,100%]` 五档分别为
`+2.750 / +3.057 / +3.589 / +1.821 / +2.778` pp；对应 95% CI 均不跨零：
`[+1.233,+4.294] / [+2.023,+4.075] / [+2.818,+4.377] / [+0.931,+2.687] / [+1.010,+4.924]`。
这支持“对不同新地址负担均有泛化收益”，但不支持“收益随 OOV 单调增加”。同时，Locality 在最高 OOV 桶为
`-1.851` pp，95% CI `[-3.108,-0.694]`，提示动态地址过多时
semantic sharing 会增加干扰；这应作为 collision-aware reliability 机制要解决的 failure mode，而不能隐藏在
aggregate 指标里。

### 4.4 能说与不能说的结论

当前能够支持：

1. Semantic 地址优于当前 deterministic runtime-randomized RQ；该控制会改变 marginal bucket loads，因而只是
   初步语义消融，仍需 load-matched shuffle 才能严格归因；
2. collision-aware selection 稳定改善 Semantic Engram 的 efficacy/generalization；
3. interaction 明显大于零，收益不是普通 top-k 稀疏化可以完全解释；
4. locality 和 multi-hop 相对 Semantic flatten 基本保持。

当前还不能声称：

- 已达到 ICLR 完整证据标准；当前只有一个 benchmark 的 10K 结果；
- RQ levels 是可解释的语义层级；地址审计不支持；
- 方法已经带来实际 wall-clock 加速；当前实现重点是统计稀疏，而非 kernel 稀疏；
- 对所有知识编辑、语言迁移或模型尺度均有效。

## 5. 正式实验协议

### 5.1 主实验：WikiBigEdit official lifelong protocol

**时间因果约束：**正式主表的 Qwen3-Embedding-4B → RQ codebook 只允许在第一个时间段
`wiki_big_edit_20240201_20240220`（26,922 edits）上校准一次，之后永久冻结。后续时间段出现的
新 n-gram 必须在线执行 frozen embedding → frozen RQ 并写入持久 cache，不能利用未来 timestep
预先拟合地址几何。使用完整 502,382 条输入拟合的 RQ 只能作为显式标注的 transductive oracle
ablation，不能作为主结果。

开发阶段固定配置：

```text
backbone        = Qwen3-1.7B-Base
address encoder = Qwen3-Embedding-4B
RQ              = M=8, K=16
active heads    = k=4（10K development 后冻结）
writes          = 1K / 5K / 10K / 50K / 100K（scale-development checkpoints）
seeds           = 42 / 123 / 456
```

当前 50K 使用官方原始 edits 转换出的 chronological stream 与固定历史 cohorts，但 evaluator 是自建的完整
target-token accuracy，Locality 评价的是 locality ground-truth accuracy，且旧 loader 遗漏 Personas。因此该实验是
**完整规模的方法开发与 scaling 证据**，不是官方 WikiBigEdit 主表；不能在论文中把两者混称。

最终主表严格复现官方八个真实时间步：每个 timestep 写入该时间段 edits，随后在 Update、Rephrase、Personas、
Mhop、Locality 五轴评测，并对所有过去 timesteps 做 retention/forgetting。评分保持官方的 `Q: ... A:`
teacher-forced target-token accuracy；Locality 使用编辑前后预测保持率，而非答案准确率。官方数据的 Personas 字段
必须保留且只用于评测。训练集、地址统计表和 timestep 边界在实验前冻结。

100K development curve 开始前也必须重新构建匹配前 100K train split 的 RQ 表；不能沿用 50K specificity prior
后宣称标准 100K scaling。所有方法使用相同数据顺序、训练 tokens、optimizer 和 checkpoint。

### 5.2 五个必要主表方法

1. Arithmetic-fixed Engram：原始无语义地址；
2. Semantic-RQ + flatten：最强直接 semantic-hash baseline；
3. RQ-Shuffled-load-matched + flatten：只在完全相同 train-access frequency 的 rows 间置换完整 code vector，
   精确保留 joint-code multiset、各 level histogram 与 access-weighted bucket load；
4. RQ-Shuffled-load-matched + collision-aware：判断 selection 是否只是一般稀疏正则；
5. **Semantic-RQ + collision-aware**：完整方法。

这五个方法形成 `address geometry × readout` 的 2×2 因果矩阵，并额外用 Arithmetic 锚定原始 Engram。
当前正在运行的 `runtime-randomized RQ` 保留为补充的强随机化对照，但不能替代上述 load-matched 主对照：它对
完整 semantic code vector 做确定性随机映射，虽基本保留 full-code equality groups，却改变 per-head marginal loads，
并在有限 code space 中引入极少量额外 joint collisions。所有动态 OOV 必须单独报告；load-matched 离线 rows 与
OOV deterministic randomization 的结果也需分别切片。

### 5.3 外部 benchmark

- CounterFact：rewrite efficacy、paraphrase、neighborhood/locality；
- ZsRE：reliability、paraphrase、locality；
- RippleEdits 或 MQuAKE：需要传播与不应传播；
- PAWS-X 只作为跨语言语义迁移诊断，不作为知识编辑主表替代品。

至少一个外部知识编辑 benchmark 必须复现 Semantic-aware 对 Semantic-flatten 的同方向优势。

### 5.4 领域基线

最终主表不能只有 Engram 内部变体。应加入：

- Frozen Base；
- Continual FT 与 LoRA；
- WikiBigEdit 官方代码中可在相同 backbone/协议上复现的 lifelong KE baseline；
- 一个 retrieval/external-memory baseline。

若公开实现无法支持 Qwen3，优先增加官方支持 backbone 的对应尺度复现实验，而不是自行做不可比的近似版本。

## 6. 消融与机制验证

### 6.1 已冻结的开发消融

`k∈{2,4,8}` 的完整 10K seed-42 sweep：k=4 是 Pareto 点；k=2 虽有相近泛化，但 locality 明显下降；k=8
泛化较弱。因此正式主表不再调 k。

### 6.2 仍必须完成

1. bucket statistic：distinct load、token frequency、uniform/random ranking；
2. selection：top-specific、bottom-specific、random-k、all-head flatten；
3. mass preservation on/off，确认不是注入幅值效应；
4. Semantic/Shuffled/Arithmetic，在完全相同 readout budget 下比较；
5. RQ `M/K` 的小规模训练集 validation sweep，只用于稳健性，不重新选择主配置；
6. embedding encoder 0.6B/4B/8B，报告质量与冷路径成本 Pareto；
7. train-table load 与运行时 dynamic-cache load 分开统计，避免 OOV 口径混乱。

### 6.3 机制链条

论文必须逐步验证：

```text
semantic embedding neighborhood
-> structured partial-code sharing
-> bucket collision load 可预测干扰风险
-> top-specific heads 复用更可靠的已写 rows
-> paraphrase/generalization 增益
-> shuffle / bottom-specific intervention 后增益消失
```

报告 query-level code overlap、selected bucket load、updated-row reuse、collision answer entropy，以及这些量与单 query
增益/失败的关系。地址层面的相关性必须由 intervention 支撑，不能只画 embedding t-SNE。

### 6.4 Held-out target-conflict probe：已完成

为避免在同一训练样本上自证 bucket purity，按 chronological stream 固定切分：前 40K edits 估计每个物理 bucket 的
target-token posterior，后 10K edits 仅作查询。每个 held-out target context 比较 top-specific、bottom-specific 与
random-k 四个 heads 的 posterior surprisal；仅使用离线表完整覆盖的 context，最终包含 9,580 cases、每组
120,012 head-events，并做 5,000 次 paired case-cluster bootstrap。

| Address | Top-specific | Random-k | Bottom-specific | Top−Random | Top−Bottom |
|---|---:|---:|---:|---:|---:|
| Semantic RQ | **8.511** | 8.784 | 9.031 | **−0.273** | **−0.520** |
| Load-matched shuffled | **8.862** | 9.068 | 9.219 | **−0.206** | **−0.358** |

两种地址中 top-specific 都优于 random/bottom，证明低负载选择包含一般 load regularization；但 Semantic 的优势
显著更强。逐 case difference-in-differences：

- `(Semantic Top−Random) − (LoadMatched Top−Random) = −0.0672`，95% CI `[-0.0922,−0.0423]`；
- `(Semantic Top−Bottom) − (LoadMatched Top−Bottom) = −0.1628`，95% CI `[-0.1915,−0.1346]`。

因此当前机制结论可以严格写为：**low-load routing generally reduces interference, while semantic ownership makes
address specificity significantly more predictive of future update compatibility**。这不是梯度余弦的直接测量；论文仍需
补一个小规模 per-example gradient agreement probe，避免把 target posterior compatibility 等同于完整优化动力学。

## 7. 统计与成功门槛

- 主结果 3 seeds，报告 mean ± sample std；
- 同一 query、同一 seed 做 paired difference；
- query-level paired bootstrap 95% CI；
- interaction 是预注册机制指标，而非只比较两个独立均值；
- 不报告最好 seed、最好 checkpoint，不在 test milestone 上重新调超参；
- 同时报告绝对指标、相对 Semantic flatten、相对 Shuffled-aware 和相对 Arithmetic。

论文继续推进的最低门槛：

1. WikiBigEdit 50K/100K generalization 相对 Semantic flatten 至少约 `+1.0` pp，三 seed 同方向；
2. paired bootstrap CI 排除 0；
3. locality 下降不超过 `0.5` pp，或综合 Pareto 显著更优；
4. Semantic interaction 保持明显为正；
5. 至少一个外部 KE benchmark 复现；
6. bottom-specific/random/shuffle intervention 支持碰撞可靠性解释。

10K 已达到效果门槛，但尚未达到 benchmark 覆盖门槛。

## 8. 四卡执行顺序

### Phase A：方法开发与因果矩阵（完成）

- 修复运行时 OOV 必须真实 embedding→RQ 的实现；
- 修复多 seed 实际初始化相同的问题；
- 完成地址审计、mass-preserving control、k sweep；
- 完成 10K 四方法 × 三 seed。

### Phase B：WikiBigEdit-50K scale-development（运行中）

- GPU0/1/2：各固定一个 seed，顺序运行 Semantic flatten、Shuffled flatten、Shuffled aware、Semantic aware；
- GPU3：Arithmetic 三 seed；
- 共 15 个 50K 完整训练，每个评测 1K/5K/10K/50K checkpoint；用于验证规模稳定性，不替代官方主表。

### Phase C：官方八 timestep 主表、100K development 与外部 benchmark

- 从官方八个 JSON timestep 重建包含 Personas 的完整 manifest；
- 使用官方 `Q:/A:` target-token 与 pre/post locality-preservation evaluator；
- 构建严格匹配各冻结 train split 的 RQ 表和 specificity prior；
- 运行官方 lifelong 主表，并补充 100K development curve；
- 并行 CounterFact、ZsRE、Ripple/MQuAKE。

### Phase D：机制、效率和尺度

- random/bottom-specific intervention；
- query-level paired bootstrap 与 collision-load 分桶；
- cold/warm latency、cache hit、throughput、CPU offload 和 memory footprint；
- 至少增加一个 backbone 尺度或官方 WikiBigEdit backbone。

## 9. 预期贡献

1. **问题发现**：semantic memory 的瓶颈不是能否制造共享，而是地址碰撞的可靠性高度不均；
2. **方法贡献**：无需额外可训练 router，直接从冻结 semantic address table 导出 collision-aware sparse readout；
3. **因果贡献**：用 Semantic/Shuffled × aware/flatten 的交互实验分离语义几何、共享容量和一般稀疏化；
4. **实证贡献**：在大规模 lifelong editing、外部 KE benchmark 与干预实验中验证 transfer–interference Pareto；
5. **系统贡献**：保留动态 OOV semantic encoding + persistent cache + offload，并如实报告冷/热路径成本。

## 10. 新颖性边界与 Related Work 审计（2026-08-18）

论文不能笼统声称“首次改造 Engram hash”或“首次处理 hash collision”。截至当前公开工作的边界如下：

- **原始 Engram** 使用 tokenizer compression、multi-head arithmetic hash 与 context-aware gate；它没有语义地址，
  也没有用 bucket occupancy 决定读取哪些 hash heads。
- **Engram-Nine** 用 MPHF 给高频 n-gram 增加 collision-free hot tier，结论是消除碰撞并不稳定改善性能，且指出
  gating credit assignment 可能比索引精度更关键。我们的假设不是“碰撞越少越好”，而是 semantic address
  产生的候选碰撞具有不同可靠性，应保留共享并选择性读取；`bottom-specific` 和 interaction 实验必须证明这一区别。
- **Tokenizer-Agnostic Engram Module**（arXiv:2607.29065）把 token n-gram 改为 byte-equivalent sequence，并用
  polynomial hash 获得跨 tokenizer 的地址等价性。它解决的是 tokenizer portability，不使用 semantic embedding、
  residual quantization、bucket-load reliability 或 lifelong editing。它与本工作的共同点仅是都修改了地址层。
- **Semantic-ID / RQ 工作**主要位于生成式推荐与检索，已经研究 code collision、动态 code 长度及 tail-item
  generalization。因此“RQ 能产生 semantic IDs”本身不是贡献；本工作的 NLP 新意必须来自：RQ code 作为可 offload
  的 Engram memory addresses、地址诱导的参数共享、以及由 train-only collision statistics 控制的 parameter-matched
  readout。
- **HashedNets / hash embeddings**早已把碰撞视为参数共享或压缩机制。因此论文不能把“bucket load”包装成全新概念；
  贡献是它在 semantic conditional memory 中作为无需训练 router 的可靠性信号，并通过完整 2×2 因果实验验证。

当前检索没有发现与“frozen semantic RQ Engram addresses + per-query occupancy-based head selection + mass-preserving
readout + lifelong knowledge-update evaluation”同构的方法，但这只是截至检索日期的公开文献审计，不是绝对无人做过的
证明。投稿前必须再次检索 arXiv/OpenReview，并把上述四个组成部分逐项对比，而不是依赖标题关键词。

### 10.1 可证伪的新颖性门槛

如果最终只观察到 `Semantic-RQ > Arithmetic`，而 collision-aware 相对 Semantic-flatten 的 interaction 不稳定，
那么工作最多是一个 semantic-address engineering follow-up，方法贡献不足以支撑 ICLR 主会。只有同时满足以下条件，
才能保留当前论文定位：

1. collision-aware 的收益在 50K/官方多时间步与至少一个外部 benchmark 上复现；
2. shuffled、random-k、bottom-specific 排除普通稀疏化和幅值缩放解释；
3. bucket load 能在 query level 预测 transfer/interference，并由干预实验支持；
4. 动态 OOV 冷路径、cache 热路径和 offload 成本可测量，证明方法没有悄悄放弃 Engram 的系统属性。

## 11. Reviewer 视角的主要风险

- **“只是 heuristic top-k”**：必须用 interaction、bottom/random intervention 和 load–error 分析证明 specificity 是地址
  可靠性，而非任意剪枝。
- **“只在一个 benchmark 有效”**：CounterFact/ZsRE/Ripple 至少一个必须复现。
- **“baseline 不够”**：加入外部 lifelong KE 与 retrieval baseline；内部 ablation 不能替代领域 SOTA。
- **“semantic encoder 成本太高”**：报告 cold/warm latency和 cache hit，不把首次 4B forward 隐藏为 O(1)。
- **“训练表统计泄漏”**：统计严格只来自 train address table；评测 query 只查冻结 statistic。
- **“RQ hierarchy 叙事过度”**：不再声称有序语义层级，方法只依赖可测量 collision load。

目前最合理的论文核心已经从失败的 CREDIT/utility-gate 假设收敛为：

> **Semantic hashing makes sharing possible; collision-aware addressing makes that sharing trustworthy.**
