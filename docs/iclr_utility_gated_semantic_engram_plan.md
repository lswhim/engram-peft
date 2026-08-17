# CREDIT-Engram：具有反事实信用分配的语义条件记忆

> 版本：v2（2026-08-18，WikiBigEdit-50K 地址审计后修订）
> 目标：ICLR 2027 级完整研究，而非 smoke test  
> 资源：4 × NVIDIA A100 40GB  
> 当前判断：单独把 Arithmetic Hash 替换为 Semantic RQ 不足以构成稳定贡献；核心问题是语义共享发生后，Engram gate 无法判断这次共享是帮助还是干扰。

## 1. 从当前结果出发，而不是从想象出发

现有完整结果给出两个同时存在的事实：

1. WikiBigEdit@50K 中 Semantic-RQ 相比 Arithmetic 在 efficacy/generalization 上约有 3 pp 增益，说明 RQ 型共享结构对大规模连续写入有价值；
2. 但 Semantic-RQ 与 RQ-Shuffled 几乎相同，说明当前模型并未稳定利用 code 的语义排列，收益主要来自共享/碰撞结构，而非语义几何本身。

ParaRel 上，Qwen3-Embedding-4B、M=8、K=16 的 Semantic-RQ 明显优于 Shuffled，但仍未稳定胜过 Arithmetic。这进一步说明语义地址不是完全无效，而是其收益高度依赖任务和 memory 使用方式。

Engram-Nine 的独立观察与此一致：单纯提高 lookup precision 并不保证效果，gate 可能长期偏好实际损失更高的 route。原始 Engram 也指出早层 memory 有利于提前完成局部模式重构，但早层上下文不足会降低 gating precision。

因此，论文不能继续讲：

> semantic embedding 更强 → hash 更语义 → Engram 自然更好。

真正的问题是：

> Semantic RQ 提供了不同粒度的共享候选，但现有 Engram 把所有 RQ head 扁平拼接，并只通过最终语言模型损失间接训练 gate。模型既不知道 RQ level 的粒度，也没有直接信号判断一次 memory 注入对当前 token 是有益还是有害。

### 1.1 WikiBigEdit-50K 地址审计：层级假设被否定

审计覆盖与现有 WikiBigEdit@50K 主实验严格匹配的 50,000 个 chronological edits，包含 52,357 条
propagation query、50,000 条 locality query，共 149,112 个唯一文本。所有 OOV n-gram 都使用冻结的
Qwen3-Embedding-4B 与同一 M=8/K=16 RQ 在线编码；Semantic 与 runtime-shuffled 逐 query 配对。

关键结果：

| Order / terminal suffix | Semantic propagation | Semantic locality | Shuffled propagation | Shuffled locality |
|---|---:|---:|---:|---:|
| 2-gram mean prefix depth | 3.533 | 3.547 | 3.120 | 3.110 |
| 3-gram mean prefix depth | 3.205 | 2.236 | 2.872 | 1.793 |

- 2-gram 的 semantic prefix 完全不能区分 propagation 与 locality；
- 3-gram 有任务区分，但 Semantic 相对 Shuffled 对 propagation 增加约 0.33 层，对 locality 增加约
  0.44 层，false sharing 同样甚至更强；
- `anywhere-in-prompt` overlap 被公共 n-gram 饱和，不可作为机制证据；
- terminal bucket 的中间层 relation purity 多数低于 Shuffled，说明 FAISS RQ residual level 是重构层级，
  不是可直接解释为“粗关系语义→细实体语义”的层级；
- 完整 joint-code 相同率在 Semantic/Shuffled 中必然一致，因为 shuffled control 保留 joint-code identity。

因此，v1 的 ordered-depth 方案正式否决。后续方法不得声称 RQ level 天然具有可用的语义抽象顺序。

## 2. 论文核心假设

### 2.1 一句话 motivation

Semantic addressing determines **who may share a memory**, but a useful conditional memory must additionally learn **how much to share and whether the shared memory causally helps the current prediction**.

### 2.2 修订后的可证伪假设

1. Semantic RQ 的不同 heads 提供不同的共享候选，但不存在可靠的有序 depth；
2. 当前 flatten gate 只能对所有 heads 混合后的 value 做统一调制，无法区分有益与有害 collision；
3. 将读写分解到逐 head route，并保持相同投影参数量，可以控制结构化共享造成的 transfer/interference；
4. 用等激活预算的反事实 route pair 产生 utility preference，可以改善 router 与真实 token loss 的对齐；
5. 改善 credit assignment 后，Semantic-RQ 相对 RQ-Shuffled 的差距应扩大；否则 semantic ordering 不能列为核心贡献。

## 3. 方法：CREDIT-Engram

## 3. 方法：CREDIT-Engram

暂定全称：**Counterfactual Residual-utility Estimation for Dynamic, Interference-aware Transfer**。

方法由一个统一原则导出：semantic memory 的 collision 不是先验的好或坏，必须在读和写时按其对当前预测的
反事实效用分配信用。RQ heads 被视为并行候选 route，而不是虚构成语义深度。

### 3.1 多头语义地址

对每个 suffix n-gram `g_t`：

```text
Qwen3-Embedding(g_t) -> frozen RQ -> (c_1, ..., c_L)
```

首次遇到 n-gram 时执行 embedding 和 RQ 编码，随后把 codes 与必要的量化统计写入 persistent cache。热路径仍为确定性查表。训练和推理使用同一冻结 encoder/codebook。

### 3.2 参数匹配的逐 head value 分解

当前实现：

```text
[2/3-gram × L levels × d_head] -> flatten -> one ContextAwareGate
```

新实现保留 16 个 head（2/3-gram × 8 RQ heads）的轴：

\[
e_{t,j}=M_j[c_{t,j}], \qquad
v_{t,j}=W^V_j e_{t,j}.
\]

这里 `W^V_j` 不是新增的大矩阵，而是原始 `W^V` 按输入维度切出的 block：

\[
W^V=[W^V_1|\cdots|W^V_H],\qquad
W^V[e_1;\cdots;e_H]=\sum_jW^V_je_j.
\]

因此逐 head value decomposition 与 flatten baseline 严格参数匹配；变化只在于允许每个候选 route 独立获得
read/write credit。

### 3.3 Head-factorized read router 与 null route

每个 head 根据 hidden state 与该 head 的 memory value 预测效用分数：

\[
r_{t,j}=g(h_t,e_{t,j},j).
\]

在固定激活预算 `k` 下选择 top-k heads，并与显式 null route 比较：

\[
m_t=\sum_{j\in S_t}a_{t,j}v_{t,j},\qquad |S_t|=k,
\]

若 null route 分数最高，则本 token 不读取 memory。Semantic、Shuffled、Arithmetic 都使用同一 router 和相同
top-k 预算，防止用更多激活 memory 换取效果。

### 3.4 等计算量反事实 route-pair credit

普通 LM loss 只评价实际执行的一条 route。训练时对部分样本采样两个具有相同 head 数的 route：

```text
route A: S_A, |S_A|=k
route B: S_B, |S_B|=k
```

两个 route 通过 batch 内复制在一次 batched 调用中执行。由监督 token NLL 得到停止梯度的 route preference：

\[
\Delta u_t=\operatorname{sg}[\ell(y_t;S_B)-\ell(y_t;S_A)].
\]

router 用 Bradley--Terry pairwise loss 预测哪条 route 更好：

\[
\mathcal L_{credit}=-\log\sigma\left(
-\operatorname{sign}(\Delta u_t)
[R(S_A)-R(S_B)]/\tau
\right).
\]

总目标：

\[
\mathcal L=\mathcal L_{LM}+\lambda_c\mathcal L_{credit}
+\lambda_0\mathcal L_{null}.
\]

另以 memory-on vs null 的配对校准 null route。正式 sweep 比较 25%、50%、100% paired examples；不做逐 head
16 次 forward。推理只执行 top-k route，不需要 utility oracle。

### 3.5 Utility-gated sparse write

同一个 route mask 同时控制 gradient 写入哪些 memory rows：未选择或预测为负 utility 的 head 不更新。这样正向
共享可以跨 paraphrase 累积，而冲突 collision 不会无条件污染所有 16 个 rows。Read-only gating 与 read+write
gating 必须作为独立消融，验证收益是否来自更好的写入隔离。

### 3.6 正样本与负样本

- 正 utility：当前事实 canonical supervision；
- transfer supervision 不使用 benchmark 测试 paraphrase，避免泄漏；
- 负 utility：batch 内其他事实的 prompt、原数据提供的 locality prompt，或从训练 split 构造的 unrelated query；
- 多个事实共享地址但答案冲突时，作为 hard negative；
- 所有样本来源和构造规则在训练前冻结。

## 4. 为什么它可能有效，以及为什么不是普通 A+B

Semantic RQ 与 utility routing 解决同一个 transfer-interference 问题的两个必要部分：

```text
Semantic RQ          -> 产生多个结构化共享候选
Factorized read/write-> 允许不同候选获得不同信用
Counterfactual pairs -> 用真实 loss 判断 route 是否有益
```

只有 semantic hash：产生共享，但无法避免 false sharing。  
只有普通 gate：没有语义结构，无法把新表述路由到已写 memory。  
只有逐 head attention：仍然依赖单路线 LM loss，不能解决 Engram-Nine 指出的 credit mismatch。

方法贡献不是“多加一个 gate”，而是第一次把 conditional memory 的 routing decision 用其因果边际 utility 显式监督。

## 5. 完整实验协议

### 5.1 主任务：WikiBigEdit lifelong scaling

使用官方 chronological/timestep 划分，不只抽一个 10K pilot。

```text
writes = 1K / 5K / 10K / 50K / 100K
seeds  = 3
backbone main = Qwen3-1.7B-Base
address encoder = Qwen3-Embedding-4B
RQ = M=8, K=16（由现有 sweep 选定，不再用测试集调参）
```

主指标必须覆盖 WikiBigEdit 定义的 Update、Rephrase、Locality、Personas、Multi-hop；另外报告历史 edit retention 和时间步间 forgetting。现有只覆盖 efficacy/generalization/locality 的自建表不足以作为最终 WikiBigEdit 主表。

### 5.2 标准知识编辑外部有效性

- CounterFact：efficacy、paraphrase、neighborhood/locality；
- ZsRE：reliability、paraphrase、locality；
- RippleEdits 或修订后的 multi-hop benchmark：传播与不应传播；
- WikiBigEdit 是唯一主线，其他 benchmark 用于证明不是只对一个数据集特化。

### 5.3 方法主表

领域基线：

- Frozen Base；
- Continual FT / LoRA；
- 至少两个可在目标 backbone 上稳定复现的 lifelong KE 方法；优先采用 WikiBigEdit 官方实现中的方法和设置；
- retrieval/external-memory baseline，因为 WikiBigEdit 原论文发现 retrieval 类方法在大规模编辑时可能更强。

Engram 因果基线：

- Arithmetic-fixed Engram；
- RQ-Shuffled Engram；
- Semantic-RQ + 原始 flatten gate；
- Semantic-RQ + head-factorized router，但无 credit loss；
- **CREDIT-Engram**。

公平约束：相同 backbone、target layers、memory rows、embedding dimension、训练 tokens、optimizer search budget。除系统表外，所有 Engram 变体保持相同 trainable parameter budget；若新增 gate 参数，给 baseline 增加同规模 projection 的参数匹配对照。

### 5.4 核心消融

1. 固定激活 heads `k in {1,2,4,8,16}`；
2. flatten gate、head-factorized soft gate、top-k router；
3. 无 credit、memory/null credit、equal-budget route-pair credit；
4. Semantic vs Shuffled vs Arithmetic；
5. 去掉 no-memory route；
6. 去掉 locality/hard-negative credit；
7. paired-example 比例和 `lambda_credit`；
8. read-only routing vs read+write routing；
9. inference 时强制 route/head reset，验证 learned routing 的因果作用。

超参只在 WikiBigEdit validation timestep 上选择一次，其他 benchmark 直接迁移。

### 5.5 机制分析

必须证明下面的链条，而不是只画 gate heatmap：

```text
语义相似
-> 至少部分 RQ heads 复用 source-updated rows
-> 这些 heads 的反事实 route utility 为正
-> router score 与 utility preference 对齐
-> harmful collision 被 null/read-write mask 拒绝
-> paraphrase / ripple gain
```

报告：

- 每个 head 的 embedding cosine、relation/entity purity、collision load 与 utility；
- route-pair counterfactual utility 分布；
- gate score 与 utility 的 AUROC、ECE、Spearman；
- paraphrase、conflict、locality、head/tail entity 的 active heads/null rate；
- reset shared rows、shuffle codes、强制 route 后的性能变化；
- Semantic 相对 Shuffled 的收益按 code overlap 和 utility 分桶。

### 5.6 效率与系统属性

- 首次 semantic encode + RQ 的 cold latency；
- persistent cache 的 warm lookup latency和 hit rate；
- Base、Arithmetic、Semantic-RQ、CREDIT-Engram 的 train/inference tokens/s；
- counterfactual paired training 的额外 FLOPs 和 wall time；
- GPU table 与 pinned CPU/offloaded table；
- memory size、optimizer state、实际 touched rows；
- top-k routing 带来的平均有效 heads，但不得把当前未实现的稀疏 gather 宣称为实际加速。

## 6. 统计规范

- 主结果 3 seeds，报告 mean ± std；
- query-level paired bootstrap 95% CI；
- 多个 WikiBigEdit milestone 使用同一 chronological stream 和固定 evaluation cohorts；
- 所有方法使用相同数据顺序；
- 预注册 primary metric：WikiBigEdit Rephrase 与 Retention 的 harmonic mean，同时约束 Locality；
- 不以最好 seed、最好 checkpoint 或测试集调参结果作为主表；
- 同时报告绝对值和相对 Arithmetic、Shuffled、head-factorized-no-credit 的差值。

## 7. 成功门槛

论文的关键门槛不是 CREDIT 比 Base 高，而是：

1. CREDIT-Engram 在 WikiBigEdit 50K/100K 的主指标上稳定优于 Semantic-RQ flatten；
2. 稳定优于 head-factorized-no-credit，证明收益来自 credit assignment，而不只是改变融合方式；
3. Semantic-CREDIT 优于 Shuffled-CREDIT，证明 semantic ordering 终于被模型利用；
4. Locality 不以不可接受的幅度下降，且综合 Pareto 优于 LoRA/FT；
5. 至少一个外部 benchmark 复现相同方向；
6. gate–utility 对齐指标和 intervention 支持因果解释。

建议预注册最小有效差异：WikiBigEdit 主指标相对 strongest matched Engram baseline 至少 +1.5 pp，三个 seed 方向一致，paired CI 排除 0。若只有 +0.2 pp 或只在单个 milestone 成立，不足以支撑 ICLR 方法论文。

### 7.1 冻结的 10K development gate

在 CREDIT 结果产生前，已有相同 Qwen3-1.7B、Qwen3-Embedding-4B、RQ M=8/K=16、seed 42、10K
chronological milestone 的旧版 flatten 结果为：overall efficacy 55.47%、generalization 54.40%、locality
44.89%；matched shuffled 为 54.65%、52.92%、44.90%。因此开发阶段不按结果临时改门槛：

- CREDIT 的 overall generalization 至少达到 **55.90%**（相对 strongest matched flatten +1.5 pp）；
- overall locality 不得低于 **44.39%**（相对 flatten 最多下降 0.5 pp）；
- 必须同时优于新的 head-factorized-no-credit，排除收益仅来自投影重参数化；
- Semantic-CREDIT 必须优于 Shuffled-CREDIT，排除 credit router 与 semantic code 无关；
- 单 seed 10K 只用于冻结方法/超参。正式 claim 仍需 50K/100K、3 seeds 和 paired bootstrap CI。

该 development gate 使用当前自建的 teacher-forced complete-target-token accuracy，只用于方法开发；不能替代
WikiBigEdit 官方完整主表中的 Update、Rephrase、Locality、Personas、Multi-hop 和 retention。

## 8. 四张 A100 的执行计划

### Phase A：地址与共享审计（已完成 matched-50K；500K 运行中）

matched-50K 已否定 ordered-depth；保留完整 prefix/head overlap、purity、collision 作为 diagnostic。500K 用于验证
该负结论是否随 benchmark 规模稳定，不再作为 ordered architecture 的前置条件。

### Phase B：完整 10K 方法开发（约 4--8 GPU-days）

四卡并行：flatten、head-factorized-no-credit、CREDIT、Shuffled-CREDIT。每个 run 都完整训练到 10K，并评官方
全部指标；随后完成 `k`、null、read/write 与 route-credit 消融。

### Phase C：50K/100K 主表（约 18--30 GPU-days）

冻结所有超参后，主要 Engram 方法三 seed。四卡流水并行。昂贵的领域基线按官方建议规模运行并完整记录失败/不适用项。

### Phase D：外部 benchmark、模型尺度与机制（约 12--20 GPU-days）

CounterFact、ZsRE、Ripple；Qwen3-0.6B/4B 尺度；intervention、效率和 offload。

保守总预算约 35--60 A100-40G GPU-days，即四卡连续约 9--15 天。实际预算必须在 Phase B 记录每个 run 的 wall time 后更新，不能用估算冒充测量。

## 9. 投稿故事

### 标题方向

**Which Collisions Should Be Shared? Counterfactual Read-Write Routing for Conditional Memory**

### 摘要核心

现有 conditional memory 使用确定性地址检索静态模式，但地址 collision 与 memory utility 之间没有显式信用
分配。我们首先在 50K lifelong edits、超过十万条配对 queries 上发现，semantic addressing 虽带来局部 prefix
共享，却不能稳定区分有益 propagation 与有害 locality，且现有 flatten gate 无法给不同 heads 分配信用。
CREDIT-Engram 将原始投影严格分解成参数匹配的逐 head read/write routes，并通过等激活预算的反事实 route
pairs 监督真实 utility。在标准 lifelong knowledge editing、知识编辑与机制干预中，它应实现更好的
transfer-retention-locality Pareto，同时保留冻结 backbone、可缓存地址与可 offload memory。

### 预期贡献

1. 诊断贡献：证明 conditional memory 中 address quality 与实际 route utility 的脱节；
2. 方法贡献：parameter-matched head-factorized memory 与 counterfactual utility-supervised read/write routing；
3. 实证贡献：标准 WikiBigEdit 大规模连续写入、标准 KE 外部验证和完整因果干预；
4. 系统贡献：报告冷/热 semantic addressing 和 counterfactual training 的真实成本，而不把 memoized lookup误写成无条件 O(1)。

## 10. 风险与止损

- **RQ prefix 没有粗到细语义**：matched-50K 已确认；v2 已改成多个并行 heads，不再声称 hierarchy。
- **credit loss 只提高训练集 efficacy**：检查 hard negatives、locality utility 与 gate calibration；若外部改写无增益，方法不成立。
- **Semantic-CREDIT 仍等于 Shuffled-CREDIT**：说明当前 n-gram embedding 几何不适合 edit utility；论文应转为一般 utility-gated Engram，而不能把 semantic hash列为核心贡献。
- **两倍训练代价过高**：降低 paired-example 比例并缓存 utility labels，但主表必须如实报告训练成本。
- **领域 KE baseline 不兼容 Qwen3**：优先复用 WikiBigEdit 官方支持的 backbone/方法；不得用无法运行的名字装饰 baseline 表。

这份计划的目标不是承诺一定得到正结果，而是让每个关键 claim 都对应标准 benchmark、严格对照和可证伪证据。只有通过成功门槛后，才把 CREDIT-Engram 定为最终投稿方法。
