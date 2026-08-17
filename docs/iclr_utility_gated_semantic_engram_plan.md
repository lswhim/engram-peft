# CREDIT-Engram：具有反事实信用分配的层级语义条件记忆

> 版本：v1（2026-08-18）  
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

## 2. 论文核心假设

### 2.1 一句话 motivation

Semantic addressing determines **who may share a memory**, but a useful conditional memory must additionally learn **how much to share and whether the shared memory causally helps the current prediction**.

### 2.2 可证伪假设

1. RQ code prefix 对语义邻域存在统计上的由粗到细结构；
2. 对 paraphrase、别名和相关事实，较浅层共享具有更高边际效用；
3. 对冲突事实、多义表达和 locality query，更深层 residual 或关闭 memory 更有利；
4. 用反事实 memory-on/off 或 depth-pair loss 构造的 utility label，可以改善 gate 与真实 memory utility 的对齐；
5. 改善 credit assignment 后，Semantic-RQ 相对 RQ-Shuffled 的差距应扩大，而不是只共同优于 Arithmetic。

若第 1 条不成立，RQ hierarchy 不能作为论文机制；若第 4/5 条不成立，方法主张失败，不能用单点 accuracy 掩盖。

## 3. 方法：CREDIT-Engram

暂定全称：**Counterfactual Residual-utility Estimation for Dynamic, Interference-aware Transfer**。

方法由一个统一原则导出，而不是任意模块相加：使用 RQ 层级表达共享粒度，并用 memory 的因果边际效用为层级 gate 分配训练信用。

### 3.1 层级语义地址

对每个 suffix n-gram `g_t`：

```text
Qwen3-Embedding(g_t) -> frozen RQ -> (c_1, ..., c_L)
```

首次遇到 n-gram 时执行 embedding 和 RQ 编码，随后把 codes 与必要的量化统计写入 persistent cache。热路径仍为确定性查表。训练和推理使用同一冻结 encoder/codebook。

### 3.2 保留 RQ level，而不是 flatten

当前实现：

```text
[2/3-gram × L levels × d_head] -> flatten -> one ContextAwareGate
```

新实现对每个 n-gram order 保留 level 轴：

\[
e_{t,l}=M_l[c_{t,l}], \qquad
v_{t,l}=W^V_l e_{t,l}.
\]

gate 输入包括当前 hidden state、level-specific memory value 和 level embedding：

\[
s_{t,l}=g(h_t,e_{t,l},r_l).
\]

第一版不引入庞大网络，只使用低秩投影和标量 gate，保证新增激活计算可忽略。

### 3.3 有序的自适应读取深度

RQ 后级编码的是前级未解释的残差，因此读取不能把 level 当作无序 experts。定义 continuation probability：

\[
p_{t,l}=\sigma(s_{t,l}), \qquad
a_{t,l}=\prod_{j=1}^{l}p_{t,j}.
\]

聚合 memory：

\[
m_t=\sum_{l=1}^{L}a_{t,l}v_{t,l}.
\]

直觉：浅层足够时停止；只有上下文判断需要更具体的 residual correction 时才继续深入。另保留一个显式 no-memory route，使错误共享可以完全被拒绝。

### 3.4 反事实 utility credit

普通 LM loss 只评价实际执行的一条 memory route，无法告诉 gate 未选择的 route 是否更好。训练时为部分 batch 构造配对 route：

```text
route A: no-memory 或 prefix depth d-1
route B: prefix depth d
```

由目标 token 的 log-likelihood 差构造停止梯度的边际效用：

\[
u_{t,d}=\operatorname{sg}\left[
\log p(y_t\mid h_t,m_{\le d})-
\log p(y_t\mid h_t,m_{<d})
\right].
\]

gate 用 pairwise ranking / calibrated binary loss 预测 `u_{t,d}>0`：

\[
\mathcal L_{credit}=\operatorname{BCE}(p_{t,d},\mathbb 1[u_{t,d}>\epsilon]).
\]

总目标：

\[
\mathcal L=\mathcal L_{LM}+\lambda_c\mathcal L_{credit}
+\lambda_d\mathbb E[\text{read depth}].
\]

训练开销通过 batch 内复制一部分样本实现，不做 8 次完整 forward。正式 sweep 比较 25%、50%、100% paired examples；推理只有一次 forward，不需要 utility oracle。

### 3.5 正样本与负样本

- 正 utility：当前事实 canonical supervision；
- transfer supervision 不使用 benchmark 测试 paraphrase，避免泄漏；
- 负 utility：batch 内其他事实的 prompt、原数据提供的 locality prompt，或从训练 split 构造的 unrelated query；
- 多个事实共享地址但答案冲突时，作为 hard negative；
- 所有样本来源和构造规则在训练前冻结。

## 4. 为什么它可能有效，以及为什么不是普通 A+B

Semantic RQ 与 utility gate 解决同一个 transfer-interference 问题的两个必要部分：

```text
Semantic RQ        -> 产生可迁移的共享候选
RQ hierarchy       -> 表达共享的粒度
Counterfactual gate-> 判断每一级共享是否真正降低当前损失
```

只有 semantic hash：产生共享，但无法避免 false sharing。  
只有普通 gate：没有语义结构，无法把新表述路由到已写 memory。  
只有 level attention：仍然依赖单路线 LM loss，不能解决 Engram-Nine 指出的 credit mismatch。

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
- Semantic-RQ + hierarchical ordered gate，但无 credit loss；
- **CREDIT-Engram**。

公平约束：相同 backbone、target layers、memory rows、embedding dimension、训练 tokens、optimizer search budget。除系统表外，所有 Engram 变体保持相同 trainable parameter budget；若新增 gate 参数，给 baseline 增加同规模 projection 的参数匹配对照。

### 5.4 核心消融

1. 固定 prefix depth `d in {1,2,4,6,8}`；
2. unordered softmax vs ordered continuation gate；
3. 无 credit、on/off credit、depth-pair credit；
4. Semantic vs Shuffled vs Arithmetic；
5. 去掉 no-memory route；
6. 去掉 locality/hard-negative credit；
7. paired-example 比例和 `lambda_credit`；
8. inference 时强制 depth，验证学习到的深度是否具有因果作用。

超参只在 WikiBigEdit validation timestep 上选择一次，其他 benchmark 直接迁移。

### 5.5 机制分析

必须证明下面的链条，而不是只画 gate heatmap：

```text
语义相似
-> RQ prefix overlap 增加
-> 浅层 row 被相关改写复用
-> 该层反事实 utility 为正
-> gate continuation 与 utility 对齐
-> paraphrase / ripple gain
```

报告：

- 每层 prefix 的 embedding cosine、relation/entity purity 和 collision load；
- 每层 counterfactual utility 分布；
- gate score 与 utility 的 AUROC、ECE、Spearman；
- paraphrase、conflict、locality、head/tail entity 的读取深度；
- reset shared rows、shuffle codes、强制 gate/depth 后的性能变化；
- Semantic 相对 Shuffled 的收益按 code overlap 和 utility 分桶。

### 5.6 效率与系统属性

- 首次 semantic encode + RQ 的 cold latency；
- persistent cache 的 warm lookup latency和 hit rate；
- Base、Arithmetic、Semantic-RQ、CREDIT-Engram 的 train/inference tokens/s；
- counterfactual paired training 的额外 FLOPs 和 wall time；
- GPU table 与 pinned CPU/offloaded table；
- memory size、optimizer state、实际 touched rows；
- adaptive depth 带来的平均有效读取层数，但不得把当前未实现的稀疏 gather 宣称为实际加速。

## 6. 统计规范

- 主结果 3 seeds，报告 mean ± std；
- query-level paired bootstrap 95% CI；
- 多个 WikiBigEdit milestone 使用同一 chronological stream 和固定 evaluation cohorts；
- 所有方法使用相同数据顺序；
- 预注册 primary metric：WikiBigEdit Rephrase 与 Retention 的 harmonic mean，同时约束 Locality；
- 不以最好 seed、最好 checkpoint 或测试集调参结果作为主表；
- 同时报告绝对值和相对 Arithmetic、Shuffled、hierarchical-no-credit 的差值。

## 7. 成功门槛

论文的关键门槛不是 CREDIT 比 Base 高，而是：

1. CREDIT-Engram 在 WikiBigEdit 50K/100K 的主指标上稳定优于 Semantic-RQ flatten；
2. 稳定优于 hierarchical-no-credit，证明收益来自 credit assignment，而不只是更多参数；
3. Semantic-CREDIT 优于 Shuffled-CREDIT，证明 semantic ordering 终于被模型利用；
4. Locality 不以不可接受的幅度下降，且综合 Pareto 优于 LoRA/FT；
5. 至少一个外部 benchmark 复现相同方向；
6. gate–utility 对齐指标和 intervention 支持因果解释。

建议预注册最小有效差异：WikiBigEdit 主指标相对 strongest matched Engram baseline 至少 +1.5 pp，三个 seed 方向一致，paired CI 排除 0。若只有 +0.2 pp 或只在单个 milestone 成立，不足以支撑 ICLR 方法论文。

## 8. 四张 A100 的执行计划

### Phase A：地址层级审计（约 0.5--1 GPU-day）

不训练新模型，完整扫描 WikiBigEdit train/eval：prefix purity、overlap、collision、量化残差。如果不存在层级趋势，立即停止 ordered-depth 路线。

### Phase B：完整 10K 方法开发（约 4--8 GPU-days）

四卡并行：flatten、hierarchical-no-credit、CREDIT、Shuffled-CREDIT。每个 run 都完整训练到 10K，并评官方全部指标；这不是 smoke test，而是方法选择阶段。固定 depth sweep 随后并行补齐。

### Phase C：50K/100K 主表（约 18--30 GPU-days）

冻结所有超参后，主要 Engram 方法三 seed。四卡流水并行。昂贵的领域基线按官方建议规模运行并完整记录失败/不适用项。

### Phase D：外部 benchmark、模型尺度与机制（约 12--20 GPU-days）

CounterFact、ZsRE、Ripple；Qwen3-0.6B/4B 尺度；intervention、效率和 offload。

保守总预算约 35--60 A100-40G GPU-days，即四卡连续约 9--15 天。实际预算必须在 Phase B 记录每个 run 的 wall time 后更新，不能用估算冒充测量。

## 9. 投稿故事

### 标题方向

**Who Should Share a Memory? Counterfactual Credit Assignment for Semantic Conditional Memory**

### 摘要核心

现有 conditional memory 使用确定性地址检索静态模式，但地址共享与 memory utility 之间没有显式信用分配。我们首先发现 semantic addressing 在大规模 lifelong editing 中带来共享收益，却无法稳定超越打乱语义的结构对照；并将其归因于扁平读取和 route utility 不可观测。CREDIT-Engram 使用层级 RQ 地址表达共享粒度，并通过低开销反事实 route pairs 监督 ordered gate 的边际 utility。在标准 lifelong knowledge editing、知识编辑与机制干预中，它应当实现更好的 transfer-retention-locality Pareto，同时保留冻结 backbone、可缓存地址与可 offload memory。

### 预期贡献

1. 诊断贡献：证明 conditional memory 中 address quality 与实际 route utility 的脱节；
2. 方法贡献：counterfactual utility-supervised hierarchical semantic memory；
3. 实证贡献：标准 WikiBigEdit 大规模连续写入、标准 KE 外部验证和完整因果干预；
4. 系统贡献：报告冷/热 semantic addressing 和 counterfactual training 的真实成本，而不把 memoized lookup误写成无条件 O(1)。

## 10. 风险与止损

- **RQ prefix 没有粗到细语义**：保留 counterfactual utility gate，但把 route 改为多个独立 semantic code heads；不能继续声称 hierarchy。
- **credit loss 只提高训练集 efficacy**：检查 hard negatives、locality utility 与 gate calibration；若外部改写无增益，方法不成立。
- **Semantic-CREDIT 仍等于 Shuffled-CREDIT**：说明当前 n-gram embedding 几何不适合 edit utility；论文应转为一般 utility-gated Engram，而不能把 semantic hash列为核心贡献。
- **两倍训练代价过高**：降低 paired-example 比例并缓存 utility labels，但主表必须如实报告训练成本。
- **领域 KE baseline 不兼容 Qwen3**：优先复用 WikiBigEdit 官方支持的 backbone/方法；不得用无法运行的名字装饰 baseline 表。

这份计划的目标不是承诺一定得到正结果，而是让每个关键 claim 都对应标准 benchmark、严格对照和可证伪证据。只有通过成功门槛后，才把 CREDIT-Engram 定为最终投稿方法。
