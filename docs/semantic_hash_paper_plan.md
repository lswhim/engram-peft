# Semantic Hash for Engram：论文规划（Reviewer-Oriented）

> 精简后的最终投稿规划见：`docs/semantic_hash_paper_plan_final.md`。本文件保留完整实验审计、运行细节与历史决策。

> 状态：执行中；Gate 0 已完成，Gate 1 与外部验证正在运行
> 更新时间：2026-08-13
> 代码分支：`semantic-hash`
> 核心资源：4 × A100 40GB

## 1. Executive Summary

本文不把 Semantic-RQ 描述成“更准确的 hash”，也不把论文包装成知识编辑、跨语言或系统 offload 工作。

论文只回答一个问题：

> **当有限的 Engram 表迫使不同 n-gram 共享参数时，语义结构化共享是否比随机共享产生更好的泛化？**

原始 Engram 使用算术 hash。它具有确定性、常数时间查找和可 offload 的优势，但发生共享或碰撞的 n-gram 之间没有语义关系。Semantic-RQ 使用冻结文本编码器和 Residual Quantization，将语义相近的 n-gram 映射到部分相同的多头地址，希望把随机碰撞转化为可利用的结构化共享。

核心因果链为：

```text
semantic similarity
    → shared RQ codes
    → shared Engram rows
    → source gradient reaches unseen semantic neighbors
    → lower target NLL / better downstream generalization
```

论文是否成立，不由某一个 benchmark 的总分决定，而由以下三条证据共同决定：

1. 在参数和访问频率严格匹配时，Semantic-RQ 优于 Arithmetic 和 RQ-Shuffled；
2. 收益集中在“训练中 exact-unseen、但存在语义邻居且地址可覆盖”的样本；
3. 目标收益随“命中已更新共享行”的程度增加，并且不造成不可接受的 false sharing。

知识编辑、XNLI/PAWS-X、Biomedical adaptation 都是外部验证场景，不是论文定义本身。

---

## 2. 论文主张与非主张

### 2.1 唯一主张

> **Semantic-RQ provides a structured collision prior for finite conditional memory: under matched capacity, semantically related n-grams share trainable rows, improving generalization to lexically novel semantic neighbors over random hashing.**

中文表述：

> 在有限条件记忆中，Semantic-RQ 将不可避免的随机参数共享转化为层级语义共享，从而提高对词面未见、语义相关表达的泛化。

### 2.2 可以作为次级贡献的内容

- RQ 多级代码是否形成从粗到细的共享结构；
- Semantic/Arithmetic mixed heads 是否在 transfer 与 specificity 之间取得更好平衡；
- 静态冻结地址表保持 O(1) lookup、训练/推理地址一致；
- Semantic-RQ 在多语言、领域适配和知识更新中的外部有效性。

### 2.3 明确不宣称

- 不宣称 Semantic-RQ 消除碰撞；
- 不宣称碰撞越少越好；
- 不宣称 Semantic-RQ 天然比 Arithmetic 更容易 CPU/SSD offload；
- 不宣称任意 OOV n-gram 都能获得语义泛化；当前 OOV 会回退 Arithmetic；
- 不宣称 Engram memory values 可直接跨 tokenizer、跨 hidden space 或跨任意 backbone 迁移；
- 不宣称本文提出新的通用 knowledge editing 算法；
- 不把单一模型、单一 seed 或单一 benchmark 的小幅领先写成普遍结论。

---

## 3. Reviewer 最可能提出的质疑

### R1：这只是外部 embedding model 带来的知识，不是 hashing 的贡献

应对：

- 使用 RQ-Shuffled：保留相同编码器、相同 code 数、相同桶频率和相同表容量，只破坏 n-gram 与语义 code 的对应关系；
- 报告离线编码器、建表语料和计算成本；
- 主结果成立后再增加轻量编码器或 lexical-clustering ablation，检验结果是否依赖特定 Qwen embedder。

### R2：Semantic-RQ 与 Arithmetic 的参数量和碰撞率不公平

应对：

- 以 Arithmetic-matched 为主基线：相同 n-gram order、head 数、每头行数、embedding 维度、注入层和优化器；
- 同时报原版 Arithmetic，便于与已有 Engram 设置连接；
- 报告每种方法的有效参数量、访问频率分布、平均 bucket load、最大 bucket load 和碰撞熵。

### R3：你只在知识编辑上有效，与原始 Engram 的语言建模目标不一致

应对：

- 主实验必须包含受控语言建模；
- 知识编辑只占应用验证的一部分；
- 主指标首先是 token NLL/PPL 及频率、覆盖率、语义邻居切片，而不是编辑成功率。

### R4：总体提升可能来自 gate/projection，而不是地址共享

应对：

- 受控机制实验中使用两阶段协议：先校准共享接口，再冻结 gate/projection/conv，只更新 memory rows；
- 另报 joint-training setting，说明实际使用性能；
- 对目标样本记录其检索行是否在 source 训练中被实际更新。

### R5：地址表偷看了测试集或目标语言

应对：

- 主协议中 RQ dictionary 只能由预先声明的训练/外部无标签语料构建；
- 禁止从 benchmark test text 建表；
- 训练、建表和测试语料做 exact 与近重复去重；
- 报告 target address coverage，并单独评估 address-OOV；
- 若增加 transductive dictionary 结果，必须明确标注，仅作上界。

### R6：语义共享会污染相似但事实不同的样本

应对：

- 把 false sharing 设为主要安全指标，而不是事后 case study；
- 使用高词面相似但语义不同、同实体不同 relation、多义词、否定和时间变化样本；
- 比较纯 Semantic-RQ 与 Mixed heads。

### R7：为什么不用更精确的索引或 MPHF？

应对：

- 本文问题不是消除碰撞，而是组织有限容量下的共享；
- Engram-Nine 已表明 collision-free 不必然改善训练；
- MPHF/Exact lookup 可作为容量上界，但不是必须主基线。

---

## 4. 方法与公平比较

### 4.1 方法定义

对 n-gram `x`，Arithmetic Engram 使用多个独立算术 hash：

```text
h_l(x) = ArithmeticHash_l(x) mod K
```

Semantic-RQ 使用冻结编码器 `Enc` 和 L 级残差量化器：

```text
z(x) = Enc(x)
c(x) = RQ(z(x)) = [c_1(x), ..., c_L(x)]
```

第 `l` 个 code 直接索引第 `l` 个 Engram head。运行时只查询冻结的 `n-gram → codes` 表，不在线调用 encoder。

### 4.2 核心方法矩阵

| 方法 | 用途 | 是否进入主表 |
|---|---|---:|
| Base / No memory | 衡量新增 memory 的绝对收益 | 是 |
| Original Arithmetic | 与现有 Engram 实现对齐 | 是 |
| Arithmetic-matched | 严格参数匹配的核心对照 | 是 |
| RQ-Shuffled | 隔离“语义组织”本身 | 是，决定性基线 |
| Semantic-RQ | 提出的方法 | 是 |
| Mixed Semantic + Arithmetic | 控制 false sharing | 是或主要消融 |
| LoRA | 实际 PEFT 外部参照 | 下游应用表 |
| Full fine-tuning | 性能上界/遗忘参照 | 仅领域适配或 KE |

#### Arithmetic-matched 的实现约束（2026-08-13 审计）

原始 `NgramHashMapping` 不能通过简单设置总容量来成为严格 matched：它为了遵循
Engram 原实现，会跨 layer/head 分配全局不同的递增质数 modulus。配置总容量 2048
时，实际四个 layer/order 容量分别是 2194、2612、3000、3396，而不是统一的
`8 × 256`。

因此主表使用新增的 `arithmetic_fixed` 控制后端：

- 每个 n-gram order 精确 8 heads；
- 每个 head 精确 256 rows；
- 每个 layer/order 精确 2048 rows；
- head 独立性来自独立的奇数 multiplier vector，而不是不同 table size；
- 与 Semantic-RQ 的可训练参数均为 26,984,448。

此前产生的低容量 matched 和 prime-capacity v2 结果均只保留为诊断，不进入主表。

### 4.3 RQ-Shuffled 的正确构造

RQ-Shuffled 不能简单随机重新采样 code，否则 bucket load 会变化。建议：

1. 先在对应 seed 的 LM train split 上统计每个静态表项的精确访问次数；
2. 按精确访问次数分组，只在同频 n-gram 之间置换完整 RQ code 向量；
3. 因组内每行权重完全相同，每个 level 中每个 code 的**实际训练访问次数**逐桶
   完全不变，同时 joint code-vector 分布也不变；
4. 固定 permutation seed，并使用至少 3 个 shuffle seeds 验证不是某次随机结果。

早期 Gate 0 seed42 的全表行置换已验证：2-gram moved fraction 1.0，3-gram
0.99998，所有 level 的未加权 histogram 完全保持；它足以用于地址结构干预，
但不作为正式 LM 主对照。正式 fixedsteps 主表改用每 seed 独立的
frequency-matched shuffle，并同时验收未加权与 access-weighted histogram。

### 4.4 Gate 0 地址诊断实测（2026-08-13）

诊断从建表使用的前 5,000 篇 FineWeb 文档恢复真实 observed surface，分别抽样
4,000 个 2-gram 和 4,000 个 3-gram。候选 pair 由 semantic nearest neighbor、
char-trigram lexical nearest neighbor 与 random pair 合并；语义表示使用冻结的
Qwen3-Embedding-0.6B。RQ-Shuffled 使用与主实验相同的 seed42 整向量置换表。

| Order | Pair 数 | Spearman(semantic, RQ overlap) | Spearman(semantic, shuffled overlap) | 高语义/低词面 RQ overlap | Shuffled overlap | Held-out coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 9,275 | 0.738 | 0.006 | 19.07% | 0.72% | 64.98% |
| 3 | 9,252 | 0.699 | -0.011 | 16.90% | 0.70% | 27.25% |

高词面、低语义象限的 RQ overlap 分别为 1.24%（2-gram）和 1.62%（3-gram），
远低于高语义/低词面象限。这说明当前地址确实携带语义结构，而非仅复现词面相似；
整向量 shuffle 几乎完全移除了该结构。

但 Gate 0 只证明地址机制存在，不证明它会改善 LM。尤其 3-gram held-out coverage
仅 27.25%，会稀释总体收益，必须在 Gate 1 同时报 covered slice 与 address-OOV；
不能仅凭上述相关性宣称模型性能提升。

### 4.5 FineWeb 数据隔离

- 流的前 5,000 篇仅用于离线地址表构建；
- LM train/eval 固定跳过前 6,000 行，留出 1,000 行安全间隔；
- 跳过后再执行固定 seed shuffle；
- train 与 eval 再从同一 shuffle 后序列做不重叠切分。

### 4.6 两种训练协议

#### Protocol A：Mechanism-Isolated

```text
1. 在独立 calibration corpus 上训练 gate/projection/conv；
2. 冻结 backbone、gate、projection、conv；
3. 只在 source corpus 上更新 memory rows；
4. 在 target corpus 上零更新评测。
```

用于回答“迁移是否来自共享 memory rows”。

#### Protocol B：End-to-End Engram Adaptation

```text
冻结 backbone；共同训练 Engram table + gate/projection/conv。
```

用于回答实际应用效果。Protocol B 不能单独承担机制结论。

---

## 5. 实验总览

实验不是四个平行故事，而是三层证据：

| 证据层 | 实验 | 回答的问题 | 论文地位 |
|---|---|---|---|
| Layer I | 受控语言建模 | Semantic-RQ 是否改善 Engram 的本职任务 | 主实验 |
| Layer II | 因果机制与边界 | 改善是否真的由结构化共享产生 | 主实验 |
| Layer III | 多语言、领域适配、知识更新 | 机制是否具有外部有效性 | 应用验证 |

---

## 6. Layer I：受控语言建模主实验

### 6.1 为什么必须做

原始 Engram 是条件语言记忆模块。若论文没有 LM loss/PPL，reviewer 有充分理由认为我们只是在把 Engram 当成另类 adapter，而没有证明地址改造对 Engram 本身有意义。

### 6.2 两个规模

#### LM-Controlled：约 185M–300M，从头训练

- 数据：FineWeb-Edu，先做 100M-token pilot，再扩至约 1B tokens；
- context length：1024；
- n-gram orders：2、3；
- 核心方法：Arithmetic-matched、RQ-Shuffled、Semantic-RQ、Mixed；
- 至少 3 seeds；
- 所有方法使用相同 tokenizer、训练 tokens、activated compute 和总 memory 参数。

目的：最干净地比较地址共享结构，不受已有 backbone 知识混淆。

#### LM-PostHoc：Qwen3-0.6B/1.7B

- 冻结 backbone；
- 在 FineWeb-Edu 或领域语料上只训练 Engram；
- 以 Qwen3-0.6B 做完整矩阵，1.7B 做复现；
- 主要验证结论可迁移到真实预训练 LLM。

### 6.3 标准评测

- FineWeb-Edu held-out PPL；
- WikiText-103 PPL；
- C4 held-out PPL；
- LAMBADA；
- 下游：ARC-E/ARC-C、HellaSwag、PIQA、SciQ、WinoGrande、MMLU（模型足够大时）。

### 6.4 决定性的 token 切片

每个 eval token 按其 suffix n-gram 分类：

1. `exact_seen`：训练中出现过相同 n-gram；
2. `semantic_neighbor`：exact 未见，但地址表中存在训练过的高语义相似邻居；
3. `long_tail`：低频但 address-covered；
4. `covered_no_neighbor`：地址覆盖但没有高相似训练邻居；
5. `address_oov`：不在静态 RQ dictionary，走 Arithmetic fallback。

主指标：

```text
NLL_exact_seen
NLL_semantic_neighbor
NLL_long_tail
NLL_covered_no_neighbor
NLL_address_oov
Overall PPL
```

#### 已实现的可复现切片协议（2026-08-13）

- 每个 seed 使用与训练完全相同的 FineWeb stream、`skip(6000)`、shuffle seed、
  48,832 train rows 与随后 200 eval rows；manifest 一次构建后供所有方法共享。
- 因果 LM 对齐按 context position 定义：位置 `t` 结束的 n-gram 对应模型在
  `logits[t]` 上预测 token `t+1` 的 NLL；不把 target token 错当作当前地址输入。
- `exact_seen` 由压缩 n-gram key 在 LM train rows 中是否真实出现定义。
- `semantic_neighbor` 必须同时满足：exact unseen、静态地址表 covered，并且用冻结
  Qwen3-Embedding-0.6B 得到的最近 LM-train n-gram cosine 超过预先固定阈值
  （2-gram 0.79、3-gram 0.76）。该定义独立于 RQ codes，避免循环论证。
- 同时保存最近邻 char-trigram Jaccard 与 RQ code overlap；低词面语义邻居使用
  Jaccard ≤ 0.10，直接检验非词面复用。
- `semantic_neighbor` 再按与最近 LM-train 语义邻居是否至少共享一个 RQ code
  拆成 `shared-code` 与 `no-shared-code`；若机制成立，收益应集中在前者。后者是
  embedding similarity 相近但缺少直接结构共享的内部负对照。
- 每个模型保存逐 token loss，从而可做严格配对 bootstrap，而非只比较聚合均值。
- 配对差值固定定义为 `ΔNLL = Semantic-RQ − control`（负值更好）；bootstrap
  以文档为 cluster，并在三 seed 间做嵌套重采样，10,000 replicates。只有 95% CI
  上界低于 0 才写“显著优于”。

自动队列包含 3 个 manifest，以及 Base、Arithmetic-matched、RQ-Shuffled、
Semantic-RQ、Mixed 在 3 seeds 上的 15 个逐 token 评测。

核心预期不是 Semantic-RQ 在所有 token 上全面领先，而是：

```text
semantic_neighbor / long_tail: Semantic-RQ > RQ-Shuffled ≈ Arithmetic-matched
exact_seen:                    Semantic-RQ ≈ Arithmetic-matched
address_oov:                   Semantic-RQ ≈ Arithmetic fallback
```

### 6.5 语言建模主表

| Method | Overall PPL ↓ | Exact-seen NLL ↓ | Semantic-neighbor NLL ↓ | Long-tail NLL ↓ | Address-OOV NLL ↓ |
|---|---:|---:|---:|---:|---:|
| Arithmetic-matched | | | | | |
| RQ-Shuffled | | | | | |
| Semantic-RQ | | | | | |
| Mixed | | | | | |

---

## 7. Layer II：机制、因果与边界

### 7.1 Shared-row exposure，而不只是 embedding similarity

对 target n-gram `x_t`，记录它检索的 rows 中有多少在 source 训练中被实际更新：

```text
exposure(x_t) = # retrieved rows updated by source / # retrieved rows
```

目标变量：

```text
target_gain(x_t) = NLL_before(x_t) - NLL_after(x_t)
```

主图：

```text
target_gain vs shared-row exposure
```

需要同时报告 Arithmetic、RQ-Shuffled、Semantic-RQ。为了避免“相似度越高、地址越相同”这一构造性相关被误当成机制，应当：

- 在相同 embedding-similarity 区间内比较不同 exposure；
- 在相同 lexical-overlap 区间内比较不同 exposure；
- 使用 mixed-effects regression 或分层 bootstrap，控制频率、长度、词面重合和 base NLL。

### 7.2 地址干预

至少做一种直接干预：

- `code permutation`：保持训练好的 memory rows不变，只替换 target 的 code mapping；
- `head masking`：逐步屏蔽 target 与 source 共享的 heads；
- `row reset`：只重置 source 与 target 共同命中的 rows。

若 target gain 随共享 head 被屏蔽而下降，因果证据比简单相关更强。

### 7.3 Memory capacity sweep

固定其他设置，调整每头行数：

```text
K ∈ {32, 64, 128, 256, 512, 1024}
```

或使用等比总 memory budget。预期：

- 小容量：共享不可避免，Semantic-RQ 的结构化碰撞优势最强；
- 大容量：exact-seen 差异缩小；
- semantic-neighbor 上仍可能保留共享收益。

该曲线直接支撑“finite memory”而不是“better indexing”的论文定位。

### 7.4 RQ level 分析

不能预设前级一定是语义 coarse、后级一定是 semantic fine；必须实证。

评测：

- 每级 code 的 semantic purity；
- 每级 bucket entropy、频率和 false-sharing rate；
- coarse-only、fine-only、full-RQ；
- semantic-first/random-last 与 random-first/semantic-last mixed variants。

只有在证据支持时，论文才使用“hierarchical semantic sharing”措辞；否则只称“multi-code semantic sharing”。

### 7.5 False-sharing benchmark

构造或从标准数据中抽取四象限：

| 组别 | 语义 | 词面 | 目标 |
|---|---:|---:|---|
| A | 相近 | 相近 | 应共享 |
| B | 相近 | 不同 | 检验真正语义迁移 |
| C | 不同 | 相近 | 检验错误共享 |
| D | 不同 | 不同 | 无关对照 |

重点类别：

- 同实体、不同 relation；
- 同 relation、不同 entity/object；
- 多义词；
- 否定；
- 时间变化和过期事实；
- PAWS 风格高词面重合但语义不同。

指标：

```text
Positive Transfer ↑
False Transfer ↓
Neighborhood Damage ↓
Specificity ↑
```

Mixed heads 的价值必须体现在降低 false transfer，而不是仅凭平均分略高。

---

## 8. Layer III-A：跨语言与跨表述外部验证

### 8.1 Benchmark

- XNLI：英语监督训练，15 语言零 target updates；
- PAWS-X：英语监督训练，7 语言零 target updates；
- 可选 mParaRel 或多语言 ZsRE：同一事实的跨语言问法。

### 8.2 当前 XNLI/PAWS-X 的正确定位

现有设置每个 benchmark 都重新训练模型，因此证明的是：

```text
cross-lingual task generalization under shared semantic addresses
```

它不证明：

```text
XNLI learned memory values transfer directly to PAWS-X
```

因此 XNLI/PAWS-X 进入外部验证表，不进入 memory portability 主张。

### 8.3 必补内容

- RQ-Shuffled；
- 3 seeds；
- 每语言 RQ coverage；
- 每样本 shared-row exposure；
- semantic similarity × lexical overlap 四象限；
- 目标语言不参与有标签训练；
- RQ dictionary 不读取 benchmark test text。

### 8.4 成功标准

Semantic-RQ 不能只在 macro 上领先 Arithmetic；还需满足：

1. 对低词面但语义相近样本提升更明显；
2. 对 PAWS 式高词面、语义不同样本不增加错误；
3. 提升与 shared-row exposure 有关；
4. RQ-Shuffled 不能复现同等提升。

---

## 9. Layer III-B：Biomedical post-hoc adaptation

### 9.1 为什么只选一个领域

领域适配用于证明现实价值，不用于堆 benchmark。先把 Biomedical 做扎实，比同时浅跑医疗、金融和法律更有说服力。

### 9.2 训练与评测

- Backbone：Qwen3-0.6B 完整矩阵，Qwen3-1.7B 复现；
- 训练：Biomed-Enriched，冻结 backbone；
- 评测：MedQA、MMLU Medical、PubMedQA（修复数据依赖后）、医学 held-out PPL；
- 通用保持：MMLU general、ARC、HellaSwag、通用语料 PPL。

方法：

- LoRA；
- Arithmetic-matched；
- RQ-Shuffled；
- Semantic-RQ；
- Mixed。

分析切片：

- 常见医学术语；
- long-tail 术语；
- 同义术语和缩写；
- 词面相似但概念不同的医学实体；
- address-OOV。

### 9.3 该实验支撑的结论

只在以下情况下写入论文主结论：

- Semantic-RQ 对 long-tail/同义表达的收益显著高于 RQ-Shuffled 和 Arithmetic；
- 通用能力保持不劣于 Arithmetic；
- false-sharing 可控。

若只在医学平均分上略高、机制切片不成立，则作为附录结果，不支撑核心 claim。

---

## 10. Layer III-C：知识编辑仅作为应用验证

### 10.1 数据集

- CounterFact；
- ZsRE；
- KnowEdit-WikiRecent；
- WikiBigEdit 作为批量更新扩展；
- MQuAKE 仅在指标定义可靠且结果可解释时保留。

### 10.2 标准指标

- Efficacy / Rewrite；
- Paraphrase Generalization；
- Specificity / Neighborhood；
- Locality；
- 批量更新后的 retention。

### 10.3 论文中如何使用

知识编辑最多承担以下结论：

> 结构化共享允许一次局部 memory update 被等义查询复用，同时保持相邻事实相对稳定。

它不能单独证明 Semantic-RQ 改善通用 Engram，也不能把论文改写成新的 editing method。

已有 5 个 KE 数据集 × 4 个 Qwen3 尺度结果可复用，但主文优先整理 CounterFact、ZsRE、WikiRecent；PopQA 不进入主表。

---

## 11. 统计与报告规范

### 11.1 Seeds

- Pilot：1 seed，只用于淘汰失败设置；
- 主结果：至少 3 seeds；
- RQ-Shuffled：至少 3 shuffle seeds，避免对单一 permutation 过拟合。

### 11.2 显著性

- token NLL：paired bootstrap，以相同 token/example 配对；
- 分类准确率：paired bootstrap 或 McNemar；
- 多任务平均：报告 task-level macro 与每任务结果，不能只报平均；
- 多重比较：主终点预先指定，其余指标明确为 secondary/exploratory。

### 11.3 预注册的主终点

建议主终点为：

```text
Semantic-neighbor slice 上，Semantic-RQ 相对 Arithmetic-matched 的 paired ΔNLL
```

主要安全终点为：

```text
False-sharing / neighborhood damage 相对 Arithmetic-matched 的差值
```

### 11.4 必报工程信息

- 总参数和训练参数；
- 每 token 激活的 memory rows；
- table storage；
- RQ 建表时间和编码器成本；
- train/eval throughput；
- GPU peak memory；
- address coverage 和 fallback rate。

系统 offload 可作为附录，不把其写成 Semantic-RQ 独有贡献。

---

## 12. Go / No-Go 标准

本规划的目的不是保证论文故事成立，而是尽快证伪错误方向。

### Gate 0：地址结构成立

在不训练 LLM 的情况下验证：

- Semantic-RQ 在低词面、高语义 pair 上具有更高 code overlap；
- 在高词面、低语义 pair 上不过度共享；
- RQ-Shuffled 消除这种结构；
- address coverage 达到可评测水平。

若 Gate 0 不成立，停止大规模训练，先修建表/表示方法。

当前判定：**通过**。实测结果见 4.4 节。该判定仅允许继续 Gate 1，不等于论文
核心性能主张已经成立。

### Gate 1：受控 LM pilot 成立

100M-token pilot 必须至少满足：

- Semantic-neighbor NLL 优于 Arithmetic-matched 和 RQ-Shuffled；
- overall validation PPL 不出现明显退化；
- false-sharing 不显著恶化。

若只在训练 loss 上领先、held-out slice 不领先，停止扩容。

### Gate 2：主结果可复现

在至少两个独立设置（例如 200M from-scratch 与 Qwen3-0.6B post-hoc）中：

- 主终点方向一致；
- 95% paired bootstrap CI 排除 0；
- 效果不能由单一 seed、单一 RQ table 或单一 benchmark 驱动。

### Gate 3：外部有效性

跨语言、Biomedical、KE 三类应用不要求全部显著，但至少两个场景应观察到与机制一致的提升，而不是无规律交替。

### Pivot 条件

若出现以下情况，不继续强推纯 Semantic-RQ：

- Semantic-RQ 与 RQ-Shuffled 无稳定差异；
- 收益只来自 exact-seen，不来自 semantic-neighbor；
- false-sharing 抵消 transfer；
- 只有单个知识编辑 benchmark 有效；
- 收益主要来自 gate/projection joint training。

可选 pivot：

- Mixed semantic/random heads；
- frequency-aware semantic sharing；
- uncertainty-aware gate；
- 将论文改为对 structured collision 的系统性负结果/诊断，而不是性能方法论文。

---

## 13. 当前执行矩阵（2026-08-13）

- Gate 1 LM pilot：Arithmetic-matched（精确容量）、RQ-Shuffled、Semantic-RQ、
  Mixed，各 3 seeds，共 12 runs；由持续调度器占用空闲 GPU 自动推进。
- 每个 seed 的训练完成后，自动构建固定 token-slice manifest，并对 Base + 4 种
  memory 方法执行逐 token NLL，共 3 manifests + 15 slice evaluations；结果进入
  同一实时 HTML。
- 当前首批：Arithmetic-matched seed42 与 RQ-Shuffled seed42 正在训练；完成后自动
  进入 Semantic-RQ 与 Mixed，再进入 seed43/44。
- 运行审计发现首批旧配置启用了 `EarlyStoppingCallback`：RQ-Shuffled 在
  1,600/12,208 steps 停止（约 13M 而非 100M tokens），且 adapter-only checkpoint
  无法由 Transformers 的通用 best-model loader 自动恢复。因此这些旧 run 全部降级为
  诊断，不进入主表。正式 `fixedsteps` run 禁用 early stopping 与 best-model reload，
  每个方法严格完成 12,208 optimizer steps，即
  `12,208 × 4 × 8 × 256 = 100,007,936` processed token slots，并保存最终步模型。
  48,832 rows 在 effective batch 32 下恰为 1,526 steps/epoch，因此 12,208 steps
  等于完整 8 epochs；frequency-matched shuffle 用单个完整 epoch 的访问计数建组，
  全训练轨迹只乘统一常数 8，其 access-weighted bucket load 保持是严格相等而非近似。
  该值包含 padding slots，不冒充实际 non-padding token 数；所有方法数据与 padding
  完全一致，因此它仍是公平的计算预算。正式结果 JSON 必须满足
  `fixed_steps_complete=true` 且 `completed_steps=planned_steps=12,208`，否则调度器、
  HTML 和 bootstrap 都不接纳。
- 正式 run 每 1,000 steps 保存 checkpoint、每 500 steps 做 held-out eval，避免频繁
  CFS I/O 改变吞吐；最终 JSON 另外记录实际 `non_padding_token_presentations` 与
  `causal_target_token_presentations`，主文报告真实值而非只报含 padding 的 slots。
- 外部验证：XNLI 的 exact Arithmetic seed42 与 Semantic-RQ seed43 正在并行；
  现有旧 matched 结果因容量不公平只作为诊断，不进入论文主表。
- 实时记录：`outputs/semantic_hash_paper/dashboard.html`；所有尚未完成的格子显示为
  pending/running，不以中途 train loss 替代最终结论。

论文 go/no-go 的下一决定点是 Gate 1：只有 Semantic-RQ 在严格容量匹配下优于
Arithmetic-matched 与频率完全匹配的 RQ-Shuffled，且收益集中在
semantic-neighbor/covered slice、false-sharing 不恶化，才进入扩规模主实验。

---

## 13. 4 × A100 40GB 执行路线

以下为相对预算，实际以 pilot throughput 校准。

### Phase 0：地址诊断（0.5–1 天）

- 构建标准 pair 数据和四象限切片；
- 统计 overlap、coverage、bucket load、semantic purity；
- 实现正确的 RQ-Shuffled；
- 不启动大规模训练。

四张卡可以并行不同语言/语料的 embedding 与 RQ table 构建。

### Phase 1：LM pilot（约 1–2 天墙钟）

四卡各跑一个方法：

```text
GPU0 Arithmetic-matched
GPU1 RQ-Shuffled
GPU2 Semantic-RQ
GPU3 Mixed
```

- 100M tokens；
- 1 seed；
- 自动生成 overall 与 slice-level NLL；
- 过 Gate 1 才扩展。

### Phase 2：核心 LM 主结果（约 4–8 天墙钟，需实测修正）

- 约 1B tokens；
- 4 core methods × 3 seeds；
- 四卡并行，每轮四个 run；
- 同时完成 capacity 小/中/大三个关键点，完整六点 sweep 可后补。

### Phase 3：现有应用结果补控制组（约 2–4 天墙钟）

- XNLI/PAWS-X：补 RQ-Shuffled 和 3 seeds；
- Biomedical：0.6B 完整矩阵；
- KE：优先补标准化报告和 shared-row analysis，不盲目重跑全部 4 个模型尺度。

### Phase 4：Scaling verification

- Qwen3-1.7B 只复现最关键的 Arithmetic-matched、RQ-Shuffled、Semantic-RQ；
- 4B 仅在 0.6B/1.7B 结论稳定后做一组确认；
- 不在证据链未成立时投入 8B 多 seed。

---

## 14. 论文结构草案

### 1. Introduction

- Engram 通过有限表对大量 n-gram 进行条件记忆；
- 共享不可避免，但原始共享是随机的；
- 问题不是如何消除碰撞，而是如何使碰撞成为有用的 inductive bias；
- 提出 Semantic-RQ，并从效果、机制和边界三方面验证。

### 2. Background and Motivation

- Engram arithmetic multi-head hashing；
- finite table 下的参数共享；
- Engram-Nine 对“更少碰撞必然更好”的反例；
- structured collision hypothesis。

### 3. Method

- offline n-gram encoding；
- Residual Quantization；
- multi-head code-to-row mapping；
- frozen dictionary 与 OOV fallback；
- complexity、storage 和建表成本；
- Mixed variant。

### 4. Controlled Language Modeling

- iso-parameter setup；
- overall PPL；
- semantic-neighbor、long-tail、OOV 切片；
- capacity scaling。

### 5. Why Does Semantic Hashing Work?

- RQ-Shuffled；
- shared-row exposure；
- code/head intervention；
- RQ level analysis；
- false sharing。

### 6. External Validity

- XNLI/PAWS-X；
- Biomedical adaptation；
- knowledge update。

### 7. Efficiency and Limitations

- offline construction cost；
- runtime O(1) lookup；
- coverage/fallback；
- external encoder dependence；
- polysemy、temporal knowledge 和跨-backbone限制。

### 8. Conclusion

- 只总结被实验支持的范围；
- 不将 address portability 外推为 value portability。

---

## 15. 预计主表与主图

### Table 1：Controlled LM

Overall PPL + exact-seen/semantic-neighbor/long-tail/OOV NLL。

### Table 2：External Validity

XNLI、PAWS-X、Biomedical、KE paraphrase/locality。

### Figure 1：方法图

Arithmetic random sharing vs Semantic-RQ multi-code sharing。

### Figure 2：关键机制图

Target gain vs shared-row exposure，包含 Arithmetic、RQ-Shuffled、Semantic-RQ。

### Figure 3：Memory capacity curve

不同 memory budget 下的 overall 与 semantic-neighbor 性能。

### Figure 4：Transfer–Interference frontier

Positive transfer 对 false transfer，比较纯 Semantic、Mixed 和 Arithmetic。

---

## 16. 当前资产如何处理

### 可直接复用

- Arithmetic、RQ、Mixed/Mixed-v2 实现；
- Qwen3-Embedding + FAISS RQ 建表流程；
- CounterFact、ZsRE、MQuAKE、WikiCF、WikiRecent 结果；
- Qwen3 0.6B/1.7B/4B/8B 多尺度 KE 结果；
- XNLI/PAWS-X runner、dashboard 和已有结果；
- Biomed-Enriched 数据链路。

### 必须新增

- 严格 RQ-Shuffled；
- LM token slicing 与 shared-row exposure 日志；
- train/test 去重和 address-table provenance manifest；
- Protocol A：冻结共享接口、table-only update；
- capacity sweep；
- false-sharing dataset/切片；
- paired bootstrap 与多 seed 汇总。

### 降级或移出主文

- PopQA；
- 没有可靠 paraphrase/locality 定义的 MQuAKE 指标；
- 单 seed 的小幅平均分差；
- 未验证的跨 tokenizer/跨 backbone memory portability；
- CPU/SSD offload 作为 Semantic-RQ 创新点的叙述。

---

## 17. 最终投稿门槛

在写完整论文前，至少应满足：

- [ ] Arithmetic-matched、RQ-Shuffled、Semantic-RQ 完全公平；
- [ ] 一个受控 LM 设置完成 3 seeds；
- [ ] 一个真实 LLM post-hoc 设置复现；
- [ ] Semantic-neighbor 主终点显著；
- [ ] shared-row exposure 与 target gain 的机制证据成立；
- [ ] capacity sweep 支持 finite-memory hypothesis；
- [ ] false-sharing 风险被量化，Mixed 或 gating 能提供合理边界；
- [ ] 至少两个外部场景与核心机制方向一致；
- [ ] 所有地址表来源、coverage、fallback 和建表成本可复现；
- [ ] 论文没有把 knowledge editing、offload 或 portability 写成未经验证的核心贡献。

如果前六项不能同时满足，这个方向目前不够支撑一篇强方法论文；应当先 pivot，而不是继续堆更多 downstream benchmark。
