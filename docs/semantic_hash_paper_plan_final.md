# Semantic-Addressed Engram：多语言、可扩展的条件知识记忆

> 版本：v4（2026-08-13）
>
> 目标会议：ICML 2026
>
> 资源上限：4 × NVIDIA A100 40GB
>
> 原则：先证明 Engram memory 的独特价值，再证明 Semantic Hash 相对 Arithmetic addressing 的增量价值。

## 0. 最终研究定位

### 一句话问题

> Can a semantically addressed conditional memory write knowledge in one language and retrieve it in unseen languages, while preserving the frozen backbone and retaining Engram's sparse, scalable memory properties?

### 一句话方法

冻结多语言 LLM backbone，只向 Engram memory 写入新事实；查询 n-gram 经过冻结多语言 embedding encoder
和冻结 RQ codebook 得到多头地址，首次计算后缓存，使不同语言、不同表述的语义等价查询能够部分共享已写入
memory rows。

### 为什么这个定位比普通 LM / PAWS-X 合理

原版 Engram 的核心不是“小参数 adapter”，而是把静态知识和局部模式从神经计算中分离到可扩展、确定性寻址、
可 offload 的条件记忆。Semantic Hash 的新增价值也不是减少平均碰撞，而是让不同表面形式可以共享一次写入。

因此论文需要同时成立两件事：

1. **Engram advantage**：冻结 backbone、稀疏写 memory，在连续知识注入中比 LoRA/FT 更少遗忘，写入成本随
   memory 容量近似不变，并可独立保存、替换和 offload；
2. **Semantic Hash advantage**：在相同 Engram 容量和训练预算下，源语言写入能够迁移到未参与训练的语言和
   paraphrase，优于 Arithmetic 与破坏语义对应的 RQ-Shuffled。

不能只证明第二件事，也不能把 Engram 本来就具备的 offload 说成 Semantic Hash 的贡献。

---

## 1. 文献实验范式给我们的约束

### 原版 Engram

原论文通过以下证据建立 conditional memory：

- iso-parameter / iso-FLOPs 比较，而不是只和裸 Base 比；
- memory slot 数量扩展曲线，证明容量可在不增加激活计算的情况下增长；
- Pile loss、知识、推理、阅读理解、代码/数学的全面 benchmark；
- 移除/缩放 Engram 后，事实知识比阅读理解下降更严重，证明 memory 功能分工；
- long-context NIAH、early-layer representation、attention 分工和 gate 可视化；
- 100B 参数表从 host memory prefetch 的吞吐实验。

### Engram-Nine

它严格匹配参数后发现 collision-free hot tier 不稳定优于 hash，并观察到 gate credit assignment mismatch。
因此我们的论文不能把“语义地址更合理”直接等同于“最终模型一定更好”；必须有 RQ-Shuffled、gate 使用和
shared-row intervention。

### Memory Grafting

它把大模型 hidden states 离线构造成 frozen memory values，比较 MoE、vanilla Engram 和 grafted memory，
并报告 grafting source layer、recipient compatibility、memory 大小和构建/推理成本。这说明后续工作必须明确
区分 address quality、value quality 和 gate compatibility。

### 对本工作的直接含义

我们只修改 **address geometry**，value 仍由目标任务训练。因此主实验必须控制：

```text
相同 memory rows / embedding dim / gate / target layers / optimizer / updates
唯一差别 = Arithmetic vs RQ-Shuffled vs Semantic-RQ addressing
```

如果 value 学习失败或 gate 不使用 memory，再好的跨语言地址也不会产生收益。

---

## 2. 论文核心任务：跨语言知识写入，而不是跨语言分类

### 2.1 主 benchmark

| Benchmark | 角色 | 为什么合适 |
|---|---|---|
| **BabelEdits** | 第一主表 | 60 种语言、实体 aliases 质量高，同时测跨语言 edit effectiveness 与模型 robustness |
| **MzsRE / Bi-ZsRE** | 第二主表 | 标准 multilingual factual editing；同一事实有跨语言问法 |
| **MLaKE single-hop** | 复现表 | 英/中/日/法/德五语言，标准 single-hop 与 cross-lingual evaluation |
| **WikiBigEdit** | 持续写入表 | ICML 2025 标准 lifelong knowledge editing，测大量真实更新与遗忘 |

CounterFact/ZsRE 单语版用于验证 canonical→paraphrase，放辅助主表。PAWS-X 与 XNLI 降为附录 task-transfer
sanity，不用于证明 knowledge memory。

### 2.2 核心协议

对每个事实提供 source language `L_src` 的 canonical prompt 和 target answer：

```text
只训练 L_src canonical prompt → new target
不训练任何 paraphrase 或其他语言

测试：
1. L_src canonical                edit efficacy
2. L_src unseen paraphrase        monolingual generalization
3. L_tgt parallel query           cross-lingual propagation
4. L_tgt entity aliases           alias robustness
5. unrelated facts/tasks          locality / model retention
```

主要设置：English→{de, es, fr, zh, ja, ko}；BabelEdits 再报告语言族和资源水平 aggregate。补一个
非英语 source（zh→English/others），避免把“English encoder hubness”误认为通用跨语言能力。

---

## 3. 方法与公平基线

### 3.1 必须进入主表

| 方法 | 用途 |
|---|---|
| Frozen Base | 写入前下限与 locality reference |
| Full FT | 高干扰写入上界/成本对照，小规模报告 |
| LoRA | 标准 PEFT 写入对照 |
| Arithmetic-fixed Engram | 原始离散寻址；严格同容量 |
| RQ-Shuffled Engram | 保持 RQ code 频率与容量，破坏语义映射 |
| Semantic-RQ Engram | 我们的方法；动态 encode + frozen RQ + persistent cache |

若工程允许，可加入一个成熟 KE 方法（ROME/MEMIT 或 EasyEdit 中对 Qwen 兼容的方法）作为任务领域基线；
但它不替代 Arithmetic 和 RQ-Shuffled 两个方法因果基线。

### 3.2 公平约束

- 同一 Qwen3-1.7B-Base；Gate A 后复现 Qwen3-4B；
- backbone 对所有 Engram 方法完全冻结；
- M=8、K=256 起步，2/3-gram、相同 target layers 和 embedding dim；
- 相同可训练参数、batch、监督 token、optimizer steps、学习率搜索空间；
- checkpoint 选择只看 source-language validation，不看目标语言；
- RQ unseen n-gram 必须动态编码并立刻用于当前 forward，fallback=0；
- 每种方法三个 seeds；逐事实 paired evaluation。

LoRA 参数量很难与 Engram 完全相同，因此同时报告：trainable parameters、实际 touched parameters、optimizer
state、写入时间和峰值显存，不能只写“参数高效”。

---

## 4. Table 1：跨语言知识迁移（Semantic Hash 的主结果）

每个 benchmark 报告：

- Source efficacy / reliability；
- Source paraphrase；
- Target-language propagation macro；
- Target alias robustness；
- Locality / specificity；
- Harmonic mean：防止用 locality 换 propagation；
- Language consistency：同一事实在多语言答案是否一致。

### Gate A：方法是否成立

Semantic-RQ 必须满足：

1. Source efficacy 不低于 Arithmetic-fixed；
2. target-language propagation 同时优于 Arithmetic-fixed 和 RQ-Shuffled；
3. locality 下降不超过预注册阈值，harmonic mean 仍领先；
4. 至少 BabelEdits 与 MzsRE 两个 benchmark 方向一致；
5. 三 seed paired bootstrap 95% CI 至少一个主比较排除 0；
6. 不得只靠中文或只靠与英语相近的语言拉高 macro。

若不通过，停止规模扩展。地址 overlap 很高不能挽救效果失败。

---

## 5. Table 2：持续写入（Engram 本体优势）

使用 WikiBigEdit 或标准可复现的 sequential CounterFact protocol：

```text
连续写入 N ∈ {1, 10, 100, 1k, 10k} facts
每个阶段评估新事实、历史事实、paraphrase、跨语言查询和无关能力
```

比较 LoRA、Arithmetic Engram、Semantic-RQ Engram：

- current edit success；
- previous-edit retention / forgetting curve；
- cross-lingual retention；
- locality；
- 每个 edit 实际更新 rows 数；
- 写入 wall time、峰值 GPU memory、optimizer state；
- memory 增长和 cache storage。

这里应该体现 Engram：backbone 不改、写入集中在离散 memory rows、旧知识受干扰范围受地址共享控制。
Semantic-RQ 可能提高 transfer，也可能因共享增加 interference；论文必须诚实展示 Pareto frontier。

---

## 6. Figure 1：容量—迁移—干扰曲线

只在 Gate A 通过后运行：

```text
K ∈ {64, 256, 1024, 4096}
M 固定为 8
methods = Arithmetic-fixed / RQ-Shuffled / Semantic-RQ
```

横轴 memory rows 或持久存储，纵轴同时画：

- cross-lingual propagation；
- locality；
- harmonic mean；
- sequential retention；
- 吞吐/首次 miss latency/热 cache latency。

合理预期不是 Semantic-RQ 永远胜出，而是小容量/长尾时结构化共享提高 transfer；容量扩大后随机碰撞减少，
Arithmetic 差距缩小；Semantic-RQ 的 false sharing 可能形成 locality 上限。

这比单点 M8/K256 更能解释方法何时 work。

---

## 7. Figure 2：机制因果链

### 7.1 逐事实地址复用

对 source prompt 与 target-language parallel prompt 记录：

- embedding cosine；
- 每个 n-gram order、每个 RQ level 的 code overlap；
- target query 访问 source 训练期间实际更新 rows 的比例；
- gate activation；
- target-language log-probability gain。

“全局 unique row 是否见过”会因小 K 饱和，不能单独使用。主分析必须是**同一事实配对**的 aligned n-gram
code overlap，以及访问频率/更新量加权 overlap。

### 7.2 三个必要干预

1. **RQ-Shuffled**：消除语义—地址对应，保留容量和访问频率；
2. **Shared-row reset**：只重置 source/target 共享且在 source 写入时更新过的 rows；
3. **Gate intervention**：固定/屏蔽 gate，判断地址几何是否被 gate 实际利用。

支持论文的完整因果链是：

```text
跨语言语义相似
→ 更高 aligned code overlap
→ 复用 source-updated rows
→ 更高 target propagation
→ shuffle 或 reset shared rows 后收益消失
```

若 gate 接近零或 shared-row reset 不影响收益，则准确率变化不能归因于 Semantic Engram。

---

## 8. Table 3：效率与 Engram 系统属性

四张 A100 无法复现 CXL 系统论文，但可以给出可信的单机证据：

- 冷启动：Qwen embedding + RQ 首次地址生成 latency；
- 热启动：SQLite/in-memory cache lookup latency；
- cache hit rate 随请求数变化；
- memory table 在 GPU、pinned CPU memory 下的 tokens/s 和 PCIe transfer；
- table size 从 10M 到可容纳上限的模拟/真实扩展；
- 相同 batch/sequence 下 Base、Arithmetic、Semantic cold、Semantic warm 的吞吐；
- persistent cache 大小和构建 GPU-hours。

必须明确：动态 Semantic Hash 的首次 miss 不再是原版 Engram 的纯 O(1) lookup；其系统 claim 是
**amortized lookup after memoization**。论文要报告冷/热两条路径，不能只报热 cache。

---

## 9. 当前 PAWS-X 的正确位置

正在运行的 PAWS-X English→7 languages 保留为 implementation diagnostic：

- 能快速验证 strict dynamic RQ 是否可训练；
- 能检查不同语言是否访问同一 rows；
- 但它写入的是 paraphrase 分类规则，不是新事实；
- 49k 训练样本可能覆盖几乎所有 M8/K256 rows，使全局 row reuse 饱和；
- 因此不进入论文主表，也不决定论文方向。

PAWS-X 完成后只回答“是否值得进入 BabelEdits pilot”，不能单独支撑跨语言 conditional memory claim。

---

## 10. 四卡执行计划

### Phase 0：当前 PAWS-X 诊断

- GPU0 Arithmetic-fixed；GPU1 Semantic-RQ；GPU2 LoRA；GPU3 保留非项目进程；
- 只跑 seed42；完成后不自动补三 seed；
- 检查 accuracy、逐语言 paired row reuse、gate 与 cold/warm cost。

### Phase 1：BabelEdits/MzsRE pilot

先 500–1,000 edits、seed42：

```text
GPU0 Arithmetic-fixed
GPU1 RQ-Shuffled
GPU2 Semantic-RQ
GPU3 LoRA 或成熟 KE baseline
```

必须跑完整 source efficacy、cross-lingual propagation、locality 和逐事实 trace。该 pilot 是 correctness +
effect-size estimation，不写论文最终数字。

### Phase 2：主表

若 pilot 中 Semantic-RQ 对两个 Engram 对照有至少 2 pp target macro 改善，且 locality 可接受：

- BabelEdits + MzsRE 全量；
- seeds 42/43/44；
- 单卡独立 run，四卡持续排队；
- paired bootstrap 和语言族 aggregate。

### Phase 3：持续写入

- WikiBigEdit N={10,100,1k,10k}；
- LoRA / Arithmetic / Semantic；
- 每阶段保存 retention、跨语言传播和资源曲线。

### Phase 4：规模、容量与系统

- 只复现最强设置到 Qwen3-4B；
- K sweep 先 seed42，再补显著点；
- CPU offload 与 cold/warm cache benchmark。

---

## 11. 论文主表和主图

1. **Table 1 — Cross-lingual Knowledge Write/Read**：BabelEdits、MzsRE；
2. **Table 2 — Lifelong Knowledge Injection**：WikiBigEdit retention/locality；
3. **Table 3 — Efficiency and Storage**：写入成本、touched rows、冷/热 latency、offload；
4. **Figure 1 — Capacity–Transfer–Interference Frontier**；
5. **Figure 2 — Paired Address Reuse and Causal Intervention**；
6. **Appendix**：PAWS-X、XNLI、单语 CounterFact/ZsRE、更多语言/seed。

---

## 12. 当前允许和禁止的结论

### 当前允许

- 修正后的 Semantic-RQ 能为未见 n-gram 动态生成 frozen RQ 地址并 cache；
- 旧 RQ 结果因 fallback 和 code 解码错误全部无效；
- 原版 Engram 的优势需要通过容量、知识存储、稀疏更新和系统属性体现；
- PAWS-X 只是诊断，跨语言事实写入 benchmark 才是方法主场。

### 当前禁止

- Semantic-RQ 已提高跨语言泛化；
- Semantic-RQ 已优于 Arithmetic；
- 动态 Semantic-RQ 保持原版 Engram 的严格零额外计算/O(1) 冷路径；
- Engram 的 offload、可扩展性是 Semantic Hash 新贡献；
- 单 seed PAWS-X 或旧知识编辑表可以作为论文结果。

最终 go/no-go 由 BabelEdits/MzsRE 的跨语言 propagation、locality 和 shared-row 因果干预共同决定。
# Formal benchmark expansion (2026-08-14)

The paper now uses three complementary official benchmark families rather than
adding more variants of CounterFact.  All methods share the same base model,
memory capacity, optimizer budget, edit order, and seeds.

1. **ParaRel-based causal address test.** Within every relation, targets are
   deterministically deranged so the evaluation cannot be solved from the base
   model's stored fact. One canonical template is used for memory writing and
   all other templates are evaluation-only. Arithmetic-fixed, Semantic-RQ, and
   frequency-matched RQ-Shuffled are compared. Queries are stratified by Qwen
   embedding cosine and character-trigram Jaccard; the primary slice is high
   semantic similarity with low lexical overlap.
2. **WikiBigEdit lifelong scale.** Official chronological increments are
   concatenated without future leakage. The same evolving adapter is evaluated
   after 1K, 5K, 10K, and 50K writes. Report current efficacy/generalization,
   fixed-cohort retention from every earlier increment, locality, and the area
   under the retention curve. This is a continuous write experiment, not four
   independently trained prefixes.
3. **RippleEdits sharing boundary.** Subject aliasing, logical generalization,
   and compositionality are `should_propagate`; relation specificity and
   forgetfulness are `should_not_propagate`. Official condition queries gate
   every testcase before it enters a denominator.

The canonical data conversion entry point is
`examples/semantic_memory_benchmarks.py`. Generated JSONL manifests retain the
source split/increment, relation, axis, aliases, and condition queries so every
reported denominator can be audited.
