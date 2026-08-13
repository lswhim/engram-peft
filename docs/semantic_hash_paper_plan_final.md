# Semantic Hash for Engram：以语义寻址的迁移—干扰边界为主线

> 版本：v3（2026-08-13）
>
> 资源：4 × NVIDIA A100 40GB
>
> 状态：旧 RQ 使用静态 100k 精确表、OOV arithmetic fallback，且 FAISS packed code 解码错误；所有旧 RQ 数字只用于提出假设，不进入论文结果。

## 0. 先把问题说清楚

普通 FineWeb CPT 主要测“Engram 能否降低平均 next-token loss”。它不能直接回答 Semantic Hash
是否有用，因为同一 n-gram 在训练和测试中可以直接命中已训练地址，Arithmetic 也能完成记忆。

Semantic Hash 真正独有的能力应出现在：

```text
训练写入：事实/关系的一种表述 x
测试读取：词面不同、语义等价的表述 x'
要求：x' 通过相近 RQ 地址复用 x 已写入的 memory values
同时：无关查询 z 不应因共享地址而被误伤
```

因此论文研究问题改为：

> 在固定容量的 Engram 外部记忆中，Semantic-RQ 能否提高从已写入表达向未见释义、别名和关系模板的迁移，并量化这种结构化共享造成的 false transfer？

这不是把 Engram 重新包装成通用知识编辑算法。知识编辑 benchmark 是测量“写入后如何寻址读取”最直接、最标准的实验载体。

---

## 1. 从已有实验能提出什么假设

旧结果不可作为最终证据，但其形状可用于预注册假设：

- CounterFact：旧 RQ 在 0.6B/1.7B/4B/8B 的 efficacy 与 paraphrase 多数高于 Arithmetic，但 specificity 多数更低；
- ZsRE：efficacy 基本饱和/持平，旧 RQ 的 specificity 在四个尺度均高于 Arithmetic；
- WikiCF/WikiRecent：部分尺度上旧 RQ 提高 specificity，但不是全面领先；
- MQuAKE 当前 loader 的 paraphrase=0、specificity=1 是构造值，不能用于方法判断；
- PAWS-X：旧 matched Arithmetic macro 83.86%，旧 RQ 83.54%，没有正向结果；且旧 RQ 实现错误，两者差异不可解释；
- XNLI：两 seed 差值变号，不能支持跨语言 claim。

由此得到待验证而非既成结论的假设：

> Semantic-RQ 的收益不是平均 LM 能力，而是 controllable sharing：更容易把一次写入迁移到语义等价表达；代价可能是近邻之间相互污染。

---

## 2. 核心实验：标准知识编辑 benchmark

### 2.1 Benchmark

主表只使用有标准 efficacy / paraphrase / locality 定义的任务：

| Benchmark | 主要作用 | 主指标 |
|---|---|---|
| CounterFact | 反事实写入与 paraphrase/generalization | Efficacy、Paraphrase、Neighborhood/Specificity |
| ZsRE | QA 事实写入与多问法泛化 | Efficacy、Paraphrase、Specificity |
| WikiRecent / KnowEdit-WikiRecent | 新知识持续写入 | Efficacy、Paraphrase、Locality |
| MQuAKE | 多跳组合泛化 | 仅在修复标准多跳评测后进入；当前 loader 结果排除 |

PAWS-X/XNLI 只做外部泛化附录，不承担主 claim。普通 FineWeb/WikiText PPL 只作为 retention
和训练 sanity，不再先花三 seed × 12k steps 建主 checkpoint。

### 2.2 方法对照

| 方法 | 回答的问题 |
|---|---|
| Base | 未写入前能力 |
| LoRA | 写入共享神经参数的标准 PEFT 对照 |
| Arithmetic-fixed | 相同容量、无语义结构的精确/随机寻址 |
| RQ-Shuffled | 保留 code 频率和容量，破坏语义—地址对应关系 |
| Semantic-RQ | 修正后的动态 encode → frozen RQ → cache |

所有 Engram 方法严格匹配目标层、heads、每头 rows、embedding dim、可训练参数、训练 token、
optimizer steps 和 checkpoint selection。RQ 对任何未见 n-gram 都必须现场编码并 cache，fallback rate 必须为 0。

### 2.3 公平的写入协议

每个 edit batch 只训练 canonical prompt + target answer，不把测试 paraphrase 放入训练。Backbone 冻结，
只更新 Engram memory/gate；LoRA 使用相同监督 token 与梯度步数。使用官方 train/eval 字段和数据切分，
不自行生成主测试集。

至少报告三种 batch size：

```text
sequential / batch size ∈ {1, 10, 100}
```

它们分别测单次写入、少量并发写入和 memory collision/interference。先用 0.6B 或 1.7B 做
三 seed；只有结果通过 Gate A 才扩到 4B/8B。

### 2.4 论文不能只报 E/P/S

必须同时报告：

- Edit Success / Efficacy：canonical prompt 是否写入成功；
- Paraphrase Generalization：未训练释义能否读出同一 target；
- Locality / Specificity：无关样本输出是否保持；
- Harmonic score：防止只靠牺牲 locality 换 paraphrase；
- Sequential retention：后续 edits 后，早期 edits 还剩多少；
- time / peak GPU memory / persistent cache size / lookup latency。

### Gate A：Semantic Hash 是否真的 work

Semantic-RQ 必须同时满足：

1. 在至少 CounterFact 和 ZsRE 两个标准 benchmark 上，paraphrase 显著优于 Arithmetic-fixed 与 RQ-Shuffled；
2. canonical efficacy 不下降，或下降小于预注册容忍区间；
3. locality 的损失没有抵消 paraphrase 收益，harmonic score 仍领先；
4. 三 seed 方向一致，paired bootstrap 95% CI 至少一个核心比较排除 0；
5. 动态 RQ 覆盖率 100%，没有 Arithmetic fallback。

不满足时，主方法结论就是负结果，不能用地址 overlap 挽救。

---

## 3. 真正体现地址几何的标准切片

这些是 benchmark 内分析，不另造 benchmark：

### 3.1 表面差异分桶

对官方 paraphrase 按 canonical↔paraphrase 的 token Jaccard / edit distance 分桶。核心预期是：

- exact/高词面重合：Arithmetic 与 Semantic-RQ 接近；
- 低词面重合但 embedding 相似：Semantic-RQ 优势最大；
- 低语义相似：不应发生正迁移。

### 3.2 地址复用链路

逐样本记录：

```text
embedding cosine
→ RQ code overlap（0…M heads）
→ 与训练 prompt 共用且被更新的 rows 数量
→ paraphrase gain / locality damage
```

RQ-Shuffled 是关键反事实：如果它与 Semantic-RQ 表现相同，收益来自访问频率或容量，而不是语义几何。

### 3.3 干预而非相关性

对产生 paraphrase gain 的共享 rows 做 reset/mask：若收益随共享 head 数单调消失，才支持因果解释。
同时对语义相近但答案不同的 counterfactual neighbor 测 false transfer。

---

## 4. 第二主实验：持续写入与可迁移性

按时间或随机顺序连续写入 WikiRecent/CounterFact edits，评估每 10/100/500 次写入后的：

- 当前 edit efficacy；
- 旧 edit retention；
- paraphrase retention；
- unrelated locality；
- memory rows touched、collision rate 与写入吞吐。

这个实验体现 Engram 相对 LoRA 的结构优势：知识值位于可寻址 memory，写入稀疏且可 offload；
Semantic Hash 的额外贡献只限定为“让未见语义等价查询能够复用已写入 rows”。不能把 offload 本身归功于 Semantic Hash。

---

## 5. 多语言实验的正确版本

英语训练 → PAWS-X/XNLI 测试并不能自动证明 memory value 跨语言迁移，因为不同语言 token pattern
不同。若做多语言，只使用同一事实的平行查询（例如 multilingual ParaRel/ZsRE/CounterFact 资源）：

```text
English canonical fact write
→ English paraphrase read
→ target-language parallel paraphrase read
```

需要同时证明目标语言查询确实复用了英语训练时更新的 rows。若跨语言 embedding 相似但 code overlap
仍低，或 overlap 与 accuracy gain 无关，就不能 claim multilingual portability。

该实验在 Gate A 通过后再跑，不作为救场实验。

---

## 6. 四张 A100 的新执行顺序

### Phase 0：实现与审计（先完成）

- 修正后的动态 RQ：所有新 n-gram 在线 encode，首次访问 cache，fallback=0；
- 使用 FAISS 官方 unpack/compute_codes；
- 给 KE runner 增加 Arithmetic-fixed、RQ-Shuffled、Semantic-RQ；
- 保存逐样本 canonical/paraphrase/locality 结果与 row-access trace；
- 用 50 个 edits 做 correctness integration test，不作为论文结果。

### Phase 1：0.6B/1.7B 标准主实验

四卡并行方法，而不是三卡重复同一个 LM run：

```text
GPU0  Arithmetic-fixed
GPU1  RQ-Shuffled
GPU2  Semantic-RQ
GPU3  LoRA / Base evaluation
```

先 CounterFact seed42，再 ZsRE seed42。看到完整 E/P/locality 与逐样本结果后执行 Gate A；通过后补
seed43/44，未通过则停止扩模型。

### Phase 2：持续写入

只对 Gate A 中最强的两个 Engram 方法与 LoRA 跑 sequential edits，测 retention 和 interference 曲线。

### Phase 3：规模与多语言

只有 Phase 1/2 正向才扩 4B/8B，并选择一个平行事实 benchmark 验证跨语言 row reuse。

---

## 7. 当前有效结论与无效结论

### 当前有效

- 修正后的 lazy RQ 能对任意新 n-gram 计算 8-level code 并立即用于当前 forward，cache 后可复用；
- 旧静态表 + fallback 实现没有测试我们真正提出的 semantic generalization；
- 普通 LM-CPT 不是验证 Semantic Hash 优势的高辨识度主任务；
- 旧结果仅提示可能存在 paraphrase–specificity trade-off，需要标准重跑。

### 当前尚不能声称

- Semantic-RQ 优于 Arithmetic；
- Semantic-RQ 提高多语言泛化；
- Semantic-RQ 提高通用 LM benchmark；
- Semantic-RQ 保持 locality 或减少遗忘；
- 旧知识编辑、PAWS-X、XNLI 的任何 RQ 数字是正式结果。

论文是否成立，取决于修正实现下的标准 KE 主表与 transfer–interference 曲线。
