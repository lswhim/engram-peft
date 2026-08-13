# Semantic Hash for Engram：Benchmark-First 论文与实验规划

> 版本：v2（完全重构）
>
> 日期：2026-08-13
>
> 资源：4 × NVIDIA A100 40GB
>
> 原则：先用公开 benchmark 判定方法是否有效；只有结果正向，才做机制解释。

## 0. 一句话研究问题

> 在相同 backbone、训练数据、训练预算和 Engram 容量下，Semantic-RQ 是否在标准公开
> benchmark 上稳定优于随机 Arithmetic hash 和破坏语义对应关系的 RQ-Shuffled？

论文首先回答“有没有用”，而不是先构造一套只能由我们解释的泛化切片。

地址相关性、shared-row exposure、masking 等分析不能代替 benchmark 收益。若公开 benchmark
不提升，即使地址几何非常漂亮，也只能得出“模型没有有效利用该几何”的负结论。

---

## 1. 唯一主 Claim 与判死规则

### 唯一主 Claim

> Under matched finite memory capacity, semantic hashing improves language modeling or
> paraphrase generalization over both random hashing and a frequency-matched shuffled-RQ control.

### 立即判死

以下两条都不成立时，停止扩规模和机制包装：

1. Semantic-RQ 在标准 LM benchmark 上稳定优于 Arithmetic-fixed 与 RQ-Shuffled；
2. Semantic-RQ 在 QQP → PAWS-QQP 分布外泛化上稳定提升，且 QQP in-domain 不退化。

不能用以下结果挽救失败的主 claim：

- 地址 overlap 或相关性很高；
- 某个自建 slice 上领先；
- 单 seed 或单 benchmark 小于 0.2 pp 的波动；
- 不公平的旧 Arithmetic baseline；
- XNLI 平均分基本持平。

---

## 2. 方法与公平基线

所有主结果必须使用相同 backbone、训练数据、训练步数、目标层、embedding 维度和可训练参数量。

| 方法 | 作用 | 是否进入主表 |
|---|---|---|
| Base / No-memory | 判断 Engram 本身是否有效 | 是 |
| Arithmetic-fixed | 同容量随机 hash 基线 | 是 |
| RQ-Shuffled-frequency-matched | 保留 RQ 容量、code/访问频率，破坏语义映射 | 是 |
| Semantic-RQ | 主方法 | 是 |
| Mixed | 仅当主方法有效但出现明显副作用时研究 | 可选 |

严格容量配置：

- Semantic-RQ：M=8、K=256；
- Arithmetic-fixed：8 heads × 256 buckets；
- 每种 n-gram order 均为 2,048 rows；
- 当前实现两者可训练参数均为 26,984,448；
- 旧 Arithmetic、旧 matched 和 matched-v2 全部标记为 invalid，不进入任何正式比较。

---

## 3. 第一优先级：标准语言建模 Benchmark

### 3.1 训练协议

先完成当前已经运行的严格 matched checkpoint：

- Backbone：Qwen3-1.7B-Base；
- 冻结 backbone，只训练 Engram 模块；
- FineWeb-Edu train/eval 严格隔离；
- context length 256；
- 12,208 optimizer steps；
- seeds 42、43、44；
- Arithmetic-fixed、RQ-Shuffled、Semantic-RQ；
- 当前 repeated-data run 只作为第一轮 checkpoint；Mixed 不承担主结论。

当前训练是 48,832 条序列重复 8 epoch，等价于 100,007,936 processed token slots，不能写成
100M unique-token 预训练。若第一轮 benchmark 正向，再用约 390,656 条独立序列做同 step 的
近 one-pass 复现。

### 3.2 公开评测集

| Benchmark | 主指标 | 用途 |
|---|---:|---|
| FineWeb-Edu held-out | token PPL/NLL ↓ | 同域语言建模 |
| WikiText-103 | word PPL ↓ | 标准文本 LM |
| C4 validation / Paloma C4-en | word PPL ↓ | 宽域迁移 |
| LAMBADA OpenAI | accuracy ↑、PPL ↓ | 上下文词预测 |
| Paloma domain aggregates | PPL ↓ | 跨域稳健性 |

不得只挑有利 corpus。所有数据版本、split、tokenization、word-normalization 和样本数在运行前固定。

### 3.3 统计与通过条件

- 报告 3-seed mean ± std；
- 在相同 evaluation examples/tokens 上计算 paired bootstrap 95% CI；
- Semantic-RQ 必须同时比较 Arithmetic-fixed 和 RQ-Shuffled；
- 主指标方向必须至少在两个公开 LM benchmark 上一致；
- FineWeb held-out 不得显著恶化；
- 若优势只有一个 seed、一个 corpus 或低于正常 seed 波动，则判为持平。

### Benchmark Gate A

满足以下条件才进入大规模 one-pass replication：

1. Semantic-RQ 对两个核心基线至少在两个标准 LM benchmark 上有一致优势；
2. 至少一个关键比较的 paired 95% CI 排除 0；
3. 三个 seed 没有由单个异常 seed 驱动；
4. 吞吐、显存或 coverage 没有不可接受的退化。

若 Gate A 不通过，停止 LM 扩规模；保留结果作为负结论。

---

## 4. 第二优先级：标准语义泛化 Benchmark

不再以自建 semantic-neighbor slice 作为首要证据。使用公开 paraphrase benchmark。

### 4.1 主协议：QQP → PAWS-QQP

```text
QQP train
  → 训练相同 backbone + 不同 Engram addressing
  → QQP validation/test（in-domain）
  → PAWS-QQP test（zero target update，OOD）
```

它直接检验：在普通同义问句上学习后，能否泛化到具有高词面重叠、但标签更困难的 PAWS 样本。

### 4.2 辅助公开任务

| 训练 → 测试 | 指标 | 地位 |
|---|---:|---|
| QQP → QQP | Accuracy/F1 | in-domain 必报 |
| QQP → PAWS-QQP | Accuracy/F1/AUROC | 主要 OOD endpoint |
| PAWS-Wiki train → PAWS-Wiki test | Accuracy | 复现性辅助 |
| MRPC train → MRPC test | Accuracy/F1 | 小样本辅助，不单独支撑结论 |

### 4.3 方法和控制

- Base、Arithmetic-fixed、RQ-Shuffled、Semantic-RQ；
- 3 seeds；
- 同一 classifier head、训练步数、batch、学习率与 early-stopping 规则；
- 不能选择每个方法各自最有利的 checkpoint；
- RQ table 和 shuffle 必须只使用训练 split 统计，禁止 test leakage。

### Benchmark Gate B

Semantic-RQ 只有在以下条件同时满足时，才能 claim 语义泛化：

1. QQP → PAWS-QQP 优于 Arithmetic-fixed 和 RQ-Shuffled；
2. 3-seed CI 或预注册统计检验支持该差异；
3. QQP in-domain 没有显著退化；
4. PAWS-Wiki 或另一公开 paraphrase 设置方向复现。

若只在 QQP in-domain 提升，而 PAWS 不提升，不能称为“地址几何泛化”。

---

## 5. 第三优先级：有限容量是否是关键变量

只有 Gate A 或 Gate B 至少一个通过后才运行。

### 设置

```text
K ∈ {64, 256, 1024}
methods ∈ {Arithmetic-fixed, RQ-Shuffled, Semantic-RQ}
```

先用 seed42 跑标准 LM 与 QQP → PAWS-QQP 中已确认最敏感的 endpoint；仅在曲线呈现明确趋势时，
补 seed43/44。

### 可支持的结论

- 小容量时 Semantic-RQ 相对优势更大，容量增大后差距收敛：支持 structured sharing；
- 所有容量均持平：有限容量叙事不成立；
- 大容量反而更好：需要重新解释，不能继续声称优势来自缓解随机碰撞；
- Semantic-RQ 提升 PAWS 但损害 in-domain：报告 transfer–interference trade-off，不包装为全面提升。

---

## 6. 第四优先级：机制解释（只解释已观察到的 Benchmark 收益）

机制实验不是 benchmark 的替代品。只有标准任务出现稳定收益后才做，并且问题由具体结果决定。

### 最小机制包

1. **地址几何 audit**：Semantic-RQ 与 RQ-Shuffled 的语义相似度—code overlap 相关性；
2. **trained-row reuse**：测试样本访问的 rows 中，训练期被更新过的比例；
3. **shared-row intervention**：mask/reset 实际共享 rows 后，benchmark 收益是否消失；
4. **error slices**：只在公开 benchmark 的预定义子集上分析，不另造一个主 benchmark。

因果链必须是：

```text
公开 benchmark 提升
  → 提升样本有更高 trained-row reuse
  → RQ-Shuffled 不具备同样关系
  → mask/reset shared rows 后提升显著下降
```

如果 benchmark 不提升，就不再做大规模 masking、token slice 或人为构造的“泛化”集合。

---

## 7. XNLI、PAWS-X 与跨语言实验的位置

### 当前 XNLI 的正式判断

| 方法 | Seed 42 | Seed 43 | 两 seed均值 |
|---|---:|---:|---:|
| Arithmetic-fixed | 75.199% | 75.525% | 75.362% |
| Semantic-RQ | 75.412% | 75.477% | 75.444% |
| Semantic − Arithmetic | +0.213 pp | −0.048 pp | +0.083 pp |

逐 seed 差值变号，因此当前结论是持平，不是提升。

XNLI 降为附录 sanity check，原因是英语训练和其他语言测试的 token/n-gram pattern 不一致；在没有
跨语言 trained-row reuse 证据前，平均 accuracy 不能证明 Engram value 被迁移。

PAWS-X 同样不承担主 claim。若未来保留跨语言章节，必须先报告平行句地址重叠和英语训练 row 的
目标语言复用率，再解释 accuracy。

---

## 8. 论文主表与主图

### Table 1：严格公平性与效率

参数量、rows/head、table coverage、fallback rate、build cost、storage、吞吐、峰值显存。

### Table 2：标准 LM Benchmark（第一主表）

Base、Arithmetic-fixed、RQ-Shuffled、Semantic-RQ 的 FineWeb、WikiText-103、C4/Paloma、LAMBADA；
3-seed mean ± std 与 paired CI。

### Table 3：公开 Paraphrase Generalization（第二主表）

QQP in-domain、QQP → PAWS-QQP、PAWS-Wiki、MRPC。

### Figure 1：容量曲线

仅在 benchmark 正向后展示 K={64,256,1024} 的实际 benchmark 指标。

### Figure 2：因果解释

trained-row reuse 与 benchmark gain，以及 shared-row masking 曲线。

地址相关性表放机制章节或附录，不再作为论文第一张结果表。

---

## 9. 四张 A100 的执行顺序

### Phase 1：完成当前 matched checkpoints

- 跑完 Arithmetic-fixed、RQ-Shuffled、Semantic-RQ × seeds 42/43/44；
- Mixed 已启动的 run 可以完成，但不再优先补齐所有扩展实验；
- 旧 early-stop 和错误容量 baseline 永久排除。

### Phase 2：立即跑公开 benchmark

- GPU0：FineWeb held-out + WikiText-103；
- GPU1：C4/Paloma；
- GPU2：LAMBADA；
- GPU3：结果校验、复跑失败项或开始 QQP checkpoint。

评测完成后立即执行 Gate A，不先跑自建 slice。

### Phase 3：QQP → PAWS

四方法 × 3 seeds 动态排队；QQP 与 PAWS 使用相同 checkpoint。先完成完整三角对照，再决定是否
加入 PAWS-Wiki/MRPC。

### Phase 4：条件执行

- Gate A/B 至少一个通过：one-pass replication → capacity → 最小机制包；
- 两个 Gate 均失败：停止扩实验，整理负结果与资源分析；
- XNLI/PAWS-X 不得阻塞主 benchmark。

---

## 10. 投稿硬门槛

- [ ] 标准 LM 或 QQP → PAWS 至少一条主线稳定优于两个核心基线；
- [ ] 所有主比较 3 seeds，报告 mean/std 和预注册 CI；
- [ ] RQ-Shuffled 参数、容量和训练访问频率匹配；
- [ ] Base / No-memory 证明新增 Engram 是否真正有用；
- [ ] 公开任务版本与评测协议可复现；
- [ ] one-pass 独立文本 setting 复现主要方向；
- [ ] capacity curve 与有限 memory 解释一致；
- [ ] benchmark 正向后，最小干预支持 shared-row 因果链；
- [ ] 报告所有负结果，不选择性隐藏 XNLI 持平或 seed 变号；
- [ ] 不把地址结构本身写成下游收益。

若第一条不满足，这不是一篇有效的方法论文。此时最诚实的结论是：Semantic-RQ 构造了语义地址
几何，但该几何没有转化为公开 benchmark 收益。

---

## 11. 论文结构

1. **Introduction**：有限 conditional memory 中，语义组织地址是否带来实际 benchmark 收益；
2. **Method**：Semantic-RQ、冻结离线建表和推理 lookup；
3. **Fair Comparison Protocol**：Base、Arithmetic-fixed、RQ-Shuffled；
4. **Standard LM Benchmarks**：第一主结果；
5. **Paraphrase Generalization**：QQP → PAWS；
6. **Capacity and Mechanism**：仅解释已经成立的 benchmark 现象；
7. **Efficiency and Limitations**：coverage、build/storage、false sharing；
8. **Appendix**：XNLI/PAWS-X、地址几何、失败 baseline 和额外切片。

## 12. Reviewer 式最终判断

这篇论文不再以“我们构造了一个语义泛化切片”为中心，而以公开 benchmark 的可复现收益为中心。

最短证据链是：

```text
公平 matched baseline
  → 标准 LM 或 QQP→PAWS 显著提升
  → one-pass 复现
  → 小容量优势更明显
  → shared-row 干预使提升消失
```

第一箭头不成立就停止；成立后才有必要解释地址几何为什么有效。
