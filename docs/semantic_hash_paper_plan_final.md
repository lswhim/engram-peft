# Semantic Hash for Engram：最终论文规划

> 版本：Reviewer-oriented v1
> 日期：2026-08-13
> 资源：4 × NVIDIA A100 40GB
> 当前阶段：受控语言建模 Gate 1 运行中

## 0. 先给结论：现在的实验方向对不对？

**方向是对的，但当前正在运行的实验只能算机制 pilot，单独不足以投稿。**

正确的部分是：主实验已经从知识编辑转回 Engram 的本职任务——语言建模，并且用
Arithmetic-matched、frequency-matched RQ-Shuffled 和逐 token 切片隔离
“语义结构化共享”本身的作用。

仍需纠正的部分是：当前 FineWeb 训练把 48,832 条序列重复 8 个 epoch。它等价于
100,007,936 个 **token slots/presentations**，但不是 100M 个不同训练 token。因此它适合
检验“训练过的 memory rows 能否迁移到未见语义邻居”，不能被写成标准的 100M-token
预训练结果。论文必须再补一个近似 one-pass、更多独立文本的 LM setting，并加入标准
held-out benchmark。

知识编辑不是主线。整篇论文的证据比例应大致为：

```text
受控语言建模与标准 LM benchmark       50%
共享机制、容量曲线与 false sharing    35%
跨语言或领域适配外部验证              15%
知识编辑                               附录/可选
```

---

## 1. 论文只回答一个问题

> 当有限 Engram 表迫使不同 n-gram 共享参数时，把随机共享改成语义结构化共享，能否提高
> 对词面未见但语义相关表达的泛化，同时不引入严重的错误共享？

原始 Arithmetic hash 在有限表中产生随机共享。Semantic-RQ 离线编码 n-gram，并用
Residual Quantization 生成多级离散地址，让相似 n-gram 共享部分 Engram rows。推理时只做
冻结表查询，不在线运行 embedding model。

核心因果链：

```text
语义相似
  → RQ code 部分重合
  → Engram row 部分共享
  → source n-gram 的梯度更新了 target 会访问的 row
  → exact-unseen target token 的 NLL 降低
```

### 唯一主 claim

> Under matched finite memory capacity, semantic organization of shared rows improves
> generalization to lexically novel semantic neighbors over random sharing.

### 明确不 claim

- 不说 Semantic-RQ “消除了碰撞”或“索引更准确”；
- 不把外部 embedding model 的知识混同为 memory 学到的知识；
- 不说它天然比 Arithmetic 更适合 CPU/SSD offload；
- 不说 memory value 能跨 tokenizer、backbone 或任务直接迁移；
- 不把本文包装成新的 knowledge editing 方法。

---

## 2. Claim—Experiment 对照表

| 论文主张 | 必要实验 | 决定性对照 | 通过标准 |
|---|---|---|---|
| 地址具有语义结构 | n-gram pair 地址分析 | RQ-Shuffled | semantic/code-overlap 相关显著，shuffle 后消失 |
| 语义共享改善 LM | held-out LM NLL/PPL | Arithmetic-matched、RQ-Shuffled | 3 seeds，paired 95% CI 排除 0 |
| 收益来自共享 row | semantic-neighbor slice、shared-row exposure、head masking/row reset | 相似度匹配但不共享 code 的样本 | 收益集中在 shared-code 样本，干预共享 head 后下降 |
| 这是有限容量效应 | memory capacity sweep | K 相同的各方法 | 小容量优势更明显，随容量增大呈可解释趋势 |
| 不只是记住重复训练集 | one-pass/高独立文本训练 | 当前 8-epoch repeated pilot | 在更多独立文本 setting 中方向复现 |
| 不产生严重污染 | false-sharing slices | PAWS 型高词面低语义、否定、多义、关系冲突 | damage 不显著高于 Arithmetic，或 Mixed 明显缓解 |
| 有外部有效性 | 选一个跨语言或 Biomedical setting | RQ-Shuffled | 结果与 shared-row exposure 一致，而非仅平均分领先 |

如果一项 claim 没有对应实验，就删 claim，不用其他 benchmark 的平均分替代。

---

## 3. 方法与必须保留的基线

### 主方法

- **Semantic-RQ**：冻结 encoder + RQ dictionary，多级 code 对应多头 Engram rows。
- **Mixed**：一部分 Semantic heads，一部分 Arithmetic heads；用于 transfer–interference
  trade-off，不预设一定进入最终主方法。

### 主表基线

1. **Base / No-memory**：新增 memory 的绝对增益；
2. **Arithmetic-matched**：8 heads × 256 rows/head，参数、层、维度完全匹配；
3. **RQ-Shuffled-frequency-matched**：保留 code-vector 分布和训练访问频率，只破坏
   n-gram—semantic-code 对应；
4. **Semantic-RQ**；
5. **Mixed**。

LoRA 只在真实 post-hoc adaptation 表中出现，不是检验 hash 机制的核心基线。
Exact/MPHF 可以作为大容量上界放附录，但论文问题不是消除碰撞。

---

## 4. 实验一：当前运行的机制 LM pilot

### 设置

- Backbone：Qwen3-1.7B-Base，冻结 backbone、训练 Engram 模块；
- 数据：FineWeb-Edu；建表与 LM train/eval 文档严格隔离；
- context length：256；
- train rows：48,832；effective batch：32；
- optimizer steps：12,208；恰好 8 个完整 epoch；
- processed token slots：100,007,936；
- 方法：Arithmetic-matched、RQ-Shuffled、Semantic-RQ、Mixed；
- seeds：42、43、44。

200 条 held-out rows 只用于构建 frequency-matched shuffle 所需的精确 train-access
统计。正式逐 token slice 使用 2,000 条 held-out rows，并加入下一节的标准公开 LM
benchmark；两类 manifest 的 48,832-row train split 完全相同。

### 正确定位

这是一个 **repeated-source mechanism stress test**，不是“100M 独立 token 预训练”。它回答：
相同 source n-grams 反复更新 memory 后，参数共享是否能帮助未见 target n-grams。

### 主指标

- FineWeb held-out overall NLL/PPL；
- `exact_seen`；
- `semantic_neighbor_shared_code`；
- `semantic_neighbor_no_shared_code`；
- `covered_no_neighbor`；
- `address_oov`；
- `low_lexical_semantic_neighbor`。

统计固定为同 token 配对、document-cluster bootstrap，并在 3 seeds 外层重采样。
差值定义为 `Semantic-RQ − control`，负数更好。

### Gate 1 通过条件

必须同时满足：

1. Semantic-RQ 在 `semantic_neighbor_shared_code` 上优于 Arithmetic-matched 和
   RQ-Shuffled，95% CI 上界低于 0；
2. `semantic_neighbor_no_shared_code` 的收益明显更弱；
3. overall PPL 不显著恶化；
4. high-lexical/low-semantic false-sharing slice 不显著变差；
5. 结果不是单 seed 驱动。

若 Gate 1 不通过，停止扩规模；可转向 Mixed/frequency-aware gating，或写成 structured
collision 的负结果与诊断。

---

## 5. 实验二：标准、近 one-pass 的 LM 主结果

当前 pilot 通过后才启动。

### 训练设置

- 使用约 390,656 条长度 256 的训练序列，使 12,208 steps 基本只看一遍数据；
- 计算预算与当前 pilot 相同，改变的是独立文本数量，而不是增加训练 FLOPs；
- 先跑 Arithmetic-matched、RQ-Shuffled、Semantic-RQ × 3 seeds；
- Mixed 只有在 pilot 明确改善 false sharing 时才加入。

### 标准评测

- FineWeb-Edu 独立 held-out PPL（in-domain）；
- Paloma WikiText-103 word PPL（标准 corpus transfer）；
- Paloma C4-en word PPL（固定 Paloma 协议的宽域 corpus transfer）；
- LAMBADA accuracy/PPL（长程词预测）；
- HellaSwag、PIQA、ARC-E 只作为 secondary，不用一串知识 QA 掩盖 LM 结论。

### 为什么这是必需的

它排除“Semantic-RQ 只在小数据重复记忆时有效”的解释，也是 reviewer 判断结果是否具有
语言建模意义的关键一关。

---

## 6. 实验三：机制因果证据

仅有总 PPL 领先仍不足以证明 semantic hash 有效。

### 6.1 Shared-row exposure

对每个 target n-gram 计算：

```text
exposure = target 检索 rows 中在训练期实际被更新过的比例
gain     = NLL(control) - NLL(Semantic-RQ)
```

在相同 embedding similarity、lexical overlap、frequency 和 base NLL 分层内，检验 gain
是否随 exposure 增长。

### 6.2 直接干预

优先做 **shared-head masking**：评测时逐步屏蔽 target 与最近 source semantic neighbor
共同命中的 heads。若收益随 shared heads 被屏蔽而单调下降，才有较强因果证据。

再选一个低成本验证：

- 只重置共同 rows；或
- 固定 memory values，仅 permutation target code mapping。

### 6.3 Capacity sweep

主文不需要六个点。先做：

```text
K ∈ {64, 256, 1024}
```

每点 Arithmetic-matched、RQ-Shuffled、Semantic-RQ。先单 seed 找曲线，中心点和关键端点再补
3 seeds。预期小容量共享更强，Semantic-RQ 相对优势更明显；若无此趋势，“finite-memory
structured sharing”叙事需要降级。

### 6.4 False sharing

固定四类切片：

- 低词面、高语义：应该正迁移；
- 高词面、低语义：不应该共享；
- 同实体不同 relation / 时间冲突：高风险污染；
- 多义、否定、词序置换：语义编码器边界。

指标同时报告 positive transfer、false transfer、neighborhood damage。Mixed 只有在形成更好
的 transfer–damage frontier 时才算贡献。

---

## 7. 实验四：只选一条外部验证主线

不建议把 XNLI、PAWS-X、Biomedical、Knowledge Editing 全放主文，会像四个互不相干的
应用拼盘。

### 首选：跨语言表述泛化

- XNLI：English supervision → 其他语言零 target update；
- PAWS-X：专门检查高词面相似但语义不同的 false sharing；
- 方法：Arithmetic-matched、RQ-Shuffled、Semantic-RQ，3 seeds；
- 分析：每语言 coverage、shared-row exposure、低词面语义邻居、false-sharing。

这条线与“同义但词面不同的表述共享”最直接。

### 备选：Biomedical adaptation

若跨语言 coverage 太低，则改选 Biomedical：术语同义词、缩写和 long-tail concept 更容易
形成机制一致的切片。只做一个领域，避免 benchmark 堆叠。

### Knowledge Editing 的位置

CounterFact、ZsRE、WikiRecent 只放附录或一个小表，说明该机制也可能帮助 paraphrase
generalization。它不承担论文主结论，MQuAKE/PopQA 不进入主表。

---

## 8. 最终主表与主图

### Table 1：受控 LM 主结果

| Method | Overall PPL ↓ | Exact-seen NLL ↓ | Semantic-neighbor/shared-code NLL ↓ | No-shared-code NLL ↓ | OOV NLL ↓ |
|---|---:|---:|---:|---:|---:|
| Arithmetic-matched | | | | | |
| RQ-Shuffled | | | | | |
| Semantic-RQ | | | | | |
| Mixed（若保留） | | | | | |

### Table 2：标准 LM benchmark

FineWeb held-out、Paloma WikiText-103、Paloma C4-en、LAMBADA，报告 3-seed mean ± std 和 paired CI。

### Figure 1：Structured sharing 方法图

Arithmetic 随机共享与 Semantic-RQ 多头部分共享对比。

### Figure 2：机制图

`target gain` 随 `shared-row exposure` 的变化，并画出 shared-head masking 曲线。

### Figure 3：容量曲线

K={64,256,1024} 下 overall gain 与 semantic-neighbor gain。

### Figure 4：Transfer–interference frontier

positive transfer 对 false-transfer damage，比较 Arithmetic、RQ-Shuffled、Semantic-RQ、Mixed。

---

## 9. 4 × A100 40GB 的执行顺序

### Phase A：完成当前 Gate 1

四卡动态排队跑 4 methods × 3 seeds；同时产出 token manifest、逐 token loss 和 bootstrap。
正式 fixed-step 队列已经就绪；当前卡上的早停旧 run 只作诊断，释放 GPU 后调度器自动接管，
不把旧结果混入主表。

### Phase B：机制分析

- GPU0：shared-head masking；
- GPU1：row-reset/permutation intervention；
- GPU2：false-sharing slices；
- GPU3：标准 LM eval。

多数是评测任务，时间显著短于重新训练。

### Phase C：one-pass LM replication

3 methods × 3 seeds，共 9 runs；四卡约三轮。每 run 的 optimizer steps 与当前 pilot 一致，
预计墙钟与当前 12-run pilot 同量级或略低。

### Phase D：capacity sweep

先 3 capacities × 3 methods × 1 seed；只有呈现预期曲线才补关键点 seeds。

### Phase E：外部验证

XNLI + PAWS-X 或 Biomedical 二选一作为主文外部验证；另一个和 KE 均放附录。

---

## 10. 投稿前硬门槛

- [ ] 3-seed Semantic-RQ 在 semantic-neighbor/shared-code 主终点显著优于两个核心基线；
- [ ] RQ-Shuffled 的访问频率和参数量严格匹配；
- [ ] one-pass/高独立文本 setting 复现方向；
- [ ] 标准 LM benchmark 至少两个不退化；
- [ ] shared-head intervention 支持共享 row 的因果链；
- [ ] capacity curve 支持 finite-memory hypothesis；
- [ ] false-sharing 被量化且风险可控；
- [ ] 至少一个外部场景与机制切片一致；
- [ ] 报告 table coverage、fallback、storage、建表成本、throughput 和显存；
- [ ] 不把 token slots 写成 unique tokens，不把地址可迁移写成 value 可迁移。

前六项缺一，这个工作都还不够成为强方法论文。若只有知识编辑或某个 downstream 平均分
领先，应该停止包装，而不是继续堆任务。

---

## 11. 论文结构

1. **Introduction**：有限 memory 中碰撞不可避免，问题是如何组织共享；
2. **Background**：Engram、Arithmetic hash、collision-free negative evidence；
3. **Method**：offline semantic encoding、RQ multi-code、frozen lookup、Mixed；
4. **Controlled Language Modeling**：matched baselines、one-pass replication、标准 benchmark；
5. **Mechanism and Boundaries**：shared-row exposure、干预、capacity、false sharing；
6. **External Validity**：只保留一条主文应用线；
7. **Efficiency and Limitations**：建表成本、coverage/OOV、encoder dependence、polysemy；
8. **Conclusion**：只总结实验真正支持的 structured-sharing 结论。

## 12. Reviewer 式最终判断

这不是一篇知识编辑论文，也不应该靠“在很多任务上平均高一点”成立。最有机会的论文形态是：

> **一个关于 finite conditional memory 中 structured collision 的语言建模方法与机制论文。**

当前 Gate 0 已证明地址确实编码语义结构；正在运行的 Gate 1 决定这种结构能否转化为
模型收益。只有 Gate 1、one-pass replication、机制干预和 capacity curve 同时成立，才有
资格把它写成方法论文。否则最诚实且可能仍有价值的产出，是 structured semantic collision
何时有效、何时因 false sharing 失败的系统性分析。
