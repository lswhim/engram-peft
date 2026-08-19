# Semantic-Keyed Engram：论文与实验规划

> 版本：v1（2026-08-19）
> 目标：ICLR 2027 方法论文
> 资源：多台 A100 40GB（当前 1 号机 4 卡、2 号机 2 卡、新机 8 卡）
> 代码分支：`semantic-hash`

## 1. 一句话故事

Engram 的地址和记忆是绑在一起的：算术 hash 用 token id 决定“存哪一行”，gating 再用这一行的内容决定
“读多少”。我们把地址改成语义化的 frozen RQ code，并在读取时把 **RQ 语义几何当作 key、可训练 memory row
当作 value**。这让模型在决定要不要相信一次记忆读取时，看的是“这个 n-gram 语义上是什么”，而不是“这一行
碰巧存了什么”，从而让语义相近的表达更稳定地复用同一次写入。

暂定标题：

> **Semantic Keys for Conditional Memory: Decoupling Address Meaning from Stored Memory**

## 2. 问题

原版 Engram 有三个相互耦合的环节：

```text
token n-gram -> arithmetic hash -> memory row
memory row + hidden state -> context gate -> readout
```

其中 `memory row` 既是被寻址的存储位置，又是 gate 计算 attention 的依据。地址本身没有语义，所以：

- 语义相近、表面不同的 n-gram 大概率落到不同行，无法共享一次写入；
- 即使发生碰撞，也是随机的，模型没有理由信任这次共享；
- gating 看到的信号混入了“这一行碰巧存了什么”，而不是“这个地址本身是否可靠”。

本工作的目标不是“让索引更准”，也不是“减少碰撞”，而是：

> 在有限 memory 和持续写入的压力下，把地址的语义信息显式地注入读取门控，从而把随机共享变成可复用、
> 可信任的结构化共享。

## 3. 方法：Semantic-Keyed Readout

### 3.1 地址层

运行时对每个 2/3-gram：

```text
n-gram -> frozen Qwen3-Embedding -> frozen FAISS ResidualQuantizer -> M discrete codes
```

- `M=8` 个 residual levels 直接映射 Engram 的 8 个 hash heads。
- 每个 code 同时索引两样东西：冻结的 RQ centroid（语义地址）和可训练的 Engram row（记忆内容）。
- 首次见到 n-gram 时执行 embedding + RQ forward，随后写入 persistent SQLite cache；之后是确定性的 O(1) 查表。
- 表外 n-gram 不回退 arithmetic hash，否则语义泛化在真正的新表面形式上会消失。

### 3.2 读取层（核心改动）

原版 flatten 读取使用 `ContextAwareGating`，让 memory row 同时充当 key 和 value。本方法改为
`SemanticKeyedGating`：

```text
value = trainable Engram row
key   = frozen semantic geometry(code)
```

对一个 n-gram 的 8 个 code，语义几何被分解为三部分：

```text
centroid_j = codebook_j[code_j]              # 当前 level 的局部 centroid
prefix_j   = sum_{l<=j} centroid_l           # 到第 j 级的 coarse 累积重建
tail_j     = sum_{l>j} centroid_l            # 第 j 级之后的剩余精细残差
```

这三个 descriptor 分别表示“局部语义中心”“由粗到细的重建前缀”“尚未解释的细粒度尾部”。它们都由冻结 RQ
几何决定，不随训练改变。读取时：

```text
geometry = softmax(geometry_mix) . (centroid, prefix, tail)
semantic_key = proj(geometry) + head_embedding
```

其中 `geometry_mix` 和 `proj` 是可训练的，但它们的输入始终是冻结的语义几何。gate 仍沿用原 Engram 的
score/softmax/sigmoid 注意力，只是把 key 从“memory row”换成了“semantic key”：

```text
score = <norm(semantic_key), norm(hidden)>
gate  = softmax(score) * sigmoid(sum(score))
output = gate @ value
```

因此，**地址语义（key）与存储内容（value）被解耦**。模型在决定读哪些 head、读多少时，依据的是该地址在
语义空间中的位置，而不是该行当前存了什么。

### 3.3 与 Semantic-Flatten 的关键区别

| 维度 | Semantic-Flatten | Semantic-Keyed |
|---|---|---|
| 地址 | RQ code | RQ code |
| key | trainable memory row | frozen RQ centroid/prefix/tail geometry |
| value | trainable memory row | trainable memory row |
| gate 依据 | 行内容 | 地址语义几何 |
| 语义信息注入 | 只通过地址共享 | 地址共享 + 读取键 |

这个区别构成论文方法贡献的核心：semantic key 不是另一个 routing/稀疏化 trick，而是把 Engram 原本模糊的
“寻址即读取”关系明确拆开。

## 4. 核心 claim 与非 claim

**主张：**

1. 语义地址让不同表面形式可以部分共享 memory row；
2. 语义 key 让读取门控依据“地址语义”而非“行内容”，从而更稳定地复用已写入的语义邻居；
3. 在高碰撞压力下，这种共享对 rewrite / paraphrase 泛化有可测量的收益。

**不主张：**

- 不主张“索引更准”。RQ 本身不是贡献，用它当 Engram 地址并做 key/value 解耦才是。
- 不主张对所有 benchmark 全面更优。收益应集中在“存在语义邻居且地址可覆盖”的样本上。
- 不主张 RQ level 是可解释的关系/实体层级；只称 residual-energy 的 coarse-to-fine 几何。

## 5. 方法矩阵

论文需要三类方法形成因果对照：

1. **Arithmetic-fixed**：原版算术地址 + flatten 读取，作为“无语义地址”的锚点。
2. **Semantic-RQ + flatten**：语义地址，但保留原版 gating，key=value=memory row。
3. **Semantic-RQ + keyed**：语义地址 + semantic-keyed 读取，完整方法。

可选诊断对照：

- **RQ-Shuffled + keyed**：保持 code 的访问频率/每级分布，但切断 code 与语义几何的对应关系；若 keyed 的优势
  主要来自“地址语义”，在此控制下应消失。
- **Shuffled + flatten**：与 Semantic-flatten 对应，分离 addressing 的贡献。

容量匹配约束：arithmetic 的每个 head 行数必须等于 RQ 的 `codebook_size`。当前 `K=256`、`heads=8`，因此
arithmetic-fixed 的总容量应为 `[2048, 2048]`，而不是 `[8192, 8192]`（后者是历史错误，已修正）。

## 6. Benchmark 与证据定位

### 6.1 知识编辑（CounterFact / ZsRE）

- 直接测“写入新事实后，paraphrase 是否被正确改写”，对应语义共享最直观的场景。
- 指标：Efficacy、Paraphrase、Specificity、Harmonic score。

### 6.2 跨语言（XNLI / PAWS-X）

- 英文训练、多语言零样本评测，测“同一语义在不同语言表面形式之间是否复用”。
- XNLI 是 15 语 NLI，PAWS-X 是 7 语 paraphrase 判别。

### 6.3 受控语言建模（FineWeb -> WikiText / LAMBADA）

- 测“语义地址与语义 key 是否改善 Engram 的本职语言建模任务”。
- 当前是 400 步 pilot，只证明可训练、不破坏冻结底座；正式主表需要更长步数和 token 切片分析。

### 6.4 WikiBigEdit official（8 timestep lifelong editing）

- 测“持续写入 + 少遗忘”，对应高碰撞压力下的结构化共享。
- 五轴：Update / Rephrase / Personas / Mhop / Locality。

## 7. 当前证据状态

截至 2026-08-19，已完成的单 seed（42）结果：

**知识编辑：**

| 数据集 | 指标 | arithmetic | semantic_flatten | semantic_keyed |
|---|---|---:|---:|---:|
| CounterFact | Harmonic | 45.33 | 47.96 | **48.85** |
| ZsRE | Harmonic | 51.42 | 51.08 | **52.36** |

**PAWS-X macro acc：**

| arithmetic | semantic_flatten | semantic_keyed |
|---:|---:|---:|
| 80.73 | 82.22 | **82.81** |

这两个结果给出核心方向信号：`semantic_keyed > semantic_flatten > arithmetic`，即 key/value 解耦在
语义地址之上还有增量。

**尚未完成或尚不能下结论：**

- XNLI 的 semantic_flatten / semantic_keyed 仍在训练；
- 语言建模是 pilot，且 arithmetic 容量刚修正，需重跑；
- WikiBigEdit official 正在跑 8 timestep；
- 全部结果目前主要是 seed 42，未到多 seed 标准。

## 8. 必须补齐

1. 三个核心方法跑满 `42 / 123 / 456` 三个 seed。
2. XNLI 两个 semantic 方法完成，补齐跨语言表。
3. 语言建模升级：拉长步数、arithmetic 用 `[2048,2048]`、跑 token 切片（exact_seen / semantic_neighbor /
   long_tail / covered_no_neighbor / address_oov）。
4. WikiBigEdit official 完成 8 timestep，报告五轴 + retention/forgetting。
5. 加入 Shuffled-keyed 因果对照，验证 keyed 的收益确实来自“地址语义”而非一般结构变化。
6. 报告动态 OOV 的 offline-hit / dynamic-OOV 切片，证明语义泛化在未见 n-gram 上仍存在。

## 9. 成功门槛

- 主结果 3 seeds，报告 mean ± sample std，并做 query-level paired bootstrap。
- `semantic_keyed` 在至少两个 benchmark 上对 `semantic_flatten` 同方向、且 95% CI 不跨零。
- 至少一个外部 KE benchmark 和至少一个跨语言/LM benchmark 复现。
- Shuffled-keyed 控制中，keyed 相对 flatten 的优势应明显缩小或消失。
- 不报告最好 seed / 最好 checkpoint，不在 test milestone 上调参。

## 10. 主要风险

- **keyed 的增量可能不稳定**：目前只有 seed 42 的 KE / PAWS-X 支持，需多 seed 验证。
- **semantic key 是否只是另一种 gating 初始化**：需要用 Shuffled-keyed 和冻结 geometry 的消融排除。
- **动态 OOV 成本**：冷路径 embedding forward 较慢，需如实报告 cold/warm latency 与 cache hit。
- **benchmark 覆盖面**：不能只靠 WikiBigEdit 一个主表，外部 KE 与跨语言必须补足。
