# 推断驱动的敏感度评估：方法设计与记录

> 本文档记录「如何从推断任务得到敏感度分数」的两条路线，供后续实现。
> 路线 A 可立即执行（稳健基线）；路线 B 是主攻创新方向（待验证）。

---

## 0. 背景：IGNN 的缺陷与我们的目标

IGNN（参考论文）的敏感度：

- **源**：confidential 锚定为 1，general 用 **AHP 专家打分**（主观）。
- **结构**：在**预设的相关性图**上用 GAT 做半监督传播（loss 只在 confidential 节点）。
- 本质 = **风险分数扩散模型**；用 DIL test（遮蔽高敏感数据去推断、看准确率 MPA）
  **事后验证**敏感度，但敏感度本身不是从推断来的。

核心矛盾：**用"推断准确率"验证敏感度，却用"主观先验扩散"定义敏感度。**

我们的目标：**让敏感度成为推断能力的函数（客观、可验证、无需 AHP）**，
并（理想情况）**沿推断任务学到的真实信息流路径扩散**。

---

## 1. 先厘清两个正交维度（避免概念混淆）

敏感度归因有**两个独立维度**，可自由组合：

- **维度一 · 归因方式**（敏感度怎么分配到字段）：遮蔽 / 单字段 / Shapley。
- **维度二 · 度量指标**（用什么衡量"推断得好不好"）：R² / 准确率 → V-usable information。

「遮蔽 vs Shapley」是维度一；「R² vs V-info」是维度二。V-info 不是第三种归因，
它是把度量的"尺子"换好，可以套在任何归因方式上。

---

## 2. 路线 A：推断能力归因敏感度（可立即执行）

### 2.1 三种归因方式（维度一）

用一个例子讲清。推断机密字段 C，有 general 字段 A、B、D：

- 只用 A→0.8；只用 B→0.8；**A、B 高度冗余**，{A,B}→0.8
- 只用 D→0.3；**D 与 A 互补**，{A,D}→0.9；全集 {A,B,D}→0.9

| 归因方式 | 定义 | 例子结果 | 衡量的概念 |
|---|---|---|---|
| **遮蔽法（leave-one-out）** | `v(全集) − v(全集\j)` | A=0, B=0, D=0.1 | **必要性**（删了它会不会出事） |
| **单字段法** | `v({j}) − v(∅)` | A=0.8, B=0.8, D=0.3 | **充分性**（它自己能泄露多少） |
| **Shapley 值** | 所有子集上边际贡献的加权平均 | A≈0.37, B≈0.37, D≈0.17 | **公平综合**（化解冗余+协同） |

**关键统一视角**：遮蔽法 = 只取"全集"这一个子集的边际项；单字段法 = 只取"空集"
这一项；**Shapley = 从空集到全集所有子集边际贡献的加权平均**。即遮蔽法与单字段法
都是 Shapley 的极端特例。

**遮蔽法的致命问题（本项目场景下）**：A、B 是最强泄露源（各自单独推 C 到 0.8），
却因**互相掩护**被判敏感度=0。本项目目标是"数据该不该公开/多危险"（不是"删谁能
降风险"），所以**遮蔽法与目标不匹配**，会系统性漏判冗余的强泄露字段。

**Shapley 的优良性质**：满足公平公理——对称字段同分、无贡献字段必为 0、
**总和守恒**（Σ_j φ_j = v(全集) − v(∅) = 系统总可泄露量）。因此可把敏感度
归一化成"占总泄露的百分比"，解释性好。是数据估值领域（Data Shapley,
Ghorbani & Zou 2019）的标准工具。

### 2.2 度量指标（维度二）

- **基础版**：v(S) = 用字段子集 S 推断 confidential 的 R² / MPA 准确率。
- **升级版**：v(S) = **V-usable information** `I_V(S→C) = H_V(C) − H_V(C|S)`
  （Ethayarajh et al. 2022），用模型 log-loss 估计。优点：单位是比特、可加、
  跨字段可比、显式以"攻击者模型族 V"为前提（契合 DIL 威胁模型）；还能用 PVI
  给每个样本/每个机密字段的细粒度泄露。

### 2.3 计算与实现要点

- Shapley 精确算需遍历 2ⁿ 子集；本项目 n=44 general 字段 → 用**蒙特卡洛采样**
  近似：随机排列字段、逐个加入、记录边际增量、多次平均（几百~几千次排列即可收敛）。
- 每个子集的 v(S) 需要一个能"只用子集 S 推断"的模型。两种做法：
  1. 对每个采样子集重训小模型（准但慢）；
  2. 训一个支持"输入掩码"的模型，推理时把非 S 字段置零/置均值（快，近似）。
- 度量模型用**强模型**（DNN probe），见 2.4。

### 2.4 在本项目怎么落地

1. 用已训练好的 **DNN probe**（每个机密字段一个强 MLP）作为"攻击者能力上界"，
   度量可推断性。
2. 对 44 个 general 字段分别算 **遮蔽 / 单字段 / Shapley** 三套敏感度（先 R²，后 V-info）。
3. 画三者排名对比图，**复现"遮蔽法把冗余强泄露源误判为低敏感"**——这张图本身
   就是论文里论证"为什么需要 Shapley / 为什么 IGNN 式扩散不够"的有力证据。

### 2.5 路线 A 的局限（也是用户的落差所在）

路线 A 得到的是**一组字段级分数**，**没有图扩散路径结构**——丢掉了相对 IGNN
最有价值的"敏感度沿推断路径传播"的可解释性。这正是路线 B 要补上的。

---

## 3. 路线 B：推断路径驱动的敏感度扩散（主攻创新方向，待验证）

### 3.1 动机

把 IGNN 扩散的两个支柱都换成推断驱动的客观量：

| 维度 | IGNN | 路线 B |
|---|---|---|
| 敏感度**源** | AHP 主观先验 | 推断能力 a_c（客观，从推断任务） |
| 扩散**结构** | 预设相关性图 + GAT | **推断模型学到的信息流路径**（GNN-LRP） |
| 扩散方向 | confidential→general 预设风险传播 | 推断相关性从 c 反传分解（数据驱动） |
| 敏感度验证 | 事后 DIL test | 内生（敏感度即推断贡献，自洽） |

### 3.2 技术载体：GNN-LRP（相关游走路径）

**GNN-LRP**（Schnake et al., *Higher-Order Explanations of GNNs via Relevant Walks*,
TPAMI 2021, arXiv:2006.03589）：把一个 GNN 预测**分解到图上的游走路径**
（walk，如 `v_j → v_k → … → v_c`），每条路径得到相关性 `R(W)`，表示
"信息沿这条路径对该预测的贡献"。基于逐层 LRP 反向传播，从输出节点把相关性
**反传**到输入节点，沿途按消息传递结构分配。

关键契合点：
- 推断模型是 general→confidential 前向推断；GNN-LRP 的相关性从输出 c **反传**
  回 general 节点——**方向天然就是"c 的可推断性沿路径分配给 general"**，正是
  "敏感度扩散"的语义。
- 相关性**守恒**：`Σ_W R(W|c) = f_c`（模型对 c 的预测强度）。若模型推不准 c，
  预测信号弱，总相关性自然小 → 路径相关性的总量**天然正比于可推断性**。

### 3.3 形式化定义

记：
- 图推断模型 M（general↔general GCN + general→confidential GAT，多层）。
- 机密字段 c 的可推断性 `a_c`（held-out R² 或 V-usable info），客观、可验证。
- GNN-LRP 把"M 推断 c"分解为路径集合 {W}，每条 `W: v_j ⇝ v_c` 得相关性 `R(W|c)`。

**字段 j 的敏感度**：

```
S(j) = Σ_c  a_c · Σ_{W: j ⇝ c}  |R(W | c)|
```

即：j 通过所有推断路径、对各机密字段贡献的信息流总和，按各机密字段的实际
可推断性 a_c 加权。

满足用户的三个要求：
1. **敏感度是推断能力的函数**：a_c 加权 + R(W|c) 来自推断模型（不割裂）。
2. **沿推断路径扩散**：路径分解 R(W|c) 可回答"敏感度从 c 经哪些边流到 j"。
3. **客观、无 AHP**：全部从推断任务导出。

### 3.4 与路线 A 的关系（B 严格泛化 A）

- 只取长度=1 的路径（j 直连 c）→ 退化为路线 A 的"单字段直接归因"。
- 用集合扰动代替路径 → 退化为 Shapley。
- 路线 B 多出的是**多跳传播结构的解释**——IGNN 图扩散的精神，但路径来自推断
  模型而非主观相关性。

因此路线 A 可作为路线 B 的**对照 / sanity check**：B 的字段级汇总 S(j) 应与 A
的 Shapley 排名大致一致，但 B 额外给出路径级解释。

### 3.5 难点与验证方案（必须诚实面对）

1. **GNN-LRP 实现复杂度**：需对每层消息传递做 LRP 反传；路径数随深度指数增长，
   需剪枝/Top-k 路径或按字段聚合。缓解：本项目层数少（3 层）、节点少（56），可控。
2. **归因保真度**：需验证 R(W|c) 真的反映信息流。Sanity check——删掉高相关性
   路径上的边后，推断 c 的准确率应明显下降（删低相关性路径影响小）。
3. **"是推断能力的函数"的具体形式**：a_c 怎么加权、R 怎么归一，需 ablation
   （如 a_c 用 R² vs V-info；是否对 R 取绝对值/正部）。
4. **方向性**：确认 LRP 反传方向 = c→general 的敏感度分配（与"风险从敏感源传播"
   的语义一致）。

### 3.6 备选技术载体（若 GNN-LRP 太重）

- **逐层雅可比 / Integrated Gradients on graph**：`∂(c 预测)/∂(general 输入)`
  沿层反传——比注意力可靠，实现比 GNN-LRP 简单，但不显式给"路径"而给"字段总贡献"
  （介于 A 与 B 之间）。
- **GraphMask**（Schlichtkrull et al. 2021）：每层学边的可丢弃性，找出真正承载
  信息流的边——给"重要边"而非"路径"。
- **注意力 × 先验扩散**（用户最初思路 1）：最简单但最不可靠（Attention is not
  Explanation），仅作下界对照。

---

## 4. 推荐执行顺序

1. **先做路线 A**（DNN probe 上跑 遮蔽 / 单字段 / Shapley，先 R²）：
   - 产出三者敏感度排名对比图，复现"遮蔽法漏判冗余强泄露源"。
   - 作为稳健基线 + 论文里"为什么需要更好方法"的论据。
2. **再做路线 B**（GNN-LRP 路径扩散，主创新）：
   - 先把图推断模型精度拉到接近 DNN probe（加残差 MLP 分支 `ŷ=ŷ_graph+β·MLP(x_G)`），
     保证"推断能力"可信。
   - 实现 GNN-LRP 路径分解 → 按 §3.3 算 S(j) → 画"敏感度传播路径"。
   - 用路线 A 的 Shapley 排名做 sanity 对照。
3. **度量升级**：R² → V-usable information（两条路线都适用）。

---

## 5. 参考文献

- IGNN 参考论文（本项目 PDF）：*IGNN: An Inference Graph Neural Network for Data
  Sensitivity Assessment Considering Data Inference Leakage in Cyber-Physical Power Grid.*
- [Schnake et al., Higher-Order Explanations of GNNs via Relevant Walks (GNN-LRP), TPAMI 2021](https://arxiv.org/abs/2006.03589)
- [Ethayarajh et al., Understanding Dataset Difficulty with V-Usable Information, 2022](https://arxiv.org/pdf/2110.08420)
- [Quantifying and Localizing Usable Information Leakage, 2021](https://arxiv.org/pdf/2105.13929)
- Ghorbani & Zou, Data Shapley: Equitable Valuation of Data for Machine Learning, 2019.
- Schlichtkrull et al., Interpreting Graph Neural Networks for NLP with Differentiable Edge Masking (GraphMask), 2021.
- Jain & Wallace, Attention is not Explanation, 2019.
