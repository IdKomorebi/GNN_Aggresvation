# 基于通用集合推断能力模型的字段级 Shapley 与交互分析：研究方案草案

> 用途：本文件可直接交给 Codex / Claude Code 结合现有项目代码继续分析、设计与实现。
>
> 当前主线建议：**暂不研究“主动搜索全部推断路径”**，把论文聚焦在“集合级推断能力学习 → 字段级总体贡献 → 字段交互/协同关系”三层结构上。

---

## 0. 一句话概括当前研究思路

对于目标字段 `Y`，真实想研究的不是“某个已经训练好的模型依赖哪些输入字段”，而是：

> **任意输入字段集合 `S` 本身能够对 `Y` 提供多少总体推断能力，以及这份集合级能力应如何合理分解到单字段及字段交互上。**

因此，先定义一个理想的集合价值函数：

\[
V^*(S)=\text{字段集合 }S\text{ 对 }Y\text{ 的总体潜在推断能力},
\]

再训练一个可快速查询的通用模型：

\[
F_\theta(S)\approx V^*(S),
\]

使原本昂贵的

\[
S\rightarrow \text{重新训练推断器}\rightarrow R^2
\]

被摊销为

\[
S\rightarrow F_\theta(S)\rightarrow \widehat V(S).
\]

之后，在同一个 `F_theta` 上：

1. 用 **Shapley value** 将集合级推断能力分解为字段级总体推断贡献；
2. 用 **Shapley interaction / Shapley-Taylor / Faith-Shap** 分析字段之间的协同、冗余与替代关系；
3. 不在本文主线中强行解决指数级的完整路径搜索问题。

---

# 1. 首先必须区分的几个概念

## 1.1 Shapley value 是数学分配规则，不等于 SHAP

Shapley value 只要求存在一个集合价值函数：

\[
v:2^N\rightarrow \mathbb{R}.
\]

对于字段 `i`：

\[
\phi_i(v)=
\sum_{S\subseteq N\setminus\{i\}}
\frac{|S|!(p-|S|-1)!}{p!}
\left[v(S\cup\{i\})-v(S)\right].
\]

它表达的是：

> 字段 `i` 在所有可能合作上下文中加入后所产生的平均边际价值。

Shapley 本身并不知道“预测”“推断”“R²”或“神经网络”是什么。**Shapley 值的实际含义完全由 `v(S)` 决定。**

---

## 1.2 普通 SHAP 主要解释“已有模型”

普通 SHAP 的中心对象通常是一个已经训练好的预测模型：

\[
f(x).
\]

它构造局部 coalition value，例如：

\[
v_x(S)=\mathbb E[f(X)\mid X_S=x_S],
\]

然后计算：

\[
\phi_i(x)=\operatorname{Shapley}_i(v_x).
\]

因此普通 SHAP 回答的是：

> **对于当前模型 `f`（通常还针对某个具体样本），字段 `i` 对模型输出贡献了多少？**

它更接近 **model reliance / model attribution**，而不等于“字段本身对目标变量所拥有的总体可推断能力”。

即使一个全量模型整体 `R²` 很高，也不代表它以唯一、完整、公平的方式利用了所有有推断价值的字段。

典型反例：

- `X1` 和 `X2` 都能单独很好地预测 `Y`；
- 全量模型在优化过程中主要使用 `X1`，几乎忽略 `X2`；
- 模型整体 `R²≈0.99`；
- SHAP 可能得到 `X1 >> X2`；
- 但实际上 `R²(X2→Y)` 也可能接近 0.99。

因此：

\[
\boxed{\text{model reliance}\neq\text{intrinsic / population predictive capability}}
\]

这也是本文不直接使用“全量模型 + global SHAP”作为最终字段推断能力定义的根本原因。

---

## 1.3 与本文最相关的是 SAGE / SPVIM 的“总体预测能力”视角

SAGE 将 global feature importance 与 predictive power 联系起来，并区分：

- model-based predictive power；
- universal predictive power。

SPVIM 则进一步研究 population-level variable importance，并建立统计估计、渐近分布、置信区间和假设检验理论。

因此本文的理论立足点不应写成：

> “首次用 Shapley 衡量字段预测能力。”

这个说法不成立。

更合理的定位是：

> **学习一个可重复、低成本查询的推断能力集合函数，再把已有成熟的 Shapley / interaction 理论作为下游解析工具。**

---

# 2. 推荐的理论问题定义

设：

- 输入字段全集：`N={1,...,p}`；
- 任意字段子集：`S⊆N`；
- 目标字段：`Y`；
- 数据总体分布：`P(X,Y)`。

## 2.1 理想总体推断能力函数

在平方损失下，一个很干净的理论定义是：

\[
f_S^*(x_S)=\mathbb E[Y\mid X_S=x_S],
\]

即只使用 `X_S` 时的 Bayes 最优预测器。

定义总体推断能力：

\[
V^*(S)
=
1-
\frac{\mathbb E[(Y-f_S^*(X_S))^2]}{\operatorname{Var}(Y)}.
\]

利用全方差公式，还可以写成：

\[
\boxed{
V^*(S)
=
\frac{\operatorname{Var}(\mathbb E[Y\mid X_S])}
{\operatorname{Var}(Y)}
}
\]

这是一个非常适合论文理论部分的定义，因为：

1. 它直接表达 `X_S` 对 `Y` 所包含的可预测信息；
2. 它不依赖某一个具体训练算法；
3. 理论上 `0≤V^*(S)≤1`；
4. 若 `S⊆T`，则理想情况下：

\[
V^*(S)\le V^*(T),
\]

即满足集合单调性；
5. 真实 `V^*(S)` 无法直接从有限样本精确观测，只能估计。

这就给本文提供了明确的“待估计对象”。

---

## 2.2 实际有限数据中的近似

现实只有：

\[
\mathcal D=\{(x^{(k)},y^{(k)})\}_{k=1}^n.
\]

因此需要先通过强预测器、独立验证集或 cross-fitting 得到：

\[
\widetilde V(S)\approx V^*(S).
\]

例如：

\[
\widetilde V(S)=R^2_{\text{CV}}(S\rightarrow Y).
\]

注意：

- 理论对象是 `V*`；
- 由实际预测器训练得到的是 `V_tilde`；
- 通用模型再拟合 `V_tilde` 得到 `F_theta`。

论文里最好严格区分三层：

\[
\boxed{
V^*(S)
\rightarrow
\widetilde V(S)
\rightarrow
F_\theta(S)
}
\]

其中：

- `V*`：理想总体推断能力；
- `V_tilde`：有限样本 + 强推断器得到的近似标签；
- `F_theta`：对集合价值函数的摊销近似。

这样理论逻辑会非常清楚。

---

# 3. 本文真正应该强调的核心模型：Amortized Inference Value Function

## 3.1 通用模型的角色

现有项目已经具备或正在研究：

\[
F_\theta(m_S)\rightarrow \widehat{R^2}(S),
\]

其中 `m_S∈{0,1}^p` 是字段 mask。

本文不应把它描述成“Shapley 网络”，而应该明确：

> **它学习的是底层 coalition value function / inference value function 本身。**

这是与 FastSHAP 的关键差异之一。

FastSHAP 的典型目标是直接学习：

\[
x\rightarrow \phi(x),
\]

即直接摊销 Shapley explanation。

本文则是：

\[
S\rightarrow V(S).
\]

因此同一个 `F_theta` 可以支持多个下游算子：

\[
F_\theta
\rightarrow
\begin{cases}
\text{Shapley field attribution}\\
\text{pairwise interaction}\\
\text{higher-order interaction}\\
\text{arbitrary coalition query}\\
\text{后续可能的其他集合分析}
\end{cases}
\]

所以创新点不应表述为“FastSHAP 的更快版本”。

更合适的是：

> **摊销底层推断能力集合函数，而非摊销某一种固定解释量。**

---

## 3.2 建议考虑 target-conditioned 版本

如果项目中存在多个目标字段，可考虑：

\[
F_\theta(t,S)\rightarrow \widehat V_t(S),
\]

其中：

- `t` 表示目标字段；
- `S` 表示输入字段集合。

这样一个模型能够学习多个目标的集合推断能力函数。

潜在优势：

1. 模型可以在不同目标间共享字段关联结构；
2. 最终可以形成完整的字段间推断贡献矩阵；
3. 实际部署/分析时无需每个目标重新训练一个完全独立模型。

若现有项目结构不适合，可以先保留单目标版本，不必为了故事强行修改。

---

# 4. 字段级总体推断贡献：Shapley on `V(S)`

定义：

\[
\phi_i^*
=
\operatorname{Shapley}_i(V^*).
\]

即：

\[
\phi_i^*
=
\sum_{S\subseteq N\setminus\{i\}}
\frac{|S|!(p-|S|-1)!}{p!}
\left[
V^*(S\cup\{i\})-V^*(S)
\right].
\]

其语义不是：

> “字段 `i` 对某一个模型输出有多重要”。

而是：

> **字段 `i` 在各种可能字段组合上下文中，对总体推断能力所产生的平均边际贡献。**

这正是本文所需的字段级“总体推断贡献”。

---

## 4.1 为什么比全量模型 SHAP 更符合本文问题

普通全量模型 SHAP：

\[
f(X)\rightarrow Y
\quad\Rightarrow\quad
\text{解释 }f\text{ 实际依赖哪些字段}.
\]

本文：

\[
V(S)=\text{subset predictive capability}
\quad\Rightarrow\quad
\text{解释字段本身在所有组合下具有多少推断贡献}.
\]

特别是在：

- 强相关；
- 冗余；
- 替代；
- 高阶交互；

存在时，两者会明显不同。

因此实验中建议将：

> **Full-model global SHAP**

作为一个重要 baseline，而不是最终定义。

---

# 5. 计算 Shapley：不要混淆“子集数量”与“单次价值评估成本”

这是论文中必须讲准的地方。

## 5.1 Williamson & Feng (SPVIM) 解决什么

精确 Shapley 需要 `2^p` 个 coalition。

Williamson & Feng 证明，在 `n` 个观测样本下，可以只随机采样约：

\[
\Theta(n)
\]

个特征子集，仍获得具有良好渐近统计性质的 SPVIM 估计，并进一步构造：

- confidence interval；
- hypothesis test；
- asymptotically optimal estimator。

它主要降低的是：

\[
\boxed{\text{需要评估多少个 subset}}
\]

而不是自动消除每个 `V(S)` 本身的模型训练成本。

---

## 5.2 本文通用模型解决什么

本文 `F_theta` 主要降低的是：

\[
\boxed{\text{单次 coalition value query 的成本}}
\]

从：

\[
\text{训练/验证一个 predictor}
\]

变成：

\[
F_\theta(S)\text{ 的一次 forward pass}.
\]

因此两类方法是互补的：

- SPVIM / Kernel-Shapley 类采样：减少需要查询的 coalition 数量；
- `F_theta`：降低每一次 coalition query 的成本。

论文中应明确区分这两个加速维度。

---

# 6. 字段协同、冗余与替代：Interaction

普通 Shapley 会把高阶 interaction 产生的价值最终分摊到单字段上，因此：

> 它能够考虑 interaction 对“字段总体价值”的影响，但无法保留 interaction 本身的结构。

例如：

\[
V(\emptyset)=0,
\quad
V(A)=0,
\quad
V(B)=0,
\quad
V(A,B)=1.
\]

普通 Shapley：

\[
\phi_A=\phi_B=0.5.
\]

它正确说明 A、B 都应分得价值，但无法直接告诉我们：

> 整个价值实际上来自 `A+B` 的联合效应。

因此需要 interaction index。

---

## 6.1 最基础的二阶离散差分

对字段 `i,j`，定义：

\[
\Delta_{ij}V(S)
=
V(S\cup\{i,j\})
-V(S\cup\{i\})
-V(S\cup\{j\})
+V(S).
\]

直观上：

- `Δ_ij > 0`：二者联合后出现额外价值，表现为 complementarity / synergy；
- `Δ_ij < 0`：二者边际作用相互抵消，表现为 redundancy / substitution / diminishing returns；
- `Δ_ij ≈ 0`：接近可加。

再按不同 coalition `S` 进行 Shapley-style 加权平均，就得到整体 interaction index。

---

## 6.2 推荐的理论工具

可优先考虑两类：

### 方案 A：Shapley-Taylor

优点：

- 经典；
- ICML 2020；
- 有明确公理；
- 可将 attribution 扩展到最高 `k` 阶交互。

### 方案 B：Faith-Shap

优点：

- JMLR 2023；
- 直接将 Shapley 视为 coalition value function 的线性近似系数；
- 再扩展到 `l` 阶多项式近似；
- 很适合本文“将复杂集合函数解析为低阶可解释结构”的故事。

推荐：

> **论文主方法可以优先选择 Faith-Shap 或 Shapley-Taylor 中一种，另一种作为 related work / robustness baseline。**

不建议同时发明新的 interaction index，除非后续确实发现现有定义不适合任务。

---

# 7. 关于“Shapley 是一阶加性压缩”的严谨说法

可以写：

\[
u_1(S)
=
\beta_0+
\sum_i\beta_i z_i,
\]

其中：

\[
z_i=\mathbb 1(i\in S).
\]

但必须强调：

> 这是对复杂集合函数 `V(S)` 的一阶归因 / 加性近似，而不是一般情况下对 `V(S)` 的精确重构。

普通 Shapley 可以把 interaction 产生的总价值公平地分到参与字段，但**会丢失 interaction 的来源结构**。

因此二阶近似：

\[
u_2(S)
=
\beta_0
+
\sum_i\beta_i z_i
+
\sum_{i<j}\beta_{ij}z_i z_j
\]

保留：

- 单字段成分；
- 成对 interaction 成分。

注意：进入二阶 Shapley-Taylor / Faith-Shap 后，一阶项通常会按对应理论重新定义，不能简单理解为“普通 Shapley 再额外加一个 pair term”。

---

# 8. 本文可以做出的几个理论结果

下面这些理论点与当前框架天然兼容，建议认真做。其中 8.1–8.3 很适合写成 Proposition / Theorem。

## 8.1 `V*` 的单调性

若允许的预测函数类足够丰富，且 `S⊆T`，则：

\[
V^*(S)\le V^*(T).
\]

原因：使用更多字段时，最优预测器至少可以选择忽略新增字段，因此最优性能不应下降。

这给通用模型提供一个非常自然的结构先验。

---

## 8.2 在 `F_theta` 中加入 monotonicity regularization

可定义：

\[
L_{mono}
=
\mathbb E_{S,i\notin S}
\left[
\max(0,F_\theta(S)-F_\theta(S\cup\{i\}))
\right].
\]

最终：

\[
L=L_{value}+\lambda_{mono}L_{mono}.
\]

作用：

1. 使模型更符合“最佳推断能力”的理论语义；
2. 减少不合理的 `V(S)>V(S+i)`；
3. 可能提高 Shapley / interaction 的稳定性；
4. 为方法部分增加一个与问题结构相关，而不是纯工程性的设计点。

注意：有限样本 cross-validation `R²` 标签本身可能不严格单调，因此不能简单硬约束到所有标签完全单调。可优先使用软约束。

---

## 8.3 surrogate 误差向 Shapley 误差的传播界

假设对所有 coalition：

\[
|F_\theta(S)-V^*(S)|\le \epsilon.
\]

则任意字段 `i` 的边际贡献误差满足：

\[
\begin{aligned}
&|[F(S+i)-F(S)]-[V^*(S+i)-V^*(S)]|\\
&\le |F(S+i)-V^*(S+i)|+|F(S)-V^*(S)|\\
&\le 2\epsilon.
\end{aligned}
\]

由于 Shapley 是这些边际贡献的凸加权平均，因此：

\[
\boxed{
|\widehat\phi_i-\phi_i^*|\le 2\epsilon
}
\]

这是一个简单但非常有用的结论：

> **只要 coalition value function 的统一逼近误差可控，字段级 Shapley 误差也有直接上界。**

这可以成为本文从“value model fidelity”到“field attribution reliability”的理论桥梁。

---

## 8.4 pairwise interaction 的误差传播界

二阶离散差分包含 4 个 coalition value：

\[
\Delta_{ij}V(S)
=
V(S+ij)-V(S+i)-V(S+j)+V(S).
\]

若每个 value 的绝对误差不超过 `epsilon`，则：

\[
\boxed{
|\Delta_{ij}F(S)-\Delta_{ij}V^*(S)|\le4\epsilon
}
\]

对这些差分做任意非负归一化加权平均后，同样可获得 interaction attribution 的误差上界。

这非常适合写进理论部分。

---

## 8.5 Shapley efficiency

若采用标准 Shapley：

\[
\sum_i\phi_i
=
V(N)-V(\emptyset).
\]

因此字段贡献具有天然的整体守恒解释：

> 所有字段 Shapley 贡献之和等于“全字段总体推断能力”相对“无字段基线”的增量。

这比 mean absolute SHAP 更容易直接解释为整体推断能力的分摊。

---

# 9. 一个重要边界：不能直接继承 SPVIM 的统计置信区间

Williamson & Feng 的 SPVIM 给出了严格的渐近统计推断理论。

但如果本文改成：

\[
V^*(S)
\rightarrow
\widetilde V(S)
\rightarrow
F_\theta(S)
\rightarrow
\widehat\phi_i,
\]

中间多了一层 learned surrogate error。

因此不能直接说：

> “因为用了 SPVIM 的采样方法，所以我们的 Shapley 也自动具有相同的 CI / p-value 理论保证。”

若要做严格 inference，需要同时处理：

- 有限数据误差；
- subset sampling error；
- predictor estimation error；
- surrogate approximation error。

建议当前论文：

1. 不把统计 CI 作为必要主贡献；
2. 用 surrogate error bound + bootstrap / repeated training 做稳定性分析；
3. 若后续理论精力足够，再单独研究 surrogate-aware inference。

这样更稳妥。

---

# 10. 为什么暂时不研究“完整路径搜索”

即使 `F_theta(S)` 的一次 query 极快，集合空间仍然是：

\[
2^p.
\]

通用 value oracle 解决的是：

\[
\text{query cost},
\]

而不是自动解决：

\[
\text{combinatorial search complexity}.
\]

贪心搜索可以快速找到一条高性能路径，但无法保证：

- 找到所有路径；
- 找到所有极小真点；
- 给出完备性证书。

因此本文如果强行加入路径，很容易让理论主线变散。

推荐：

> **本文聚焦 field-level global contribution 与 interaction-level structure；完整路径枚举/认证作为独立后续研究。**

---

# 11. 推荐的论文故事线

## 11.1 问题动机

现有字段重要性方法存在两个层面的错位：

### (1) 全量模型 SHAP 更像“模型依赖”

一个高准确率模型可能：

- 在冗余字段中任意选择一个；
- 忽略另一个实际同样有推断能力的字段；
- 将高相关字段的作用以与训练过程相关的方式分摊。

因此：

\[
\text{high prediction accuracy}
\not\Rightarrow
\text{correct population inference attribution}.
\]

### (2) population predictive importance 很合理，但 coalition evaluation 昂贵

SAGE / SPVIM 提供了理论依据，但当 `V(S)` 需要重新训练预测器时，大规模 coalition analysis 仍然昂贵。

---

## 11.2 本文核心切入点

提出：

> **Amortized Inference Value Model**

学习：

\[
F_\theta(S)\approx V^*(S).
\]

将昂贵的 subset-specific retraining 摊销为统一函数查询。

---

## 11.3 在同一个 value model 上完成两级解析

### 字段级

\[
F_\theta
\xrightarrow{\text{Shapley}}
\phi_i
\]

得到字段总体推断贡献。

### interaction 级

\[
F_\theta
\xrightarrow{\text{Shapley-Taylor/Faith-Shap}}
\phi_{ij}
\]

得到：

- synergy；
- redundancy；
- substitution / diminishing returns。

最终不是只得到一个 feature ranking，而是得到一个：

> **field-level inference contribution + interaction structure**

的统一结果。

---

# 12. 推荐的“贡献点”写法

下面是比较稳妥的贡献结构。

## Contribution 1：推断能力集合函数的摊销建模

不是逐个 subset 反复训练预测器，而是学习：

\[
F_\theta(S)\approx V^*(S),
\]

使任意 coalition 的推断能力可以快速查询。

如果现有通用模型本身已经是前一篇论文/已有工作，则新论文中需要明确：

- 哪部分属于既有模型；
- 本文新增了什么训练约束、multi-target 机制或 attribution framework；
- 避免仅把已有模型作为黑盒后接 Shapley 而显得创新不足。

---

## Contribution 2：从 coalition-level ability 到 field-level population contribution

基于 SAGE / SPVIM 的 population predictive importance 理论，将：

\[
V(S)
\]

通过 Shapley 分解为：

\[
\phi_i,
\]

用于表达字段在所有组合上下文中的平均推断贡献。

注意：Shapley 定义本身不是新贡献。

真正贡献应写成：

> 在所学习的 amortized inference value function 上进行低成本、可复用的 population-style field attribution。

---

## Contribution 3：从单字段贡献扩展到字段交互结构

利用成熟 interaction theory：

\[
\phi_{ij}
\]

区分：

- 独立强推断字段；
- 联合协同字段；
- 冗余/替代字段。

重要的是“统一 value function 支持多个解析算子”，而不是声称本文发明了 interaction index。

---

## Contribution 4：结构先验与 attribution 可靠性理论

推荐至少包括：

- monotonicity regularization；
- Shapley surrogate error bound：`≤2 epsilon`；
- interaction error bound：`≤4 epsilon`；
- 大规模实验验证 attribution 稳定性。

这是让文章从“工程组合”变成“较完整方法论文”的关键。

---

# 13. 推荐实验设计

论文实验最好不要只证明“最终排名看起来合理”，而是逐层验证每一个逻辑环节。

## Experiment 1：集合价值模型是否真的学准了？

在完全未参与训练的 masks 上评估：

- MAE / MSE；
- Pearson / Spearman；
- `R²(F_theta(S), V_tilde(S))`；
- 按 `|S|` 分层的误差；
- 极小集合 / 中等集合 / 大集合误差；
- 不同目标字段误差（若 multi-target）。

必须证明：

\[
F_\theta(S)
\]

不只是对训练 masks 有效。

---

## Experiment 2：monotonicity

统计：

\[
S\subset S+i
\]

时出现：

\[
F(S)>F(S+i)
\]

的比例与平均违反幅度。

对比：

- 无 monotonicity loss；
- 有 monotonicity loss。

同时确认加约束不会显著损害 value regression accuracy。

---

## Experiment 3：小规模“精确 Shapley”基准

选择较小字段数，例如：

\[
p=10\sim15,
\]

能够穷举所有：

\[
2^p
\]

coalitions。

得到近似 ground-truth：

\[
\phi_i^{exact}.
\]

对比：

- `F_theta + Shapley sampling`；
- SPVIM-style sampled estimator；
- SAGE；
- full-model global SHAP；
- FastSHAP（若任务定义可以合理对齐）。

指标：

- Shapley MAE；
- ranking correlation；
- top-k overlap；
- runtime。

这是最有说服力的实验之一。

---

## Experiment 4：专门构造“冗余字段”验证 SHAP 与 population attribution 的区别

例如：

\[
X_1=Z+\epsilon_1,
\quad
X_2=Z+\epsilon_2,
\quad
Y=Z+\eta.
\]

训练一个全量 DNN，可能主要依赖 `X1`。

比较：

1. mean |SHAP| of full model；
2. `Shapley(V(S))`。

理想结果：

- full-model SHAP 易受模型实际选择影响；
- population/game Shapley 更稳定地反映 `X1/X2` 均具有推断价值。

这个实验可以直接支撑论文动机。

---

## Experiment 5：interaction ground truth

设计几组可控 synthetic data。

### Additive

\[
Y=X_1+X_2+\epsilon.
\]

应有较弱 interaction。

### Pure synergy

\[
Y=X_1X_2+\epsilon
\]

或 XOR-like 构造。

期望：

\[
I_{12}>0.
\]

### Redundancy / substitution

\[
X_2\approx X_1,
\quad
Y=X_1+\epsilon.
\]

期望 pair interaction 体现 diminishing returns / redundancy。

验证：

- interaction sign；
- interaction ranking；
- 对 noise / correlation 强度的敏感性。

---

## Experiment 6：真实数据上的字段总体贡献与 interaction 图

最终输出可以包括：

- Top-k field contribution；
- contribution heatmap；
- pairwise interaction matrix；
- synergy network；
- redundancy network；
- 不同目标字段的贡献矩阵。

重点解释少量代表性字段关系，不要堆大量图而缺乏验证。

---

## Experiment 7：效率实验

分别测：

### 传统 subset-specific evaluation

\[
S\rightarrow\text{训练 predictor}\rightarrow R^2
\]

### 通用 value model

\[
S\rightarrow F_\theta(S)
\]

报告：

- 单 coalition 查询时间；
- 1k / 10k / 100k coalition 查询时间；
- 总 Shapley 计算时间；
- interaction computation time；
- GPU batch throughput。

重点强调：

> 本方法的优势在于“底层 value query 被摊销”，因此同一模型可被多个下游分析重复使用。

---

# 14. 推荐 baseline

至少考虑：

1. **Full-model mean |SHAP|**
   - 代表 model reliance / model attribution。

2. **Permutation Feature Importance**
   - 传统模型依赖型 global importance。

3. **SAGE**
   - 最重要的 global predictive importance baseline。

4. **SPVIM sampled estimator**
   - 若实现成本可接受，用于 population Shapley 对比。

5. **FastSHAP**
   - 作为 amortized Shapley literature 参考；
   - 但任务定义不同，不应硬做完全一一对应的性能对比。

6. 项目已有 feature-selection / gating / NMI / Pearson 等方法
   - 若论文需要与既有推断源识别线对接，可作为 supplementary baseline。

---

# 15. 论文结构建议

## 1 Introduction

建议按以下逻辑：

1. 字段级推断能力对于数据关联/隐性推断分析很重要；
2. 全量模型 SHAP 主要解释模型依赖，并不等价于字段总体潜在推断能力；
3. population predictive importance 更符合目标，但需要大量 coalition evaluation；
4. 现有方法通常直接估计某个 importance，而本文选择摊销更底层的 `V(S)`；
5. 一个 value model 同时支持字段贡献和 interaction 分析；
6. 给出贡献点。

---

## 2 Related Work

建议分四部分：

### 2.1 Model explanation and SHAP

说明普通 SHAP 的研究对象主要是预测模型输出。

### 2.2 Global / population predictive importance

重点介绍：

- SAGE；
- SPVIM。

### 2.3 Amortized Shapley estimation

重点介绍：

- FastSHAP。

然后明确：

> FastSHAP 摊销 Shapley explanation；本文摊销底层 coalition value function。

### 2.4 Interaction attribution

介绍：

- Shapley Interaction；
- Shapley-Taylor；
- Faith-Shap。

---

## 3 Problem Formulation

明确：

\[
V^*(S)
=
\frac{\operatorname{Var}(\mathbb E[Y\mid X_S])}
{\operatorname{Var}(Y)}.
\]

然后定义：

- finite-sample oracle `V_tilde(S)`；
- amortized estimator `F_theta(S)`；
- Shapley field inference contribution；
- interaction contribution。

---

## 4 Method

### 4.1 Coalition-value label generation

说明如何获得训练 masks 与 `V_tilde(S)`。

### 4.2 Amortized inference value model

网络输入、mask 编码、target encoding、训练目标。

### 4.3 Monotonicity regularization

加入问题结构先验。

### 4.4 Field-level Shapley inference contribution

说明 sampling / weighted regression / permutation estimator。

### 4.5 Interaction analysis

选择 Shapley-Taylor 或 Faith-Shap。

### 4.6 Error propagation analysis

给出：

\[
|\hat\phi_i-\phi_i|\le2\epsilon
\]

与 pairwise interaction 的 `4 epsilon` bound。

---

## 5 Experiments

建议顺序：

1. `F_theta` 是否学准；
2. monotonicity；
3. Shapley accuracy；
4. synthetic redundancy / synergy；
5. real-world attribution；
6. efficiency；
7. ablation。

这样的顺序比直接先给真实数据结果更严谨。

---

## 6 Discussion

主动说明：

- `V*` 无法直接观测；
- `F_theta` 的意义取决于 label oracle 质量；
- 当前方法不是 causal attribution；
- 当前不解决所有 minimal inference paths；
- 严格 SPVIM-style CI 仍需额外 surrogate uncertainty theory。

审稿人反而会更容易接受。

---

# 16. 论文标题方向（暂定）

不要把 “Shapley” 放得过重，否则容易被理解成一个新的 SHAP 方法。

可考虑：

1. **Amortized Inference Capability Modeling for Global Feature Contribution and Interaction Analysis**

2. **Learning Set-Valued Inference Capability for Global Feature Attribution and Interaction Discovery**

3. **From Coalition Inference Capability to Field-Level Contribution: An Amortized Set-Function Approach**

如果需要强调电力数据场景，再加：

> ... for Power-System Data

或：

> ... in Energy Data Analytics

但建议方法标题优先保持通用。

---

# 17. 论文中应避免的过度声称

以下表述尽量不要使用：

### 不要说

> Shapley 就是真实因果贡献。

应说：

> Shapley 是相对于所定义的 predictive/inference value function 的公平总体归因。

---

### 不要说

> `F_theta` 输出的就是绝对真实的 `V*(S)`。

应说：

> `F_theta` 逼近由有限数据和强预测 oracle 估计得到的 population predictive capability，并通过 held-out coalition evaluation 验证其 fidelity。

---

### 不要说

> 本文首次用 Shapley 做 global predictive importance。

SAGE / SPVIM 已经做过。

---

### 不要说

> 本文解决了指数级组合搜索问题。

本文只降低 coalition query 成本，不等价于解决任意组合搜索。

---

### 不要说

> FastSHAP 只能做 Shapley，所以我们比它更完整。

更准确地说：

> FastSHAP 直接摊销某种 Shapley explanation，而本文摊销更底层的 coalition value function，因此一个 value model 可以支持多种下游 functional。

---

# 18. 当前最需要 Codex / Claude Code 检查现有项目的内容

请结合仓库实际代码逐项回答，不要直接重写整个项目。

## A. 当前 `V(S)` 标签到底是什么？

检查：

- 输入字段 mask 如何生成；
- 对每个 mask 是否重新训练预测器；
- predictor 是固定结构还是多算法；
- `R²` 是训练集、验证集还是测试集；
- 是否 cross-fitting；
- 是否存在数据泄漏；
- 是否对不同 mask 使用同一 train/test split。

这是最重要的一步。

---

## B. 当前 universal model 的输入输出是什么？

确认：

\[
S\rightarrow R^2
\]

还是：

\[
(t,S)\rightarrow R^2_t.
\]

检查：

- mask encoding；
- target encoding；
- 网络结构；
- loss；
- train/val split 是按 mask 还是按数据样本；
- 是否可能 memorization。

---

## C. 是否具备真正的 unseen-mask generalization？

必须确保测试集 masks：

- 从未用于 universal model 训练；
- 最好按 cardinality 分层；
- 包括 distribution-shift masks，例如训练时少见的 mask size。

否则不能说模型学到了集合函数。

---

## D. 加 monotonicity loss 的最小改法

尽量不要大改架构。

实现随机 pair：

\[
(S,S\cup i)
\]

然后附加：

```text
mono_loss = relu(F(S) - F(S_plus_i)).mean()
loss = value_loss + lambda_mono * mono_loss
```

观察：

- value MAE；
- monotonic violation rate；
- Shapley stability。

---

## E. 实现 Shapley 时优先使用通用模型批量 forward

对固定字段 `i`：

1. 采样背景 coalition `S`；
2. 构造 `S` 与 `S+i`；
3. 合并成大 batch；
4. 一次或少量 GPU forward；
5. 计算 marginal contribution；
6. 加权平均。

不要为每个 coalition 单独 Python for-loop forward。

---

## F. interaction 同样批量化

对 `(i,j)`：

需要：

- `S`；
- `S+i`；
- `S+j`；
- `S+i+j`。

将 4 组 mask 合并 batch forward，再计算：

\[
\Delta_{ij}(S).
\]

---

# 19. 推荐的开发顺序

不要一开始就同时做全部理论与实验。

## Phase 1：确认 value model 是否可靠

先只做：

- unseen-mask fidelity；
- cardinality-stratified error；
- monotonicity。

如果这一层不可靠，后面的 Shapley 全部没有意义。

---

## Phase 2：做小规模 exact Shapley sanity check

缩小到 `p≤15`，穷举 coalition。

验证：

\[
F_\theta\text{-Shapley}
\approx
\text{exact oracle Shapley}.
\]

这是整个方法最关键的 sanity check。

---

## Phase 3：interaction synthetic test

验证 synergy / redundancy 的符号与排名。

---

## Phase 4：真实数据实验

再做完整字段贡献、interaction 和效率结果。

---

# 20. 核心参考文献及其在本文中的作用

## [R1] Shapley, L. S. (1953)

**A Value for n-Person Games.**

作用：

- Shapley value 的原始合作博弈理论。

---

## [R2] Lundberg, S. M., & Lee, S.-I. (2017)

**A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.**

https://arxiv.org/abs/1705.07874

作用：

- SHAP；
- 说明 Shapley 在模型局部预测解释中的经典用法；
- 用来与本文的 population inference attribution 区分。

---

## [R3] Covert, I., Lundberg, S. M., & Lee, S.-I. (2020)

**Understanding Global Feature Contributions With Additive Importance Measures. NeurIPS 2020.**

https://papers.neurips.cc/paper/2020/hash/c7bf0b7c1a86d5eb3be2c722cf2cf746-Abstract.html

作用：

- SAGE；
- global feature contribution；
- model-based vs universal predictive power；
- 本文用 predictive capability 定义 Shapley game 的最重要理论依据之一。

---

## [R4] Williamson, B. D., & Feng, J. (2020)

**Efficient nonparametric statistical inference on population feature importance using Shapley values. ICML 2020.**

https://proceedings.mlr.press/v119/williamson20a.html

作用：

- SPVIM；
- population feature importance；
- `Theta(n)` subset sampling；
- 渐近最优估计；
- confidence interval / hypothesis testing；
- 支撑“字段总体 predictive importance 是一个有严格统计意义的问题”。

---

## [R5] Sundararajan, M., Dhamdhere, K., & Agarwal, A. (2020)

**The Shapley-Taylor Interaction Index. ICML 2020.**

https://proceedings.mlr.press/v119/sundararajan20a.html

作用：

- 将 attribution 从单字段扩展到 interaction；
- 支撑协同字段分析。

---

## [R6] Covert, I., Lundberg, S. M., & Lee, S.-I. (2021)

**Explaining by Removing: A Unified Framework for Model Explanation. JMLR 2021.**

https://www.jmlr.org/papers/v22/20-1316.html

作用：

- 系统区分 feature removal、模型行为、importance summarization；
- 帮助论文更严谨地区分 model explanation 与 population capability attribution。

---

## [R7] Aas, K., Jullum, M., & Løland, A. (2021)

**Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. Artificial Intelligence.**

https://doi.org/10.1016/j.artint.2021.103502

作用：

- 强调相关字段下普通 KernelSHAP / feature removal 的困难；
- 支撑本文“高相关、冗余字段下 model SHAP 不等于字段固有推断能力”的动机。

---

## [R8] Jethani, N., Sudarshan, M., Covert, I., Lee, S.-I., & Ranganath, R. (2022)

**FastSHAP: Real-Time Shapley Value Estimation. ICLR 2022.**

https://arxiv.org/abs/2107.07436

作用：

- amortized Shapley estimation；
- 必须讨论，避免将“神经网络加速 Shapley”误写为本文创新；
- 本文与其区别：`FastSHAP` 摊销最终 Shapley explanation，本文摊销底层 coalition value function。

---

## [R9] Tsai, C.-P., Yeh, C.-K., & Ravikumar, P. (2023)

**Faith-Shap: The Faithful Shapley Interaction Index. JMLR 2023.**

https://www.jmlr.org/papers/v24/22-0202.html

作用：

- 将 Shapley 解释为 coalition value function 的 faithful linear approximation；
- 推广到高阶 polynomial interaction；
- 非常适合作为本文“复杂集合函数 → 低阶可解释字段/交互结构”的理论支撑。

---

# 21. 我目前对论文潜力的判断

如果论文只是：

\[
\text{已有 universal model}
+
\text{Shapley}
\]

然后得到一个字段排名，创新性偏弱。

如果扩展成：

\[
\boxed{
\text{Amortized inference set-function learning}
}
\]

+

\[
\boxed{
\text{population-style field attribution}
}
\]

+

\[
\boxed{
\text{interaction decomposition}
}
\]

+

\[
\boxed{
\text{monotonic structural prior + attribution error theory}
}
\]

则整体会更像一篇完整的方法论文。

真正决定论文水平的不是“用了 Shapley”，而是：

1. `F_theta` 是否真的能够可靠泛化到 unseen coalitions；
2. 你的 coalition label 是否足够接近 `V*(S)`；
3. 你是否能证明 `F_theta` 的误差不会破坏 attribution；
4. interaction 是否能在 synthetic ground truth 上得到准确验证；
5. 相比重新训练式 population methods，效率优势到底有多大；
6. 在真实电力/能源数据上是否发现了传统 model SHAP 看不到的冗余与协同结构。

---

# 22. 最后给 Codex / Claude Code 的具体任务提示

请先阅读项目代码，不要立即重构。按以下顺序输出分析：

1. 找到当前所有与 `mask / subset / field set / R² / universal predictor` 相关的代码；
2. 画出当前数据流：`subset mask → label generation → universal model → evaluation`；
3. 判断当前训练标签对应本文中的 `V* / V_tilde / F_theta` 哪一层；
4. 检查是否存在训练/验证泄漏、mask 泄漏或 target leakage；
5. 给出 unseen-mask fidelity 的现有结果或补充实验方案；
6. 判断最小代价加入 monotonicity regularization 的位置；
7. 设计 batch Shapley estimator，优先复用现有 GPU pipeline；
8. 设计 pairwise interaction estimator；
9. 给出一个 `p≤15` 的 exact-enumeration sanity-check 实验；
10. 给出运行时间、显存占用和复杂度估计；
11. 不要实现完整路径枚举；
12. 最后基于实际代码判断：本文最可靠的 2–4 个创新点分别是什么，哪些当前想法在代码层面还不成立。

---

# 23. 当前建议的最简方法框架图（文字版）

```text
原始数据 (X, Y)
      │
      │ sampled field subsets S
      ▼
强推断器 / cross-fitting
      │
      ▼
有限样本 coalition labels
V_tilde(S) ≈ V*(S)
      │
      ▼
Amortized Inference Value Model
F_theta(S) ≈ V*(S)
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
Shapley attribution      Interaction attribution
phi_i                    phi_ij
      │                      │
      ▼                      ▼
字段总体推断贡献         synergy / redundancy
      │                      │
      └──────────┬───────────┘
                 ▼
     字段级整体推断结构解析
```

---

# 24. 最核心的论文表述

如果后续只保留一句话，可以使用下面这个版本：

> 本研究不直接解释某个固定预测模型对输入字段的依赖，而是首先学习“任意字段集合能够对目标字段提供多少总体推断能力”的集合价值函数，并将昂贵的子集级模型训练摊销为统一的快速价值查询；在此基础上，利用 Shapley value 对集合级推断能力进行字段级总体贡献分解，并利用 Shapley interaction 类方法进一步解析字段之间的协同与冗余关系，从而形成由集合能力、字段贡献到字段交互的一体化推断结构分析框架。

