# 通用 oracle、廉价适配与高阶搜索：文献映射和下一步

## 一、先把问题拆成三个不同目标

当前项目把三个目标混在一个 MLP 上：

1. **任意集合的廉价粗估**：最好一次前向；
2. **少数关键集合的高精度值**：允许比一次前向贵，但不能每个集合 K=25；
3. **组合爆炸中的发现**：不要求列出 `2^44` 个答案，而是尽量不漏危险协同或最小危险集。

一个方法很难同时最优。建议把论文方法明确写成多保真攻击者组合，而不是继续寻找一个
“所有集合都精确的一次前向模型”。

## 二、文献告诉我们的不是“换个模型就行”

### 1. 任意条件预测已有成熟范式

- [Knockout（TMLR 2025）](https://openreview.net/forum?id=K71y5pge84) 与当前随机替换/
  随机失活思想最接近；
- [ACFlow（ICML 2020）](https://proceedings.mlr.press/v119/li20a.html) 直接学习任意
  `p(x_u | x_o)`；
- [Set Transformer（ICML 2019）](https://proceedings.mlr.press/v97/lee19d.html)
  用注意力建模集合元素间交互，而不是依赖固定相关图。

它们说明“一个模型接任意观测子集”不是空白。ACFlow/条件生成模型可以作为更强
conditional baseline，但对本项目不是第一优先：R² 的 Bayes 最优点预测仍是条件均值，
生成完整条件分布会带来更大训练和 Monte Carlo 成本，未必解决 rare-task 摊薄。

### 2. 快速适配不等于 Reptile

- [CAVIA（ICML 2019）](https://proceedings.mlr.press/v97/zintgraf19a.html) 只更新低维
  context 参数；
- [R2D2/闭式 solver（ICLR 2019）](https://arxiv.org/abs/1805.08136) 把岭回归等
  快速解算器放进元学习内部；
- [ALFA（NeurIPS 2020）](https://proceedings.neurips.cc/paper_files/paper/2020/hash/ee89223a2b625b5152132ed77abbcc79-Abstract.html)
  学的是逐层、逐步学习率和衰减，而不是只学初始化；
- [Amortized Proximal Optimization（NeurIPS 2022）](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3af25aa3de8b7b02ddbd1b6be5031be8-Abstract-Conference.html)
  学低开销预条件器。

因此 D65 Reptile 失败只否定“一个更好的全网初始化”；D66 FiLM 失败只否定“mask 直接
生成调制参数”。它们没有验证“共享表示 + 每集合闭式 head”或“只适配几十个 context/
residual 参数”。

### 3. 高阶空间不能靠每个候选固定 K

- [Successive Halving（AISTATS 2016）](https://proceedings.mlr.press/v51/jamieson16.html)
  与 [Hyperband（JMLR 2018）](https://www.jmlr.org/beta/papers/v18/16-558.html)
  的核心是只给仍有希望的候选追加预算；
- [BOCS（ICML 2018）](https://proceedings.mlr.press/v80/baptista18a) 与
  [COMBO（NeurIPS 2019）](https://proceedings.neurips.cc/paper/2019/hash/2cb6b10338a7fc4117a80da24b582060-Abstract.html)
  用组合空间 surrogate 主动选择昂贵评估点；
- [稀疏 Möbius 恢复（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/520b379123d16e41f85472e766846486-Abstract-Conference.html)
  在低阶、稀疏假设下把指数系数恢复降到与非零项数相关的查询复杂度；
- [SPEX（ICML 2025）](https://proceedings.mlr.press/v267/kang25a.html) 用稀疏 Fourier
  恢复在长输入中找交互。

Möbius/Fourier 交互不等于本项目
`syn(S)=v(S)-max_i v(S\\{i})`，不能直接替换定义；但非常适合做候选生成，再用本项目
定义认证。

## 三、推荐的最终框架

### 层 0：一次前向的全局地图

保留 uniform MLP K0，输出：

- `v_K0(S,c)`；
- S1：同 conf 的 parent-child 差；
- S2：any-conf 的 parent 绝对值；
- 多 seed / dropout disagreement 作为不确定性之一。

它负责便宜扫描和排序，不再被叫作最终答案。

### 层 1：零梯度结构化解算

对低阶集合或候选集合构造固定字典：

- raw；
- square/product；
- 稳健 ratio；
- 后续可加领域公式候选，但必须在训练集内选择，不能看 test。

用岭回归闭式求解多 confidential head。下一版应预计算每个数据切分上的充分统计量：

`G = ΦᵀΦ, h_c = Φᵀy_c, y_cᵀy_c`

查询集合 S 时只抽取支持落在 S 内的行列，直接解小矩阵，并通过二次型算 test SSE；
无需再次扫描 4,589 行。高阶只保留在二/三阶阶段证实的重要交互列，避免字典随
`|S|²` 无限制增长。

### 层 2：少参数残差适配

对仍有分歧的候选测试两条，而不是再次做 Reptile：

1. CAVIA 式 context：冻结主体，每个 `(S,c)` 只更新 16–64 个 context 参数；
2. R2D2 式 learned embedding + closed-form head：训练共享特征，让闭式 head 的
   1 次求解更强。

训练目标应是少量已认证集合上的**适配后 regret**，不是原始预测 MSE。D83 已证明
“原始 MSE 大”会把不可预测噪声误认为值得加权的 hard task。

### 层 3：主动保真分配

不对所有候选做 K20。建议：

1. 所有可枚举候选：K0；
2. S1、S2、结构化 solver 任一路高，或三者分歧大：进入候选；
3. 先 K=1/5，只保留仍可能越过阈值者；
4. 边界附近 K=25；
5. 最终 top 与随机审计样本重训。

预算晋级标准同时看：

- 距离报警阈值；
- 模型间 disagreement；
- 统计上界；
- 对最终协同图/防护集结论的影响。

### 层 4：不枚举高阶

同时跑三条互补搜索：

1. **层级 beam + 随机探索**：强低阶节点扩展，保留纯高阶随机通道；
2. **稀疏 Möbius/Fourier 候选生成**：先恢复重要 support，再按项目 syn 定义认证；
3. **最小危险集剥离**：从高泄露集合反复删除不必要字段，多次随机重启，直接产出
   防护问题需要的超边。

若目标是“找到最危险的一部分”，可用 BOCS/COMBO；若目标是“给任意查询一个近似值”，
则单独训练：

`r_c(S) = v_portfolio(S,c) - v_K0(S,c)`

的 residual set-function surrogate。它直接拟合审计标量，而不是再次绕回预测 Y；
用主动学习选新重训点，并输出不确定区间。

## 四、实验优先级

1. **立即做**：缓存结构化充分统计量，复现 D83 数值并报告实际 query/s；
2. **立即做**：把真值统一改成攻击者 portfolio，下游协同图重新确认；
3. **高优先**：CAVIA context 与 learned embedding + ridge head，比较 K0/K1/K5；
4. **高优先**：在现有 2,197 真值三元组上离线模拟 Successive Halving，画
   recall–GPU-second 曲线；
5. **中优先**：训练直接 residual set-function，严格按 triple 分 tune/test；
6. **中优先**：验证稀疏 Möbius 候选生成对 PJM 的 recall–query 曲线；
7. **较低优先**：Set Transformer、ACFlow；
8. **停止投入**：更多 size 比例、原始 MSE-CVaR、只增宽共享 MLP。

## 五、必须保留的论文边界

- 一次前向模型优化的是随机 mask 下平均预测风险，不等于逐集合 `sup_f`；
- 重训 MLP 只是指定攻击族的下界，不能叫数学真值；
- 稀疏 Möbius 的复杂度优势依赖稀疏/低阶假设，必须在当前数据上实证；
- beam/Apriori 会漏纯高阶，必须保留无偏随机认证通道；
- D83 目前只有一个数据集和一次结构化内部切分，属于很强的机制证据，还不是跨数据集结论。
