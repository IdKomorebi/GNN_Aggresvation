# DNN_Aggresvation54 日志

## Modify by Claude: 2026-07-01

## 1. 子项目定位（去风险实验）

决定要不要做"推断驱动的敏感度扩散"(confidential 种子=1 往外扩散)之前,先用最小代价
验证一个问题:**纯图扩散能否揭示直接归因/直接相关性漏掉的"间接泄露",并被推断模型证实?**
若能→扩散有存在理由,值得做;若不能→直接归因已足够,安心写核心版,不碰扩散。

## 2. 方法
- 关联先验图(avg 相关系数 × 模型 bipartite 图结构,对称)。confidential 按推断准确率
  a_c 作种子,PPR 往外扩散(无参数,只有 restart β,扫了 0.5/0.7/0.85)。
  - direct d_g = 1 跳:g 与 confidential 的直接相关连接。
  - diffused s_g = 多跳 PPR:g 经 general-general 间接连到 confidential 的可达性。
- 推断模型(gcn_dynamic, R²=0.8746)当 oracle 给"真实泄露"参照:mask(necessity)、
  single(sufficiency)。
- 判据:排名一致性 + top-k 推断 oracle(按各排名取 top-k general 测 12-conf R²)+
  被扩散提拔的字段是否被模型证实泄露。

## 3. 结果(决定性负面）

| 对比 | 数值 | 解读 |
|---|---:|---|
| Spearman(direct, diffused) | **0.989** | 扩散几乎不改变直接相关性排名 |
| Spearman(diffused, mask) | 0.750 | **< direct↔mask 0.797**：扩散后更不像真实泄露 |
| Spearman(diffused, single) | 0.737 | **< direct↔single 0.786**：同上 |
| top-k 前10平均 direct | 0.292 | **> diffused 0.280**：扩散挑字段更差 |
| top-k 前10平均 mask/single(模型) | 0.386/0.389 | 模型归因远胜任何纯图法 |

被扩散最提拔的字段(low direct→high diffused):total_pjm_reg_purchases、gen_fuel_gas_pct、
rmccp 等,其模型 mask/single 全 ≈0(single 中位数仅 0.0139)——**提拔的全是不泄露的尾部
字段,扩散没捞出任何被证实的间接泄露。**

## 4. 结论与原因

**扩散不带来任何增量,甚至略有害**:
1. 它几乎不改变直接相关性排名(0.989);
2. 改动的地方反而**更不**符合模型实测泄露(一致性 0.75<0.80,top-k 0.28<0.29);
3. 没有发现一个"直接低、间接高、且被模型证实泄露"的字段。

**根因**:推断模型本身就是多跳的(4 层 GCN 传播),**mask/single 归因已经把间接泄露
算进去了**(遮蔽 g 会连带去掉它经其它 general 节点对 confidential 的间接影响)。所以
"间接泄露"已经被推断模型内部捕捉,外挂一个图扩散是**重复且会稀释信号**(把尾部噪声平
摊进来),自然只会变差。"真正只靠间接路径泄露"的字段在本数据里也不存在(提拔字段
single≈0)。

## 5. 对项目决策的意义
- **不要做敏感度扩散模型**(静态已无增量;动态版还多了无 label 监督的坑,更不值)。
- **客观敏感度 = 直接的推断能力归因**(mask/shapley/single),它已隐含多跳间接泄露,
  这本身就是相对 IGNN"主观先验+注意力扩散"的干净论据:**不需要扩散,推断模型即已编码
  传播,直接归因即得到正确敏感度。**

## 6. 输出
```
outputs/topk_oracle.png        graph-only(direct/diffused) vs model-based 的 top-k 曲线
outputs/diffusion_scatter.png  direct vs diffused 排名散点(颜色=模型 single 实测泄露)
outputs/diffusion_vs_direct.csv  逐字段 direct/diffused/mask/single
outputs/summary.json
inference_model/model.pt       复用 DNN52 gcn_dynamic(mlp) 作 oracle
```
