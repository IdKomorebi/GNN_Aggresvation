# DNN_Aggresvation47 日志

## Modify by Claude: 2026-06-26

## 1. 子项目定位

推断驱动敏感度：在训练好的 multi graph-combo 模型上，用 **7 种归因方法**算每个
general 字段的"推断能力/敏感度"分数，每种独立标准化(max=1)+排名，做一致性分析。
载体模型自训（DNN43 最优配置，mean R²≈0.86）。

## 2. 七种方法（均用推断准确率 a_c = per-confidential test R²≥0 加权）

| 方法 | 原理 | 衡量 |
|---|---|---|
| mask | 遮蔽 general j → 各 confidential loss 增幅 | 必要性 |
| single | 只留 general j → 对各 confidential 推断 R² | 充分性 |
| attn | 逐层 general→confidential 注意力(含 gate 混合先验)对 j 平均权重 | 路径(简化) |
| ig | Integrated Gradients：a_c 加权输出对 general 输入积分梯度幅值 | 梯度归因 |
| shapley | 蒙特卡洛 Shapley：value(S)=只用 S 推断的 a_c 加权 R² | 公平边际(化解冗余) |
| **gate** | **输入门控 g 与模型联合训练 + L1 稀疏，训练后 |g| 为分数** | 不可替代性(最小集合) |
| **lrp** | 节点级 LRP-0：冻结注意力(固定路由)，relevance=输入×梯度沿推断图回流 | 路径归因 |

代码：`compute_sensitivity.py --method ...`；gate 由 `multi_graphcombo_gate` 配置
（`use_input_gate=True`, `gate_l1_lambda=0.01`）联合训练后读 `input_gate`。
`aggregate_sensitivity.py` 自动汇总所有方法。

## 3. 核心发现：六方法高度一致，gate 独立离群

**一致性 Spearman（节选）**：mask-lrp 0.95、ig-lrp **0.97**、mask-shapley 0.95、
attn-lrp 0.92、mask-ig 0.94 …… **mask/single/attn/ig/shapley/lrp 六个互相 0.83~0.97**；
**gate 与所有方法仅 0.54~0.68（明显离群）**。

1. **lrp ≈ ig（0.97，最高相关）**——印证了"节点级 LRP 本质≈梯度类方法"。冻结注意力
   后整网在输入上分段线性，input×grad 就是 LRP-0，和 IG 几乎一样。**所以节点级 LRP
   没带来独立信息；LRP 真正的独特价值在 walk-level（路径分解），那是后续工作。**

2. **gate 是唯一独立视角（联合稀疏选择）**。它测"不可替代性/最小充分集合"：L1 把
   "可被替代"的字段门控压向 0。所以 system_energy_price_da 在其他 6 法都排前列，
   gate 却排第 14——gate 认为它可被替代（冗余信号）。这与边际/梯度归因正交，是
   论文里"必要性≠不可替代性"的好论点。

3. **caveat**：gate 联合训练 L1=0.01 偏强，g 被压得很小（max 0.092、mean 0.015，
   且 g 与 input_proj 权重耦合 → 绝对值失真），相对排名仍有信号、且 gate 离群在
   "固定模型版(0.72~0.77)"与"联合训练版(0.54~0.68)"两次都稳定出现，是真现象；
   但若要 g 绝对值可解释，应调小 L1（如 0.001）让门控更分化。

## 4. 综合敏感度排名（7 方法平均排名 top）

| general 字段 | avg_rank |
|---|---:|
| forecast_load_mw_latest_available | 2.86 |
| system_energy_price_da | 4.29 |
| da_as_nsr_mw_primary_reserve | 5.14 |
| gen_fuel_nuclear_mw | 5.86 |
| forecast_load_mw_day_ahead | 6.00 |
| da_as_as_req_mw_synchronized_reserve | 8.00 |

最高泄露源：负荷预测、系统能价、备用容量、燃料发电——符合电力系统直觉。

## 5. 结论与下一步

- 7 方法里 6 个高度一致 → 敏感度排名极稳健（不同原理都指向同一批字段）。
- **节点级 LRP≈IG，要兑现 LRP "推断路径"的独特卖点必须做 walk-level GNN-LRP**
  （relevant walk search），这是真正的核心创新、下一步主攻。
- gate 提供唯一正交视角（不可替代性），揭示冗余；可调 L1 让其更可解释。

## 6. 输出
```
outputs/sensitivity/
├── {mask,single,attn,ig,shapley,gate,lrp}_raw.csv  各方法原始分数
├── sensitivity_ranking.csv      7 方法 norm 分数 + 排名（独立标准化 max=1，按 avg_rank 排序）
├── consistency_spearman.csv/png 一致性矩阵（单独图）
└── sensitivity_table.png        44 字段 × 7 方法大表格（按 avg_rank 降序，每格 0-1 分数）
outputs/tuning/multi_graphcombo/<ts>/        敏感度载体模型（6 方法用）
outputs/tuning/multi_graphcombo_gate/<ts>/   gate 联合训练模型
```
