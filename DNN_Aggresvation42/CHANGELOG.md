# DNN_Aggresvation42 日志

## Modify by Claude: 2026-06-25

## 1. 子项目定位

DNN40 在 window=1、修正信息集后，GNN single 推断精度（mean R² 0.8418）落后于
强 baseline DNN probe（0.8937），缺口 0.052。**当前参数是 window=4 时代调的，
随机划分 + window=1 后数据变干净，参数需要重调**。本项目针对 GNN 专属参数
重新调参，目标：在**不引入图外旁路、推断仍全程经过图**（保住后续"推断路径→
敏感度"方案）的前提下，把 GNN single 精度拉近 probe。

## 2. 诊断（先定位问题，再调参）

读 DNN40 训练日志：
- **价格类字段欠拟合**：congestion_price_da 连训练集都拟合不动（train MSE 0.65、
  R²≈0.35），train/test gap 还很小——是**表达力不足**，不是过拟合。
- **难字段早停过早**：congestion 类 epoch 80~100 就被 patience=80 早停，net_actual
  跑满 300。
- **alpha 过快塌缩**：edge_alpha_temperature=0.35 + alpha_lr_multiplier=12 让
  general-general 边权迅速塌到单一相关性度量。

→ 根因是欠拟合，调参方向应是**增表达力 + 充分训练**，不是加正则。

## 3. 单因素扫描（congestion_price_da，最难、缺口最大）

baseline 0.4559，probe 上界 0.660。每个配置只改一项：

| 配置 | Δ vs base | 结论 |
|---|---:|---|
| input_encoder=mlp | **+0.048** | 增输入非线性，最有效 |
| 充分训练 epochs500/patience150 | **+0.035** | 早停过早被证实 |
| attention_temperature 0.8→0.4 | +0.013 | 注意力要更 sharp |
| lr 1e-3 | +0.012 | |
| alpha_lr_multiplier 12→2 | +0.011 | 稳住 alpha 塌缩 |
| attention_temperature→1.5 | **−0.097** | 注意力太 soft 大崩 |
| num_layers→2 | −0.065 | 需要深度 |
| edge_alpha_temperature↑ | −0.04/−0.03 | 不该提高（塌到 pearson 本就对） |
| hidden_dim→128 | −0.018 | 单字段小数据，加宽过剩 |
| 去 gate / 纯 prior | −0.013/−0.007 | **先验有用**（反驳"先验没用"猜想） |

关键发现：① 欠拟合诊断证实（最有效的全是增表达力/充分训练）；② attention
温度要**更 sharp**、edge_alpha 温度**不要动**；③ 增容量靠 **mlp 编码而非加宽**；
④ hybrid gated 先验对价格类有正面作用。

## 4. 组合验证（难字段）

combo = mlp + 充分训练 + attention_temperature 0.4(后 0.25) + lr 1e-3 + alpha_lr 2：

| 字段 | baseline | combo | Δ | probe |
|---|---:|---:|---:|---:|
| congestion_price_da | 0.456 | **0.565**(attn0.25) | **+0.109** | 0.660 |
| marginal_loss_price_da | 0.769 | **0.841** | +0.072 | 0.882 |
| congestion_price_rt | 0.608 | 0.605 | −0.004 | 0.670 |

叠加基本成立；attention 0.25 比 0.4 又 +0.021（越 sharp 越好）；congestion_rt
不吃这套（瓶颈不同，留待后续）。

## 5. 全 12 字段推广（最终结果）

combo 配置 = `input_encoder=mlp`、`attention_temperature=0.25`、
`alpha_lr_multiplier=2`、`lr=1e-3`、`epochs=500`、`patience=150`
（其余沿用 DNN40：hidden_dim=64、num_layers=3、edge_alpha_temperature=0.35、
bipartite、window=1）。

| 指标 | DNN40 baseline | DNN42 combo | probe 上界 |
|---|---:|---:|---:|
| **mean R²** | 0.8418 | **0.8688 (+0.0270)** | 0.8937 |
| GNN–probe 缺口 | 0.0519 | **0.0249（减半）** | — |

逐字段（详见 `outputs/tuning_comparison.csv`、`outputs/tuning_comparison.png`）：
- **12 个字段全部不降**（最差 total_lmp_da −0.0005，可忽略）——激进设置没拖累
  容易字段。
- 大改善：congestion_da +0.109、gross_actual +0.056、net_actual +0.043、
  marginal_loss +0.038、total_losses +0.033、thirty_min +0.030。
- 剩余缺口集中在价格类：congestion_da −0.096、marginal_loss −0.075、
  congestion_rt −0.054（字段本身难，probe 也不高）。

## 5.1 multi 模式同样验证（combo 直接迁移）

把同一套 combo 参数用到 multi（一次训 12 字段）：

| 指标 | DNN40 multi baseline | DNN42 multi combo | probe 上界 |
|---|---:|---:|---:|
| **mean R²** | 0.8096 | **0.8415 (+0.0319)** | 0.8937 |
| multi–probe 缺口 | 0.0840 | **0.0521（缩 38%）** | — |

- multi 提升幅度（+0.032）甚至略大于 single（+0.027）；12 字段仅 synchronized_reserve
  微降 0.002，其余全升。大改善：congestion_da +0.131、congestion_rt +0.069、
  net_actual +0.050、thirty_min +0.048。
- 详见 `outputs/tuning_comparison_multi.csv`、`outputs/tuning_comparison_multi.png`。
- 说明 combo 参数对 single/multi 都稳健，window=1 后重调参的收益是普适的。

## 6. 结论

1. **window=1 后确实该重调参，旧（window=4）参数次优**——你的判断对。
2. **推荐新默认（GNN single）**：input_encoder=mlp、attention_temperature=0.25、
   alpha_lr_multiplier=2、lr=1e-3、epochs=500、patience=150。mean R² 0.8418→0.8688，
   GNN–probe 缺口减半，且无字段退化、推断全程经过图。
3. **剩余缺口主要在价格类**（congestion/marginal）。纯 config 调参到此接近天花板；
   要再逼近需图内架构增强（value 用 MLP、多头），或残差 MLP 旁路（但旁路会破坏
   "推断路径→敏感度"的纯粹性，慎用）。
4. **温度参数确实关键**（你提的对）：attention_temperature 越 sharp 越好（价格类
   尤甚）；但 edge_alpha_temperature 不该动。

## 7. 输出目录结构

```
configs/
├── full_<field>.yaml                 # 全 12 字段 combo 配置（推荐参数）
├── _sweep1_congestion_da_done/       # 第一批单因素扫描（归档）
└── _combo_hardfields_done/           # 难字段组合验证（归档）
outputs/
├── tuning/<config>/<timestamp>/      # 各调参 run
├── _probe_reference/                 # DNN40 probe 上界参照
├── relationship_cache/               # 相关性张量缓存（复用）
├── tuning_comparison.csv             # baseline vs combo vs probe 逐字段
└── tuning_comparison.png             # 柱状图
```
