# DNN_Aggresvation48 日志

## Modify by Claude: 2026-06-26

## 1. 子项目定位

用 **top-k 测试**评估 DNN47 七种敏感度方法的**排名质量**:对每种方法按敏感度从高到
低取 general 字段,从 top-1 递增到 top-44,每次只用这 top-k 个字段(其余置 0)在同一
载体模型(multi graph-combo, mean R²=0.8663)上推断 12 个 confidential,记录平均 R²。
排名越准 → 用越少字段越快逼近全字段上界。

代码:`scripts/evaluate_topk.py`(复用 DNN47 的模型与排名)。

## 2. 结果(`outputs/sensitivity/topk_curves.png`)

全 44 字段上界 mean R² = **0.8663**(图中水平虚线)。

| 方法 | top5 | top10 | top20 |
|---|---:|---:|---:|
| **shapley** | **0.374** | **0.623** | 0.790 |
| single | 0.374 | 0.593 | 0.779 |
| mask | 0.288 | 0.569 | 0.786 |
| attn | 0.322 | 0.482 | **0.813** |
| ig | 0.291 | 0.554 | 0.766 |
| lrp | 0.309 | 0.554 | 0.784 |
| **gate** | **0.195** | **0.231** | 0.714 |

## 3. 关键发现

1. **六个方法(mask/single/attn/ig/shapley/lrp)曲线成簇、排名质量相近**:都能用约
   25 个字段逼近全字段上界 0.8663,验证了这些敏感度排名"确实选出了最能推断的字段"
   ——这是敏感度分数有效性的直接证据(类比 IGNN 的 DIL test,但更系统)。

2. **Shapley 在小 k 段最优**(top10=0.623 最高):用最少字段就能推最准。符合 Shapley
   "公平边际、选出高效字段组合"的理论优势。single 紧随其后;attn 在中段(k≈18-22)反超。

3. **gate 排名质量明显最差**(曲线全程最低,top10 仅 0.231,比其他低 ~0.35)。但这
   **不意味 gate 无用,而是目标不匹配**:gate(联合 L1 稀疏)测的是"不可替代性/最小
   集合",不是"用 top-k 充分推断";top-k 测试的评判标准(推断准确率)天然偏向充分性/
   边际类方法,gate 在此标准下吃亏是必然的。gate 的价值在它的正交视角(揭示冗余),
   不在 top-k 充分性。

4. lrp 曲线≈ig(再次印证节点级 LRP≈梯度类)。

## 4. 结论

- top-k 测试**证明了敏感度排名的有效性**:按敏感度 top-k 选字段能高效逼近上界。
- **按"推断充分性"标准,Shapley 排名质量最高**,是首选敏感度方法。
- gate 不适合 top-k 评估(它测不可替代性);若论文要用 top-k 作为方法有效性证据,
  应以 shapley/single/mask 等边际/充分性方法为主,gate 作为"另一种敏感度定义"单列。
- 节点级 LRP≈IG,要兑现 LRP 独特价值仍需 walk-level(后续)。

## 5. 输出
```
outputs/sensitivity/
├── topk_curves.png   7 方法 top-k 曲线 + 全字段上界(0.8663)水平虚线
└── topk_curves.csv   每方法 k=1..44 的 mean R²
（沿用 DNN47 的 *_raw.csv / sensitivity_ranking.csv / 载体模型）
```
