# DNN_Aggresvation37 走读报告

## 子项目定位

用户反驳:不追求每字段达 single 最优,而是**代偿**——12 个字段共享 **K 个专家**,
用软组合做到「等价推断 K 个字段」的中间效果。DNN37 落地这个思路并画
**K–精度代偿曲线**。

---

## 改动

12 字段共享 K 个专家(只专属 value 投影 + 输出头;q/k/attention/backbone 共享),
每字段用全局软分配 `π_c ∈ Δ^K`(低温 softmax)组合:

```
msg_c  = Σ_k π_c[k]·(Σ_j α(c,j)·v^k_j)
pred_c = Σ_k π_c[k]·head^k(h_conf_c)
```

低温促 π 稀疏,负载均衡 aux loss 防塌缩。扫 K∈{1,3,4,6,12}。

---

## 核心结果：代偿思想部分成立

| K | mean R² | best_loss | 距 single |
|---:|---:|---:|---:|
| 1 (multi) | 0.7829 | 0.2290 | -0.0443 |
| **3** | **0.7901** | **0.2213** | **-0.0371** ← 甜点 |
| 4 | 0.7870 | 0.2247 | -0.0403 |
| 6 | 0.7893 | 0.2224 | -0.0380 |
| 12 | 0.7852 | 0.2267 | -0.0420 |

single 上限 mean R² = 0.8273。

- **K=1→3 上升**:拿回约 **16%** 的 multi–single 差距——代偿确实生效,方向对,
  这是 DNN35/36 的 per-target 独立修正做不到的。
- **但 K=3 见顶后饱和/回落**,没有单调爬向 single,K=12 反而退回。

---

## 为什么没继续爬向 single

专家**只专属 value + 输出头**,q/k/attention 和 general backbone(共享 `h_general`)
全共享。增加 K 只增 **readout 专属性**,没解决最上游的**特征共享**→撞上和
DNN35/36 同一个瓶颈。K=12 是"专属 readout + 共享特征",而 single 连特征提取
都专属。K 太大还更难训(路由稀释 + 单头 attention 弱于 DNN34 的 8 头)。

---

## 结论

- 用户的代偿思想**对**,但有一个由取舍决定的**上限**:轻量专家能拿回约 16%、
  K=3 就饱和。要更大代偿必须让专家更"重"(专属特征通路),成本向 single 靠拢
  ——一条清晰的成本–代偿权衡。
- 跨 DNN34→37:multi 天花板(best_loss≈0.218)极稳固,**任何只动 value/readout 的
  手段**(多头/对角/低秩/K专家)都突破不了共享特征瓶颈。
- 实用:省成本多字段用 K=3 甜点(或 DNN34 dynamic_8head);逼近 single 只剩
  "专家专属特征通路 / FiLM backbone",成本接近 single;隐私评估用 single 上限。

详见 [CHANGELOG.md](CHANGELOG.md)、`outputs/experts_curve.png`、`outputs/experts_comparison.csv`。
