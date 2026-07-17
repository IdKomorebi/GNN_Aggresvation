# DNN_Aggresvation53 日志

## Modify by Claude: 2026-07-01

## 1. 子项目定位

DNN52 发现 transformer 式 q·k 注意力(独立 Q/K/V)在本任务上退化为近均匀、拖垮 GAT。
本子项目把 GAT 换成**经典加性 GAT**重测 gat_noprior / gat_static / gat_dynamic 三种,
看更简单的注意力是否改变 GAT 的结论。

### 经典 GAT(本子项目)vs transformer GAT(DNN52)
经典 GAT:**共享一个 W**得 z = W·h(**K = V = Wh**),加性注意力
`e_ij = LeakyReLU(a_l^T z_i + a_r^T z_j)`,`α = softmax_j(e_ij)`,`h'_i = σ(Σ α_ij z_j)`。
transformer GAT(DNN52):独立 Q/K/V 三套投影,`score = q·k/√d`。

**只替换注意力打分这一处**,其余管线(温度 0.25、先验静态/动态混合方式、softmax、
mask、聚合)与 DNN52 完全一致 → 干净隔离"注意力机制"这一个变量。3 模式 × 3 编码
(linear_noact/linear/mlp)= 9 个训练,与 DNN52 对应格子直接比。

## 2. 结果(12-conf 平均 test R²，DNN probe 上界 0.8937)

| 编码 | mode | classic(本) | transformer(DNN52) | Δ(c−t) | gcn_dynamic(DNN52) |
|---|---|---:|---:|---:|---:|
| 弱 linear_noact | gat_noprior | 0.8308 | 0.8244 | **+0.0064** | 0.8646 |
| 弱 | gat_static | 0.8446 | 0.8382 | **+0.0064** | 0.8646 |
| 弱 | gat_dynamic | 0.8463 | 0.8470 | −0.0007 | 0.8646 |
| 中 linear | gat_noprior | 0.8551 | 0.8509 | +0.0042 | 0.8667 |
| 中 | gat_static | 0.8547 | 0.8532 | +0.0015 | 0.8667 |
| 中 | gat_dynamic | 0.8489 | 0.8552 | −0.0063 | 0.8667 |
| 强 mlp | gat_noprior | 0.8604 | 0.8506 | **+0.0098** | 0.8746 |
| 强 | gat_static | 0.8613 | 0.8523 | **+0.0090** | 0.8746 |
| 强 | gat_dynamic | 0.8628 | 0.8727 | **−0.0099** | 0.8746 |

图:`outputs/_summary/classic_vs_transformer.png`、`per_encoder_bars.png`。

## 3. 关键发现

### 3.1 经典 GAT 改善"裸"注意力(noprior/static)，但略伤动态门控模式
- gat_noprior：classic 全面优于 transformer(+0.004~+0.010)。
- gat_static：classic 全面优于(+0.002~+0.009)。
- gat_dynamic：classic **劣于** transformer(中/强 −0.006/−0.010)。动态模式里逐对 gate
  靠注意力信号做混合,transformer 的 q·k 给 gate 的信号更"可分解",反而更好用。

### 3.2 经典注意力**不崩**(熵证据，回应 DNN52 的崩溃)
gat_noprior 注意力归一化熵 / 近均匀(熵≥0.95)目标数:
| 编码 | classic 熵 | classic 近均匀 | transformer 熵 | transformer 近均匀 |
|---|---:|---:|---:|---:|
| 中 linear | 0.935 | **0/12** | 0.944 | **8/12** |
| 强 mlp | 0.891 | **0/12** | 0.948 | **7/12** |

→ transformer q·k 在中/强编码下 7~8/12 个目标注意力退化为均匀(纯平均);**经典加性
注意力 0/12,始终保持区分性**。这正是 classic 的 noprior/static 不再崩、反超 transformer
的原因——**经典 GAT 是本任务上更稳健的注意力机制,不会退化为均匀聚合**。

### 3.3 经典 GAT 把三模式"压平"
强编码下 classic 三模式 0.8604/0.8613/0.8628(极差 0.0024),而 transformer 是
0.8506/0.8523/0.8727(极差 0.0221)。即 **classic 注意力本身已做了不少工作,先验/门控
的边际作用变小**;transformer 因注意力会崩,更依赖先验/gate 去补救。

### 3.4 核心结论不变:仍**打不过 gcn_dynamic**
即便更稳健的 classic GAT,最佳(强/dynamic 0.8628)仍 **低于 gcn_dynamic 0.8746**,
每一档都如此。**"动态先验门控的无注意力 GCN 最好"这一 DNN52 结论,在更换注意力机制
后依旧成立。** 换注意力只是在 ~0.86 区间内挪动 GAT,天花板没抬过 gcn_dynamic。

## 4. 回答与启示

- 你换经典 GAT 的直觉**部分正确**:它确实更好——修好了 transformer q·k 的"注意力崩成
  均匀"问题,裸注意力(noprior/static)各档 +0.4~1.0%。
- 但**没有改变大局**:① classic 反而让最佳的 dynamic 模式略降;② 任何注意力变体都仍
  低于 gcn_dynamic。**若只看准确率,gcn_dynamic 仍是首选。**
- **项目价值点**:若叙事需要"真正在注意的注意力"(做推断路径/敏感度,而非退化的均匀),
  **经典加性 GAT 明显比 transformer 更合适**(熵证据:0/12 vs 7~8/12 退化)。即:
  主干用 gcn_dynamic 拿准确率与可解释门控;若要 attention 形式的路径对照,用 classic GAT
  而非 transformer。

## 5. 局限
单 seed。3.1/3.2(classic 改善裸模式、不崩)幅度明确、跨档一致;弱编码内部细小排序在
噪声内,不过度解读。

## 6. 复跑
```
scripts/scheduler.py   # 9 训练(经典 GAT × 3 编码 × 3 GAT 模式，4 卡)
scripts/diagnose.py --encoder <e> --mode <m>
scripts/compare.py     # classic(本) vs transformer(DNN52) + gcn_dynamic/probe 参考
```
输出结构同 DNN52：outputs/<编码档>/<gat模式>/ 含 vs_probe、attention_report、
(static)alpha_bar、(dynamic)gate_by_edge_long；_summary/ 为跨编码对比。
```
