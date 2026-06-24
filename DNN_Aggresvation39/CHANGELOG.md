# DNN_Aggresvation39 日志

## Modify by Claude: 2026-06-23

### 1. 子项目定位

权重分析（DNN37 后）发现：single 模式下每个字段的图结构（alpha_general）和注意力
（query/key）差异巨大，而 multi 被迫共享一套。用户进一步质疑前向结构：

> 之前以为 multi/single 只差在「聚合到 confidential」那一步，但 confidential 会通过
> 反向传播反过来影响 general-general 聚合。我说过「不应该阻断」——那就先把前向
> 也放开，让 General 节点也从 Confidential 聚合，看 multi 会不会提升。

DNN39 在**最基础版**（DNN32_test：window=1、无双重注意力、bipartite）上做这个单点改动，
直接对比 DNN32_test。

### 2. 改动：general 聚合「不阻断」

原始（DNN32_test，`general_only`）：`compute_general_adjacency` 用 mask 只保留
General-General 块，General 节点前向聚合时**看不到 Confidential 邻居**。

DNN39 新增开关 `general_aggregation`：
- `general_only`（原始）：General 只从 General 聚合。
- `include_confidential`（本项目）：General 行保留所有列，**General 也从
  Confidential 的（推断中的）表示聚合**。Confidential 节点初始是可学 embedding
  （不含真值），故不泄露标签；前向变成 General↔Confidential 双向传播。

Confidential 自身仍走原 bipartite 注意力（只从 General）。其余完全同 DNN32_test
口径（window=1、shuffle、nostaged、allloss、seed 42），重跑 1 multi + 12 single。

#### 代码
- `src/model.py`：`__init__` 加 `general_aggregation` + 校验；`compute_general_adjacency`
  按开关选择行掩码；`forward` 的 general_messages 在 include 模式用
  `A[:n_general, :] @ h`（从所有节点聚合）。
- `scripts/run_pipeline.py`：传 `general_aggregation` + 打印；scheduler 指向 DNN39；
  新增 `make_comparison.py` 对比 DNN32_test。

### 3. 实验结果

**include（DNN39）vs general_only（DNN32_test），逐字段 model R²**

| | multi block | multi include | Δmulti | single block | single include | Δsingle |
|---|---:|---:|---:|---:|---:|---:|
| MEAN | 0.8096 | 0.8073 | **-0.0024** | 0.8898 | 0.8890 | -0.0008 |
| multi 胜出字段 | | | **5/12** | | | |

- multi best_test_loss：0.2008（block）→ **0.2033（include，反而变差）**
- single–multi gap：0.0802 → 0.0818（几乎不变）
- 受损最明显：困难字段 congestion_price_da multi 0.4175 → 0.4034（-0.014）

### 4. 关键发现

1. **「不阻断」没有带来提升，multi 反而略降**（mean -0.0024、best_loss +0.0025、
   5/12 胜）。single 几乎不变（n_confidential=1，General 从单个 Confidential 聚合
   影响微乎其微）。

2. **为什么没用**：Confidential 节点的表示是模型**推断中的、带误差的**表示，且
   它本身就是 General 的函数（模型里没有 Confidential 真值输入）。让 General 从它
   聚合，等于把「General 信息的二手加工 + 推断误差」回流到 General，**没有新信息
   增量，反而引入噪声**——困难字段（误差大）受损最明显。

3. **这是一个有价值的排除性结果**：它证明 **multi/single 的差距不在「前向 general
   聚合阻不阻断 confidential」这个结构**上。Confidential 影响 General 主要靠**反向
   传播塑造共享参数**（这条路一直在起作用，且不该、也无法用前向开关关掉），
   而真正的差距来源是 DNN37 权重分析揭示的：**图结构 alpha_general 与注意力
   query/key 被 12 个 target 折衷**（single 之间这些量差异巨大）。

### 5. 结论

- 用户「不应该阻断」的方向**在前向结构上验证为无效**（甚至略损）——前向让 General
  看 Confidential 不是杠杆。
- 焦点应回到真正的差异源：**让图结构（alpha_general）/注意力（query/key）变得
  target-specific 或 K-专家化**（DNN38 方向），而不是动前向聚合连接。
- 实用层面：保持 `general_only`（DNN32_test）即可，本改动无收益。

### 6. 输出目录结构

```
outputs/
├── multi/<timestamp>/                # include_confidential，12 字段
├── single/<field>/<timestamp>/       # 12 个 single（n_confidential=1）
├── relationship_cache/               # 相关性张量缓存（复用自 DNN32_test）
├── include_vs_block_comparison.csv   # 逐字段对比 DNN32_test
└── include_vs_block.png              # 柱状图：single 上限 / multi block / multi include
```
