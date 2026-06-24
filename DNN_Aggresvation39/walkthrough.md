# DNN_Aggresvation39 走读报告

## 子项目定位

验证用户的质疑——"不应该阻断"：让 General 节点的**前向聚合也纳入 Confidential
邻居**（不只靠反向传播间接受影响），看 multi 是否提升。在最基础版
（DNN32_test：window=1、无双重注意力、bipartite）上做单点改动，对比 DNN32_test。

---

## 改动

`src/model.py` 新增 `general_aggregation`：
- `general_only`（原始）：General 只从 General 聚合（前向阻断 Confidential）。
- `include_confidential`（本项目）：General 行保留所有列，也从 Confidential 的
  推断表示聚合（Confidential 初始为可学 embedding，不泄露真值）。

其余完全同 DNN32_test 口径，重跑 1 multi + 12 single。

---

## 核心结果：不阻断没用，multi 反而略降

| | multi (block) | multi (include) | Δ |
|---|---:|---:|---:|
| MEAN R² | 0.8096 | 0.8073 | **-0.0024** |
| best_loss | 0.2008 | 0.2033 | +0.0025（更差） |

single 几乎不变（-0.0008，n_confidential=1 影响微乎其微）；single–multi gap
0.0802→0.0818 基本不动。困难字段 congestion_price_da multi 受损最大（-0.014）。

---

## 为什么没用 + 这个结果的价值

Confidential 节点的表示是模型**推断中的、带误差**的，且本身就是 General 的函数
（无真值输入）。General 从它聚合 = 把「General 信息的二手加工 + 推断误差」回流，
**无新信息、反而加噪**。

**排除性结论**：multi/single 差距**不在「前向 general 聚合阻不阻断」这个结构**。
Confidential 影响 General 主要靠**反向传播塑造共享参数**（一直在起作用，不该也无法
用前向开关关掉）；真正差距源是 DNN37 权重分析揭示的——**图结构 alpha_general 与
注意力 query/key 被 12 个 target 折衷**。

---

## 结论与下一步

- "不阻断"在前向结构上无效（略损），不是杠杆，保持 `general_only` 即可。
- 回到真正的差异源：**target-specific / K-专家化的 alpha_general 与 query/key**
  （DNN38 方向），而非动前向聚合连接。

详见 [CHANGELOG.md](CHANGELOG.md)、`outputs/include_vs_block_comparison.csv`、`outputs/include_vs_block.png`。
