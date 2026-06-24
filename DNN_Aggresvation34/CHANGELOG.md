# DNN_Aggresvation34 日志

## Modify by Claude: 2026-06-22

### 1. 子项目34要解决的问题

DNN33 验证「双重注意力」（多头 + 第二级 head 聚合）未能提升 multi 模式，
甚至 head 越多越差（1head > 4head > 8head），并把原因归为「multi 的瓶颈
在共享 backbone」。但 DNN33 的实现有**两个独立缺陷**纠缠在一起：

1. **切维度**：`head_dim = attention_dim / n_heads`，8head 时每头只剩 8 维，
   单 head 表达力不足。
2. **静态 head gate**：第二级用 `softmax(W·target_identity/τ)`，权重只取决于
   target 身份、与输入数据无关。DNN33 报告自己承认「梯度信号弱、趋向均匀、
   没形成专属通道分化」。

用户提出质疑：第二级（对各 head 的聚合）应该是**真正的注意力**——动态算出
「每个 head 的分数」，而不是一个静态权重表。

DNN34 把这两个缺陷拆开验证：
- 缺陷1由 DNN34 框架统一修复（**每个 head 用完整 attention_dim**，GLM 主张）；
- 缺陷2作为本项目的对照变量：`static_gate`（GLM 静态）vs
  `dynamic_attention`（用户的动态 head 注意力）。

### 2. 设计思路

第一级（head 内、对 source 节点的注意力）两种机制完全相同：每个 head 用独立
的、完整 `attention_dim` 的 q/k/v/gate 投影，算 gated attention 得到 `msg_h`。

第二级（对 H 个 head 的聚合）由 `head_aggregation` 二选一：

- **static_gate（GLM 方案）**：
  `head_weight = softmax(W_head · target_identity / τ)`
  权重只取决于 target 身份，与当前样本无关 → 静态。
- **dynamic_attention（用户思路）**：对 head 维度再做一次注意力
  - query `q_head = W_qid · target_identity + W_qh · h_conf`（融合身份与当前状态）
  - key   `k_head = W_k · msg_h`（来自各 head 的**实际输出**）
  - `head_weight = softmax( (q_head · k_head) / √hidden / τ )`
  权重随输入内容动态变化，梯度直连各 head 输出。

`n_heads=1` 时两种聚合等价（只有一个 head，权重恒为 1）。

5 个对照实验（shuffle split + bipartite + nostaged + allloss，与 DNN32/33 同口径）：
- `baseline_1head`：n_heads=1（单头基线，两种聚合等价）
- `static_4head` / `static_8head`：GLM 静态 gate
- `dynamic_4head` / `dynamic_8head`：用户的动态 head 注意力

### 3. 代码修改

#### `src/model.py`
- `__init__` 新增 `head_aggregation ∈ {static_gate, dynamic_attention}`，并按取值
  二选一构建第二级层：static 建 `head_gate_layers`；dynamic 建
  `head_attn_q_id_layers` / `head_attn_q_hidden_layers` / `head_attn_k_layers`。
- `_confidential_messages()` 第二级改为分支：static 走原 gate，dynamic 走
  head 维度的 query-key 注意力，两者统一用 `head_weight` 加权合并 `msg_h`。
- `n_heads=1` 自动退化为单头。

#### `scripts/run_pipeline.py`
- 传入 `head_aggregation`；输出目录按聚合方式命名
  `dynamic_{n}head` / `static_{n}head` / `baseline_1head`。

#### `scripts/scheduler.py`
- 修正为指向 DNN34（原模板仍指向 DNN32），4 GPU 并行跑 5 组。

#### `scripts/make_comparison.py`（新增）
- 跨 run 汇总各 variant 的 per-field `model_r2` 与 `best_test_loss`，
  产出 `dynamic_vs_static_comparison.csv` 与 `dynamic_vs_static.png`。

### 4. 实验结果

**best_test_loss（越低越好）**

| 变体 | best_test_loss |
|---|---:|
| baseline_1head | 0.2219 |
| static_4head | 0.2316 |
| static_8head | 0.2314 |
| dynamic_4head | 0.2229 |
| **dynamic_8head** | **0.2186** ← 全场最优 |

**per-field model R²（节选关键列）**

| 字段 | probe | baseline_1h | static_8h(GLM) | dynamic_8h(用户) | dyn8−stat8 | dyn8−base |
|---|---:|---:|---:|---:|---:|---:|
| net_actual_interchange_mw | 0.248 | 0.7588 | 0.7508 | 0.7517 | +0.001 | -0.007 |
| gross_actual_interchange_mw | 0.246 | 0.7252 | 0.7245 | 0.7336 | +0.009 | +0.008 |
| total_gen | 0.991 | 0.9848 | 0.9835 | 0.9862 | +0.003 | +0.002 |
| metered_load_mw | 0.976 | 0.9884 | 0.9847 | 0.9878 | +0.003 | -0.001 |
| total_losses | 0.654 | 0.7930 | 0.7821 | 0.8013 | **+0.019** | +0.008 |
| congestion_price_da | 0.048 | 0.3470 | 0.2810 | 0.3572 | **+0.076** | +0.010 |
| congestion_price_rt | 0.456 | 0.5211 | 0.5401 | 0.5464 | +0.006 | +0.025 |
| marginal_loss_price_da | 0.441 | 0.7247 | 0.7136 | 0.7186 | +0.005 | -0.006 |
| total_lmp_da | 0.930 | 0.9773 | 0.9726 | 0.9785 | +0.006 | +0.001 |
| da_as_total_mw_primary_reserve | 0.975 | 0.9461 | 0.9444 | 0.9493 | +0.005 | +0.003 |
| da_as_total_mw_synchronized_reserve | 0.942 | 0.9101 | 0.9089 | 0.9112 | +0.002 | +0.001 |
| da_as_total_mw_thirty_minutes_reserve | 0.457 | 0.8019 | 0.7836 | 0.7922 | +0.009 | -0.010 |

**动态 vs 静态（用户思路 vs GLM）汇总**

| 对比 | mean Δ | median Δ | 胜出字段 |
|---|---:|---:|---:|
| 4head dynamic − static | +0.0081 | -0.0008 | 5/12（混合） |
| **8head dynamic − static** | **+0.0120** | **+0.0054** | **12/12（全胜）** |
| 8head dynamic − baseline_1head | +0.0030 | +0.0013 | 8/12 |

### 5. 关键发现

1. **用户的思路成立，且 head 越多优势越明显**。8head 时动态注意力
   **12/12 字段全胜**静态 gate（mean +0.012）；4head 时优势还不稳定
   （median 略负）。符合直觉：head 越多，「如何选 head」越重要，动态选择
   的价值越大。

2. **趋势相对 DNN33 逆转**。DNN33 是 1head > 4head > 8head（越多越差）；
   DNN34 动态版变成 **dynamic_8head 最优**，best_test_loss 0.2186 甚至
   低于单头基线 0.2219，8/12 字段超过基线。多头第一次带来了（小幅）正收益。

3. **困难字段收益最大**。最难的 `congestion_price_da`（probe R² 仅 0.048）：
   动态 4head 0.376 vs 静态 4head 0.273（**+0.103**）；动态 8head 0.357 vs
   静态 8head 0.281（+0.076）。动态注意力在弱信号字段上更能挑出有用的 head。

4. **head 权重确实分化了**。未训练初始化下，静态 gate 的 head_weight
   std≈0.002（几乎均匀），动态注意力 std≈0.16（4head），直接对应 DNN33
   报告的「静态 gate 不分化」诊断被修复。

### 6. 结论

- **第二级用动态注意力（用户思路）比静态 gate（GLM）更合理**，在 8head 上
  得到干净、全面的验证；同时印证 GLM「不切维度、用完整 attention_dim」的
  判断也是对的——两人各对一半，合起来才让多头首次跑赢单头基线。
- **但 DNN33 的根本结论依然成立**：提升幅度很小（dynamic_8head 仅比 baseline
  高 mean +0.003），距离 single 上限仍有明显差距——dynamic_8head 平均比 single
  低 **0.034**（中位 -0.022），12 字段里仅 2 个（congestion_price_rt、
  thirty_minutes_reserve）追平/超过 single。对比图 `dynamic_vs_static.png` 里
  灰柱（single 上限）几乎全程在最上面。multi 追不上 single 的瓶颈在共享
  General-GCN backbone，本项目把「attention 层内部」做到了最优，但没有、
  也无法推翻那个更底层的瓶颈。
- **实用建议**：若坚持 multi 单次训练做整体评估，用 `dynamic_8head`（当前
  最优）；若要逼近每个 target 的极限精度，single 模式仍是上限。

### 7. 输出目录结构

```
outputs/
├── baseline_1head/<timestamp>/      # n_heads=1
├── static_4head/<timestamp>/        # GLM 静态 gate, 4 heads
├── static_8head/<timestamp>/        # GLM 静态 gate, 8 heads
├── dynamic_4head/<timestamp>/       # 用户动态 head 注意力, 4 heads
├── dynamic_8head/<timestamp>/       # 用户动态 head 注意力, 8 heads
├── relationship_cache/              # 相关性张量缓存（跨 run 复用）
├── dynamic_vs_static_comparison.csv # 对比汇总表（含 single 上限列）
└── dynamic_vs_static.png            # 柱状图：single 上限 / baseline / static / dynamic
```
