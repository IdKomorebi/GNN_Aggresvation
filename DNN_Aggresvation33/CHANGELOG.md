# DNN_Aggresvation33 日志

## Modify by opencode: 2026-06-22

### 1. 子项目33要解决的问题

DNN32 证明 multi 模式比 single 模式平均低 0.04 R²，困难字段低 0.22。
用户提出**双重注意力机制**：多头 attention + target-conditioned head gate，
让每个 confidential 通过二级权重选择自己的"专属 head 组合"，在不拆成
12 个独立模型的前提下逼近 single 模式的效果。

### 2. 设计思路

在 DNN32 bipartite 主干的 `_confidential_messages` 层引入：

1. **多头 attention**：将 `attention_dim=64` 切成 `n_heads` 份（每份
   `head_dim=64/n_heads`），每个 head 独立计算 q/k/v/attention score。
   不同 head 可以捕获不同类型的 source-target 关系。
2. **target-conditioned head gate**：
   `head_weight_c = softmax(W_head · target_identity_c / τ)`
   每个 confidential target 学习自己的 head 组合权重。用 target_identity
   embedding 而非 hidden state，使 gate 学的是"target 身份→head 偏好"，
   更稳定可解释。
3. **加权合并**：`msg_c = Σ_h head_weight_c[h] · attention_h(c) · V_h`
   再过一层 `Linear(head_dim → hidden_dim)` 投影回 backbone 维度。

3 个对照实验：
- `baseline_1head`：n_heads=1（退化为 DNN32 单头，验证一致性）
- `dual_4head`：n_heads=4, head_gate_temperature=0.5
- `dual_8head`：n_heads=8, head_gate_temperature=0.5

### 3. 代码修改

#### `src/model.py`
- `__init__` 新增 `n_heads`, `head_gate_temperature`, `head_gate_bias_init`
  参数。每层新增 `head_gate_layers`（`Linear(hidden_dim, n_heads)`）和
  `head_output_projs`（`Linear(head_dim, hidden_dim)`）。
- `_confidential_messages()` 重写：
  - q/k/v 切分为 `(B, H, C/S, Dh)`
  - 每个 head 独立计算 gated attention score + softmax
  - `head_weight = softmax(head_gate(target_id) / τ)` 加权合并各 head 消息
  - 投影回 hidden_dim
- `n_heads=1` 时退化为单头（head_weight 恒为 1.0）。

#### `scripts/run_pipeline.py`
- 传入 `n_heads`, `head_gate_temperature`, `head_gate_bias_init`
- 输出目录：`outputs/<model_variant>/<timestamp>/`

### 4. 实验结果

| 字段 | 1head | 4head | 8head | single(DNN32) | 8h-1h | 8h-sgl |
|---|---:|---:|---:|---:|---:|---:|
| net_actual_interchange_mw | 0.7531 | 0.7071 | 0.6394 | 0.8198 | -0.114 | -0.180 |
| gross_actual_interchange_mw | 0.7191 | 0.7104 | 0.7088 | 0.7351 | -0.010 | -0.026 |
| total_gen | 0.9840 | 0.9835 | 0.9819 | 0.9953 | -0.002 | -0.013 |
| metered_load_mw | 0.9876 | 0.9875 | 0.9852 | 0.9946 | -0.002 | -0.009 |
| total_losses | 0.7855 | 0.7878 | 0.7825 | 0.8268 | -0.003 | -0.044 |
| congestion_price_da | 0.3447 | 0.3395 | 0.3162 | 0.5171 | -0.029 | -0.201 |
| congestion_price_rt | 0.5417 | 0.5506 | 0.5601 | 0.5028 | **+0.018** | +0.057 |
| marginal_loss_price_da | 0.7201 | 0.7131 | 0.7394 | 0.7624 | **+0.019** | -0.023 |
| total_lmp_da | 0.9736 | 0.9727 | 0.9695 | 0.9976 | -0.004 | -0.028 |
| da_as_total_mw_primary_reserve | 0.9446 | 0.9444 | 0.9491 | 0.9921 | **+0.005** | -0.043 |
| da_as_total_mw_synchronized_reserve | 0.9100 | 0.9077 | 0.9095 | 0.9959 | -0.001 | -0.086 |
| da_as_total_mw_thirty_minutes_reserve | 0.7984 | 0.7736 | 0.7512 | 0.7876 | -0.047 | -0.037 |

- best_test_loss: 1head=0.2231, 4head=0.2303, 8head=0.2377

### 5. 关键发现

1. **双重注意力没有提升，反而轻微下降**。8head vs 1head：
   mean Δ=-0.014, median Δ=-0.003。只有 2/12 字段（congestion_price_rt,
   marginal_loss_price_da）微升 >0.01。

2. **8head 仍然远低于 single**。8head vs single：mean Δ=-0.053,
   median Δ=-0.032。11/12 字段 single 仍然更好。

3. **head 数量越多效果越差**：1head > 4head > 8head。
   `net_actual` 从 0.75(1head) 降到 0.64(8head)。

4. **原因分析**：
   - 每个 head 的 `head_dim=64/n_heads` 太小（8head 时只有 8 维），
     单个 head 的表达能力严重不足。
   - head gate 的 softmax 权衡让不同 head 趋向均匀，没有形成"专属通道"
     的分化——因为 head gate 用 target_identity，而 target_identity 本身
     在训练中只通过 attention 间接获得梯度，信号太弱。
   - 本质问题：**multi 模式的瓶颈不在 attention head 数量，而在 General-GCN
     backbone 的共享表示**。所有 target 共享同一套 General 节点隐藏状态，
     head gate 只能在消息传递层做软选择，无法改变 backbone 已被 12 个
     target 平均化的事实。

### 6. 结论

双重注意力机制（multi-head + target-conditioned head gate）在本项目
设定下**未能缩小 multi 与 single 的差距**。根本原因是 General-GCN backbone
的共享表示瓶颈不在 attention 层，而在更底层的节点表示层。

**single 模式仍然是最优选择**（如果目标是最大化每个 target 的推断精度）。
如果需要在 1 次训练内做整体评估，baseline 1head 已足够，dual attention
不提供额外收益。

### 7. 输出目录结构

```
outputs/
├── baseline_1head/<timestamp>/    # n_heads=1
├── dual_4head/<timestamp>/        # n_heads=4
├── dual_8head/<timestamp>/        # n_heads=8
├── relationship_cache/
├── dual_attention_comparison.csv  # 对比汇总表
└── dual_attention_vs_baseline.png # 可视化柱状图
```
