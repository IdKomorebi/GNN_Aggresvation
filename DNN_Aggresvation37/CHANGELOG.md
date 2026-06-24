# DNN_Aggresvation37 日志

## Modify by Claude: 2026-06-23

### 1. 子项目37要解决的问题

用户提出关键反驳：不要追求「12 个字段各达 single 最优」，而是**代偿**——
让 12 个字段共享 **K 个专家通道**（K=3/4/6），用低秩/软组合牺牲一点精度，
做到「等价于推断 K 个字段」的中间效果。并指出前几次（DNN34/35/36）是
**实现方式没做好**：低秩用成了 per-target 独立修正，没做"少数共享专家 + 软组合"。

DNN37 落地这个思路，并画 **K–精度代偿曲线**检验它是否成立。

### 2. 设计思路

12 个字段共享 **K 个专家**（每个专家只专属 value 投影 + 输出头；q/k/gate/
attention/general backbone 全共享，单头）。每个字段用一个**全局软分配
π_c ∈ Δ^K**（低温 softmax，由 target 身份生成）组合这 K 个专家：

```
msg_c  = Σ_k π_c[k] · ( Σ_j α(c,j)·v^k_j )    # value 专家按 π 组合
pred_c = Σ_k π_c[k] · head^k(h_conf_c)         # 输出头按 π 组合
```

12 个字段的有效自由度 ≤ K → 「等价推断 K 个字段」。K = `value_experts`：
- K=1 → 全共享（multi 下界）
- K=12 + π→one-hot → 每字段独占一专家
- K=3/4/6 → 中间代偿带

低温 `expert_route_temperature=0.33` 促 π_c 稀疏；`expert_diversity_weight=0.01`
加负载均衡 aux loss（防所有字段塌缩到同一专家）。扫 K∈{1,3,4,6,12}。

### 3. 代码修改

- `src/model.py`：新增 `value_experts` / `expert_route_temperature` /
  `expert_diversity_weight`。`use_experts` 时 `_confidential_messages` 走专家分支
  （单头共享 attention + K 个 value 专家按 π 组合），forward 输出层用 K 个专家
  输出头按 π 组合，并写 `self.aux_loss`（负载均衡）。`_expert_route()` 生成 π_c。
- `src/train.py`：训练循环把 `model.aux_loss` 加到主 loss（对旧模型无副作用）。
- `scripts/`：run_pipeline 传 3 个专家参数；scheduler 跑 5 组；make_comparison
  画 K–精度代偿曲线 + 每字段图。

### 4. 实验结果

**K–精度代偿曲线（single 上限 mean R² = 0.8273）**

| K | mean model R² | best_test_loss | 距 single 均值 |
|---:|---:|---:|---:|
| 1 (multi) | 0.7829 | 0.2290 | -0.0443 |
| **3** | **0.7901** | **0.2213** | **-0.0371** ← 甜点 |
| 4 | 0.7870 | 0.2247 | -0.0403 |
| 6 | 0.7893 | 0.2224 | -0.0380 |
| 12 | 0.7852 | 0.2267 | -0.0420 |

### 5. 关键发现

1. **代偿思想部分成立**：K=1→3 mean R² 从 0.7829 升到 0.7901（best_loss
   0.2290→0.2213），**拿回了约 16% 的 multi–single 差距**（0.0072 / 0.0443）。
   引入少数几个共享专家通道确实代偿了一点——方向是对的，这是 DNN35/36
   的 per-target 独立修正做不到的。

2. **但曲线在 K=3 见顶后饱和/回落，没有单调爬向 single**。K=4/6/12 都不如
   K=3，K=12 反而退回 0.7852。「K 越大越接近 single」的预期**没有兑现**。

3. **根因——专家只专属 value+输出头，特征提取仍共享**。为了低成本，K 个专家
   共用同一套 q/k/attention 和 general backbone（共享的 `h_general`）。增加 K
   只增加了 **readout 的专属性**，没解决最上游的**特征共享**——于是撞上和
   DNN35/36 完全相同的瓶颈。K=12 ≠ single：它是「专属 readout + 共享特征」，
   而 single 连特征提取都专属。

4. **K 太大反而更难训**：K=12 时路由要从 12 个专家里分，软分配稀释、每专家
   见的有效梯度少，叠加单头 attention（DNN37 为接专家用了单头，比 DNN34 的
   8 头弱），整体回落。

5. **历史最优仍是 DNN34 dynamic_8head（best_loss 0.2186）**。DNN37 的 K=3
   （0.2213）没超过它——多头 attention 和 value 专家是两种加容量的方式，
   但都撞同一个共享特征天花板。

### 6. 结论

- **用户的代偿思想是对的，但不是错在"没做好"，而是有一个由取舍决定的上限**：
  「轻量专家（只 value+输出头）」能代偿，但只能拿回约 16% 的差距、且 K=3 就
  饱和。要更大的代偿，必须让专家更"重"（专属到特征提取/整条通路），而那会
  让成本和参数向 single 靠拢——这正是一条清晰的**成本–代偿权衡曲线**。

- **跨 DNN34→37 的总账**：multi 的天花板（best_loss≈0.218、mean R²≈0.79）极其
  稳固，距 single（mean R² 0.827）的约 0.04 差距，靠**任何只动 value/readout 层
  的手段**（多头 / 对角 FiLM / 低秩 LoRA / K 专家）都突破不了——因为它们都没
  动到共享的特征表示 `h_general`。

- **实用建议**：
  - 要"一次训练、多字段、最省"：K=3 是当前性价比甜点（比纯 multi 好、参数省）；
    若不在乎单头，DNN34 dynamic_8head 仍是 multi 全局最优。
  - 要真正逼近 single：只剩"专家专属特征通路 / FiLM 整个 backbone"，成本接近
    single；或对个别困难字段单独上 single。
  - 隐私泄露评估：single 是每个字段的泄露上限，结论已足够清晰。

### 7. 输出目录结构

```
outputs/
├── experts_k1/<timestamp>/      # K=1（multi 下界）
├── experts_k3/<timestamp>/      # K=3（甜点）
├── experts_k4/<timestamp>/
├── experts_k6/<timestamp>/
├── experts_k12/<timestamp>/     # K=12
├── relationship_cache/          # 相关性张量缓存（跨 run 复用）
├── experts_comparison.csv       # 每字段 × 各 K 的 R²（含 single 上限）
├── experts_curve.png            # K–精度代偿曲线
└── experts_per_field.png        # 每字段 × 各 K + single 上限柱状图
```
