# DNN_Aggresvation36 日志

## Modify by Claude: 2026-06-23

### 1. 子项目36要解决的问题

DNN35 用对角 FiLM（`diag(γ_c)·W_V`）让 value 变成 target-specific，整体无增益。
诊断：**对角变换只能逐维缩放、不能跨维度组合**——若某字段需要的信号是隐藏态
各维的某个线性组合、而共享 `W_V` 没投出来，对角 γ 救不回来。

DNN36 把"对角"换成"低秩"，补上跨维组合能力：

```
W_V^(c) = W_V + B_c · A_c          ΔW_c=B_c·A_c 是满矩阵，秩≤r，可跨维组合
```

`value_lora_rank=r` 控制（0=关闭）。验证：补上跨维组合后能否把 multi 往
single 拉近。

### 2. 设计思路

每个 target c 学一个低秩增量 `ΔW_c=B_c·A_c`（A: r×d，B: d×r），加到共享
value 投影上。per-(layer,head) 一对，A:(H,C,r,d)、B:(H,C,d,r)，**B 零初始化**
→ 起步 ΔW=0（identity）。

实现用恒等式（避免把张量撑大到 (B,C,S,d)）：

```
m_c = (W_V + B_c·A_c)·h̄_c ,   h̄_c = Σ_j α(c,j)·h_j   （注意力加权的源隐藏态）
    = bmm(α, v)  +  B_c·(A_c·h̄_c)
```

三臂（都在 DNN35 最优的 dynamic head attention + 8 heads 上）：
- `dynamic_8head`：无 LoRA（= DNN35 最优锚点）
- `lora_r4`：秩 4
- `lora_r8`：秩 8

### 3. 代码修改

#### `src/model.py`
- `__init__` 新增 `value_lora_rank`；`value_lora_rank>0` 时按 per-(layer,head)
  构建 `lora_A`（normal 初始化）、`lora_B`（零初始化）。
- `_confidential_messages()` head 循环里，对每个 head 的消息加低秩修正
  `delta = B_c·(A_c·h̄_c)`。

#### `scripts/`
- run_pipeline 传 `value_lora_rank` + 打印；scheduler / make_comparison 适配 3 臂。

### 4. 实验结果

**best_test_loss（越低越好）**

| 变体 | best_test_loss |
|---|---:|
| dynamic_8head（无 LoRA） | **0.2186** ← 仍最优 |
| lora_r4 | 0.2242 |
| lora_r8 | 0.2211 |

**LoRA 增益（per-field R²，相对无 LoRA 的 dynamic_8head）**

| rank | mean Δ | median Δ | 胜出字段 |
|---|---:|---:|---:|
| 4 | -0.0053 | -0.0040 | 2/12 |
| 8 | -0.0025 | -0.0055 | 1/12 |

**距 single 上限**：三臂里每字段取最好的，平均仍比 single 低 **0.030**
（DNN35 是 -0.030、DNN34 单臂 -0.034，几乎没改善）。

**唯一明显的亮点：最难字段 congestion_price_da**

| 字段 | probe | single | 无LoRA | lora_r4 | lora_r8 |
|---|---:|---:|---:|---:|---:|
| **congestion_price_da** | 0.048 | 0.517 | 0.3572 | 0.3518 | **0.4064 (+0.049)** |
| net_actual_interchange_mw | 0.248 | 0.820 | 0.7517 | 0.7520 | 0.7446 |
| synchronized_reserve | 0.942 | 0.996 | 0.9112 | 0.9100 | 0.9097 |
| congestion_price_rt | 0.456 | 0.503 | 0.5464 | 0.5407 | 0.5265 |

### 5. 关键发现

1. **LoRA 整体仍无增益**：rank4 平均 -0.0053、rank8 平均 -0.0025，best_test_loss
   仍是无 LoRA 的 dynamic_8head 最低。和 DNN35 的 FiLM 一样，整条「在 value 层
   做 per-target 修正」的路线没能系统缩小与 single 的差距。

2. **但在最难字段上，低秩确实比对角强**。congestion_price_da（probe 0.048，
   连线性探针都几乎不可分，最难）：rank8 把它从 0.357 拉到 **0.406（+0.049）**，
   到 single 的差距从 -0.16 缩到 -0.11。这是 DNN34/35/36 三个子项目里对这个
   字段**唯一明显的提升**，且符合理论预期：
   - 它是「弱信号 + 强非线性」字段，需要专属的跨维组合容量；
   - **需要足够的秩才显现**：rank4 没用（-0.005），rank8 才起效（+0.049）——
     对角 FiLM（秩=0 的特例）更不行，正好印证「对角太弱、低秩才够」的诊断。

3. **代价是其他字段普遍小幅退化**。LoRA 给每字段每 head 增加 r·d 量级参数，
   在 multi 共享框架里梯度信号弱、易过拟合/互相干扰，多数字段（net_actual、
   total_lmp_da、marginal_loss 等）略降，把 congestion_price_da 的亮点抵消了。

### 6. 结论

- **低秩 value（LoRA）也不是系统性的解**，整体增益≈0；但它在最难字段上
  确实兑现了「跨维组合」的承诺（congestion_price_da +0.049），方向判断是对的，
  只是收益局限在个别字段、被其他字段的退化抵消。

- **三步连起来给出一个明确的结论**：在注意力的**下游**做 per-target 修正——
  DNN34（路由 α）→ DNN35（对角 value）→ DNN36（低秩 value）——每一步整体
  收益都≈0，只能零星帮到个别字段。瓶颈被牢牢锁定在**更上游的共享 backbone
  表示 `h_general`**：所有 value 都从它投影而来，它一旦被 12 个 target 平均化，
  下游再怎么做 per-target 变换也只是「在贫信息上重新加权」。

- **要系统性提升，必须换战场**（不再动 value，而是动表示本身）：
  1. **FiLM / 条件化整个 general backbone**：用 target embedding 调制 general
     节点每层隐藏态，让"菜单"在更上游就 target-specific（需引入 target 维度，
     显存/算力上升，接近「软 single」）。
  2. **MoE**：每个 target 路由到独立子网络（expert 间连 backbone 都不共享）。
  3. 若仅个别困难字段（如 congestion_price_da）值得救，可只对它单独上
     LoRA-r8 或干脆 single。

- 趋势提示：multi 想追平 single，代价正在逼近「接近 single」。若目标是隐私
  泄露评估，single 才是每个字段的泄露上限；multi 用 dynamic_8head 即可。

### 7. 输出目录结构

```
outputs/
├── dynamic_8head/<timestamp>/   # 无 LoRA（= DNN35 最优锚点）
├── lora_r4/<timestamp>/         # 秩 4
├── lora_r8/<timestamp>/         # 秩 8
├── relationship_cache/          # 相关性张量缓存（跨 run 复用）
├── lora_comparison.csv          # 对比汇总表（含 single 上限 + LoRA 增益）
└── lora_comparison.png          # 柱状图：single 上限 / 无LoRA / r4 / r8
```
