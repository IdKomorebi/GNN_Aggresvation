# DNN_Aggresvation35 日志

## Modify by Claude: 2026-06-22

### 1. 子项目35要解决的问题

DNN34 把双重注意力的「路由层」（attention weights + head 聚合）做到了 multi
框架内的最优（dynamic_8head），但仍追不上 single 上限（平均低 0.034）。

诊断指向一个更上游的瓶颈：注意力里的 **value `v_j = W_V·h_j` 只依赖 source
j、与 target c 无关**，12 个 confidential 字段共用同一组 `{v_j}`（同一份
「菜单」），字段间差异只在注意力权重。single 之所以是上限，是因为它为每个
字段单独学一个完整的 value 投影 `W_V^(c)`。

本项目验证假设：**让 value 变成 target-specific，能否把 multi 往 single 拉近。**

### 2. 设计思路

给每个 target c 一组逐维缩放/平移 (γ_c, β_c) 调制 value：

```
v(c,j) = γ_c ⊙ v_j + β_c
```

由于注意力权重 α(c,j) 与 head 权重都是凸组合（对 j、对 head 求和为 1），
γ_c/β_c 可提到求和外，**等价于在「汇聚后的消息」上做一次 FiLM**：

```
msg_c ← (1 + Γ_c) ⊙ msg_c + B_c
```

即用对角变换 `diag(γ_c)·W_V` 廉价逼近 single 才有的 per-target value 投影，
且仍是一次前向同时算完所有 target（不退化成 single 的 12× 成本）。

`value_film` 三档（全部叠在 DNN34 最优的 dynamic head attention + 8 heads 上）：
- `none`    ：不调制（= DNN34 dynamic_8head 复现，对照锚点）
- `static`  ：Γ_c, B_c 为 per-(layer,target) 自由参数（查找表，对所有样本相同）
- `dynamic` ：Γ_c, B_c 由 target 身份 e_c + 当前 conf 隐藏态 h_c 生成，随样本变化
- 三档初始化均为 identity（Γ=0, B=0），起步等价于无 FiLM。

### 3. 代码修改

#### `src/model.py`
- `__init__` 新增 `value_film ∈ {none, static, dynamic}`，按取值构建：
  static → `film_gamma/film_beta`（ParameterList，零初始化）；
  dynamic → `film_{g,b}_{id,h}`（Linear，权重/偏置零初始化）。
- `_confidential_messages()` 在 head 聚合得到 `msg` 后做 FiLM 调制。

#### `scripts/run_pipeline.py`
- 传入 `value_film`；输出目录直接用 `experiment.name`（dynamic_8head /
  film_static_8head / film_dynamic_8head）。

#### `scripts/scheduler.py` / `scripts/make_comparison.py`
- 适配 DNN35 的 3 臂；汇总加 single 上限与 FiLM 增益统计。

### 4. 实验结果

**best_test_loss（越低越好）**

| 变体 | best_test_loss |
|---|---:|
| dynamic_8head（无 FiLM） | **0.2186** ← 仍最优 |
| film_static_8head | 0.2188 |
| film_dynamic_8head | 0.2224 |

**FiLM 增益（per-field R²，相对无 FiLM 的 dynamic_8head）**

| FiLM 档 | mean Δ | median Δ | 胜出字段 |
|---|---:|---:|---:|
| static | **+0.0001** | -0.0010 | 4/12 |
| dynamic | **-0.0037** | -0.0019 | 2/12 |

**距 single 上限**：三臂里每字段取最好的，平均仍比 single 低 **0.030**
（DNN34 dynamic_8head 单臂是 -0.034，几乎没改善）。

**零星有效的字段（符合「强信号被稀释」的预期方向，但被其他字段抵消）**

| 字段 | single | 无FiLM | film_static | film_dynamic |
|---|---:|---:|---:|---:|
| net_actual_interchange_mw | 0.820 | 0.7517 | **0.7769 (+0.025)** | 0.7480 |
| marginal_loss_price_da | 0.762 | 0.7186 | **0.7313 (+0.013)** | 0.7167 |
| thirty_minutes_reserve | 0.788 | 0.7922 | 0.7905 | **0.8070 (+0.015，超 single)** |
| congestion_price_da | 0.517 | 0.3572 | 0.3574 | 0.3398 |
| synchronized_reserve | 0.996 | 0.9112 | 0.9070 | 0.9093 |

### 5. 关键发现

1. **FiLM 整体没有带来增益**：static 平均 +0.0001（基本持平），dynamic 平均
   -0.0037（反而略降）；best_test_loss 仍是无 FiLM 的 dynamic_8head 最低。
   target-specific value 这个假设**在本项目上未能成立**。

2. **但它精确地定位了瓶颈**。把三个子项目连起来看：
   - DNN34 改「怎么选」（attention 路由）→ 小幅提升（+0.003 vs baseline）
   - DNN35 改「value/菜单本身」（汇聚消息的 per-target 仿射）→ 几乎无提升
   - 唯一没动过的：**general 节点的隐藏表示 `h_general` 本身**——它是共享的、
     被 12 个 target 平均化的。

   FiLM 作用在 `Σα·v = Σα·(W_V·h_general)` 上，`h_general` 已经是平均化的表示。
   **对角变换 `diag(γ_c)` 只能逐维缩放、不能跨维度重新组合（旋转）**：若某字段
   需要的信号是 `h_general` 各维的特定线性组合、而共享 `W_V` 没投出来，对角 γ
   救不回来。瓶颈在更上游的「共享节点表示」，且需要比对角更强的变换才能动它。

3. **零星成功给了佐证**：net_actual（single 0.82、被稀释类型）static FiLM
   +0.025，确实把它往 single 拉了一点——说明个别字段的缺口确实是「value
   逐维加权」能补的；但多数字段（如 synchronized_reserve 缺口 0.085）纹丝不动，
   说明它们的缺口在对角 FiLM 够不到的地方。

4. **dynamic 比 static 还差**：dynamic FiLM 多了 ~50k 弱梯度参数，在 multi 共享
   框架里反而扰动训练——与 DNN33 「target_identity 信号弱」是同一类问题。

### 6. 结论

- **对角 value 调制（FiLM）不是这个问题的解**，整体增益≈0。
- 价值在于：DNN34（路由层）+ DNN35（value 仿射层）两步系统地**排除**了两个
  层次，把瓶颈精确逼到**共享的 general backbone 表示 `h_general`**——这是
  multi 与 single 差距的真正源头，且需要「能跨维组合 / 能改 backbone 本身」的
  更强手段。
- **下一步候选**（按强度/成本递增）：
  1. **LoRA / 低秩 value**：`W_V^(c) = W_V + U_c V_cᵀ`，能跨维度组合，比对角强；
  2. **FiLM 整个 general backbone**：用 target embedding 调制 general 节点每层
     隐藏态（需引入 target 维度，显存/算力上升，接近「软 single」）；
  3. **MoE**：每个 target 路由到独立的子网络（expert 间不共享 k/v），表达力最强、
     最贵。
- 若目标是隐私泄露评估，single 仍是每个字段的泄露上限；multi 用 dynamic_8head
  即可（FiLM 不必加）。

### 7. 输出目录结构

```
outputs/
├── dynamic_8head/<timestamp>/       # 无 FiLM（= DNN34 最优复现）
├── film_static_8head/<timestamp>/   # 静态 FiLM
├── film_dynamic_8head/<timestamp>/  # 动态 FiLM
├── relationship_cache/              # 相关性张量缓存（跨 run 复用）
├── film_comparison.csv              # 对比汇总表（含 single 上限 + FiLM 增益）
└── film_comparison.png              # 柱状图：single 上限 / 无FiLM / static / dynamic
```
