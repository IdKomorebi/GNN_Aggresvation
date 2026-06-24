# DNN_Aggresvation35 走读报告

## 子项目定位

DNN34 把注意力的「路由层」做到了 multi 内最优，但追不上 single（平均低 0.034）。
诊断指向更上游:注意力的 value `v_j=W_V·h_j` 与 target 无关,12 个字段共用一份
「菜单」。DNN35 验证:**让 value 变成 target-specific(FiLM 调制)能否拉近 single。**

---

## 改动

`src/model.py` 新增 `value_film`,对汇聚后的消息做 `msg_c ← (1+Γ_c)⊙msg_c + B_c`
(数学上等价于 per-target value `v(c,j)=γ_c⊙v_j+β_c`,因注意力/head 权重是凸组合)：

- `none`   ：不调制(= DNN34 dynamic_8head)
- `static` ：Γ_c,B_c 为 per-target 查找表参数
- `dynamic`：Γ_c,B_c 由 target 身份 + 当前隐藏态生成
- 三档初始为 identity,均叠在 DNN34 最优的 dynamic head attention + 8 heads 上。

---

## 核心结果（诚实的负面结果）

| 变体 | best_test_loss | FiLM 增益 (mean Δ R²) |
|---|---:|---|
| dynamic_8head (无 FiLM) | **0.2186** ← 仍最优 | — |
| film_static_8head | 0.2188 | +0.0001（基本持平） |
| film_dynamic_8head | 0.2224 | −0.0037（略降） |

**FiLM 整体没带来增益。** 三臂里每字段取最好的,距 single 仍差 0.030(对比
DNN34 单臂 -0.034,几乎没改善)。零星例外:net_actual 静态 FiLM +0.025、
thirty_minutes 动态 FiLM +0.015(超 single),但被其他字段抵消。

---

## 为什么没 work / 它定位了什么

把三步连起来：
- DNN34 改「怎么选」(attention 路由) → 小幅提升
- DNN35 改「value/菜单本身」(汇聚消息的 per-target 仿射) → 几乎无提升
- 唯一没动的:**general 节点表示 `h_general` 本身**(共享、被 12 个 target 平均化)

FiLM 作用在 `Σα·(W_V·h_general)` 上,而 `h_general` 已是平均化表示;**对角
`diag(γ_c)` 只能逐维缩放、不能跨维组合**,救不回共享 `W_V` 没投出来的方向。
→ 瓶颈精确逼到「共享 backbone 表示」,需要更强的手段。

---

## 结论与下一步

- 对角 value 调制不是解;DNN34+DNN35 系统排除了「路由层」和「value 仿射层」。
- 候选(强度/成本递增)：① LoRA/低秩 value(能跨维组合) ② FiLM 整个 backbone
  ③ MoE(每 target 独立子网络)。
- 隐私评估用 single 上限;multi 用 dynamic_8head 即可,FiLM 不必加。

详见 [CHANGELOG.md](CHANGELOG.md)、`outputs/film_comparison.csv`、`outputs/film_comparison.png`。
