# DNN_Aggresvation36 走读报告

## 子项目定位

DNN35 的对角 FiLM 无效,诊断:对角只能逐维缩放、不能跨维组合。DNN36 把对角
换成**低秩 LoRA**(`W_V^(c)=W_V+B_c·A_c`,秩≤r,可跨维组合),验证补上跨维组合
能否把 multi 往 single 拉近。

---

## 改动

`src/model.py` 新增 `value_lora_rank`,给每个 head 的 value 加 per-target 低秩
修正,用恒等式 `m_c=(W_V+B_c·A_c)·h̄_c`(h̄_c 为注意力加权源隐藏态)实现,
`B` 零初始化保证 identity 起步。三臂叠在 DNN35 最优的 dynamic_8head 上:
无 LoRA 锚点 / lora_r4 / lora_r8。

---

## 核心结果（又一个负面结果，但有局部亮点）

| 变体 | best_test_loss | LoRA 增益 (mean Δ R²) |
|---|---:|---|
| dynamic_8head (无 LoRA) | **0.2186** ← 仍最优 | — |
| lora_r4 | 0.2242 | −0.0053 |
| lora_r8 | 0.2211 | −0.0025 |

**整体仍无增益。** 但最难字段 **congestion_price_da**（probe 0.048）：
rank8 把它从 0.357 拉到 **0.406（+0.049）**,到 single 的差距从 -0.16 缩到 -0.11
——这是 DNN34/35/36 三步里对它**唯一明显的提升**,且 rank4 没用、rank8 才起效,
印证「对角太弱、低秩才够」。代价是其他字段普遍小幅退化,抵消了亮点。

---

## 三步连起来的结论

在注意力**下游**做 per-target 修正:
- DNN34 改路由 α → 整体≈0
- DNN35 对角 value(FiLM) → 整体≈0
- DNN36 低秩 value(LoRA) → 整体≈0,仅最难字段 +0.049

每步只能零星帮个别字段。瓶颈牢牢锁在**更上游的共享 backbone 表示 `h_general`**
——所有 value 都从它投影,它一旦被 12 个 target 平均化,下游再 per-target 也只是
「在贫信息上重新加权」。

---

## 下一步

要系统提升必须换战场(不再动 value,而动表示本身):
① FiLM/条件化整个 general backbone ② MoE(每 target 独立子网络)。
个别困难字段(congestion_price_da)若值得救,可单独上 LoRA-r8 或 single。
若目标是隐私评估,single 是上限,multi 用 dynamic_8head 即可。

详见 [CHANGELOG.md](CHANGELOG.md)、`outputs/lora_comparison.csv`、`outputs/lora_comparison.png`。
