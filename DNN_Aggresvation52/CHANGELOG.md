# DNN_Aggresvation52 日志

## Modify by Claude: 2026-06-26

## 1. 子项目定位

验证一个假设:**当前的输入编码(MLP)是否太强,把 GAT / 动态权重的能力盖住了**——
即"如果减弱输入编码,让图聚合承担更多工作,GAT/动态先验是否会拉开优势"。

做法:**聚合方式(6) × 输入编码强度(3)** 的网格扫描,只看 12-conf 平均 test R²。

### 聚合 = 2×3 网格 {gcn,gat} × {无先验, 全局静态先验, 全局动态先验}
| mode | 是否用GAT(q·k) | 相关系数权重 | 机制 |
|---|---|---|---|
| gcn_noprior | 否 | 无 | 均匀邻接行归一聚合 raw h |
| gcn_static | 否 | 全局静态 | 先验加权固定邻接 |
| gcn_dynamic | 否 | 全局动态 | 先验边 × 逐样本 sigmoid 门控(调制每条先验边强度),无 q·k |
| gat_noprior | 是 | 无 | 纯 q·k 注意力 |
| gat_static | 是 | 全局静态 | q·k + 单一全局 scale×log先验(no_gate) |
| gat_dynamic | 是 | 全局动态 | 逐对 gate 混合 q·k 与先验 |

`gcn = 无 q·k 内容注意力(结构化聚合 raw h);gat = q·k 注意力聚合 value(h)`。

### 编码强度(3 档,从弱到强)
- `enc1_linear_noact`:单线性、无激活(最弱)
- `enc2_linear`:单线性 + ReLU
- `enc3_mlp`:2 层 MLP(当前默认,最强)

## 2. 目录结构
```
outputs/
  enc1_linear_noact/ | enc2_linear/ | enc3_mlp/     ← 每档编码一个文件夹
    <6 个 mode 子文件夹>/
        per_target_r2.csv, summary.json, model.pt
        vs_probe.png/csv          ← 每个方法都有：该方法 vs DNN probe 上界
        alpha_bar.png             ← 静态先验(gcn_static/gat_static)：5 个全局静态权重柱状图
        gate_by_edge_long.png     ← 动态先验(gcn_dynamic/gat_dynamic)：逐边门控长图
        attention_report.md + attention_entropy.png  ← GAT：注意力是否起作用/是否很平均
    comparison_6methods.png + comparison.csv         ← 该档 6 方法对比
  _summary/
    encoder_effect.png   ← 每方法 R² 随编码强度(核心图)
    heatmap.png          ← mode×encoder 热图
    spread.png           ← 每档 6 方法极差
    summary.csv
```

## 3. 核心结果(12-conf 平均 test R²，DNN probe 上界 0.8937)

| mode | enc1 弱(linear_noact) | enc2 中(linear) | enc3 强(mlp) |
|---|---:|---:|---:|
| gcn_noprior | 0.8526 | 0.8579 | 0.8737 |
| gcn_static | 0.8535 | 0.8653 | 0.8664 |
| **gcn_dynamic** | **0.8646** | **0.8667** | **0.8746** |
| gat_noprior | 0.8244 | 0.8509 | 0.8506 |
| gat_static | 0.8382 | 0.8532 | 0.8523 |
| gat_dynamic | 0.8470 | 0.8552 | 0.8727 |
| **6 方法极差** | **0.0402** | 0.0158 | 0.0240 |

(enc3 列与 DNN51 完全一致:gcn_noprior 0.8737 / gat_dynamic 0.8727 / gat_static 0.8523,验证可信。)

## 4. 关键发现:假设不成立,结论与预期相反

### 4.1 减弱编码**没有**让 GAT 起飞,反而让它崩
弱编码下 6 方法极差最大(0.0402),但**不是因为 GAT 拉开优势,而是 GAT 塌下去**:
enc1 下 4 个 GCN/动态 方法都 ≥0.8526,而 3 个 GAT 方法 0.8244~0.8470 全垫底,
gat_noprior 最差(0.8244)。**每一档编码下,同等先验类型 GCN ≥ GAT。**

### 4.2 纯 q·k 注意力退化为近均匀(熵证据)
gat_noprior 的注意力归一化熵:0.834(弱) → **0.944 / 0.948**(中/强),
中/强编码下 **7~8/12** 个 confidential 目标注意力 ≥0.95(几乎完全均匀)。
→ **q·k 没学出有用的邻居偏好,退化为均匀聚合**(所以 gat_noprior≈gcn_noprior)。
gat_static/gat_dynamic 熵更低(0.71~0.86),但那是**先验把注意力压集中的**,不是
q·k 自己学到的。

### 4.3 「动态权重」的好处在 GCN 侧,不在 q·k 注意力
- **gcn_dynamic 在每一档都是最好或并列最好**(0.8646/0.8667/0.8746),且最接近 probe。
- 动态 > 静态在两支都成立:gcn_dynamic > gcn_static(各档 +0.001~+0.011);
  gat_dynamic > gat_static(各档 +0.002~+0.020)。**动态先验门控始终有用,且没有被
  强编码盖住**——与"强编码掩盖动态能力"的假设相反。

### 4.4 GAT 的 q·k 需要强编码喂养(模型自己学会了)
gat_dynamic 的动态门控均值(gate→1 信 q·k,→0 信先验)随编码增强而上升:
**0.276(弱) → 0.346(中) → 0.441(强)**。即**编码越弱,模型越不信任 q·k、越靠先验**;
编码越强,q·k 特征越丰富,模型才敢用它。所以 gat_dynamic 只有在强编码(0.8727)才追上
GCN 簇,弱编码(0.8470)远落后于 gcn_dynamic(0.8646)。**强编码是在帮 GAT,不是在掩盖它。**

## 5. 回答你的假设

**「输入编码太强,盖住了 GAT/动态权重」——不成立,而且反了。**
1. 动态权重(先验门控)的好处在所有编码强度下都存在,且强编码并未掩盖它;最佳实现是
   **gcn_dynamic**(GCN 结构 + 逐样本门控调制先验,无 q·k)。
2. 真正被"编码强度"左右的是 GAT 的 q·k 注意力:它需要强编码喂丰富特征才勉强可用,
   弱编码下直接退化为近均匀、拖垮性能。所以减弱编码不会释放 GAT,只会暴露 q·k 在本
   任务上不是有效机制。
3. 一句话:**有用的是"动态先验门控",没用的是"q·k 内容注意力";前者用 GCN 就能拿到
   (gcn_dynamic 最佳),后者削弱编码也救不回来。**

## 6. 对项目的启示
- **gcn_dynamic 可能是更好的主干**:准确率每档最高(强编码 0.8746,最接近 probe 0.8937),
  且它的**逐样本门控**本身就是一份可解释的"动态推断路径"信号(每条先验边被放大/抑制
  多少),不依赖噪声大的 q·k。即"动态权重 + 可解释路径"可以不靠 GAT 拿到。
- 若叙事仍需"注意力崩溃"的反例(DNN50)或 q·k 形式的路径,可保留 gat_dynamic 作对照;
  但论文主张应改为:**推断驱动的动态边门控(gcn_dynamic)** 而非 q·k 注意力。

## 7. 局限
单 seed。4.1/4.3 幅度大、跨 3 档方向一致,基本不受 seed 影响;enc1 各方法的细小排序
(如 gcn_static vs gcn_noprior)在噪声内,不宜过度解读。

## 8. 复跑
```
scripts/scheduler.py     # 18 训练(3 编码 × 6 聚合，4 卡)
scripts/diagnose.py --encoder <e> --mode <m>   # 单方法诊断产物
scripts/compare.py       # 每档对比图 + _summary 跨编码汇总
```
