# V3_en：英文 SCI 稿（102–109 号主线）

**《Inference-Aware Field Grading for Power Market Data Release: Amortized Subset
Inferability and Budgeted Marginal Risk》**（作者：杜浩存）

## 与 V1 / V2 的关系

V1/V2 讲的是 91→95 号收束出的故事线（摊销层次错配 → 闭式读出，中文《计算机学报》模板）。
本版是 **102–109 号新主线**，故事不同：

| 维度 | V1 / V2 | **V3_en** |
| --- | --- | --- |
| 主问题 | 通用模型能不能估准集合敏感度 | 字段能不能发布（集合级能力 → 字段级风险 → 发布决策） |
| 核心量 | V(S) 的保真度 | **M^(K)**：带公开基底 B 与攻击预算 K 的最大边际推断能力 |
| 理论 | 无 | 截断 super-efficiency / 安全余量刻画 / 关键性⇔小 MUS（三定理带证明） |
| 落点 | 摊销层次 | **发布审查**：单字段规则放行 19%/56% 危险集合，逐阶预算零危险放行 |
| 语言模板 | 中文 CjC | 英文 elsarticle |
| 数据 | PJM + CAISO | 同（41 字段 × 12 目标，协议冻结） |

闭式读出在本版中降为 §4 的一个组件（RQ-A2 消融），不再是主线。

## 编译

本机无 LaTeX 环境，需在 Overleaf 或装有 TeX Live 的机器上编译：

```
pdflatex main → bibtex main → pdflatex main ×2
```

`elsarticle.cls` 与 `elsarticle-num.bst` 是 TeX Live 标准组件，无需额外下载。
正文只用标准宏包（amsmath / graphicx / booktabs / multirow / algorithm / hyperref）。

## 结构

```
V3_en/
├── main.tex          # 正文，8 节 + 2 附录
├── ref.bib           # 42 条（V2 继承 22 条 + 新增 20 条）
├── figures/          # 9 张图（pdf 用于排版，png 用于目检）
└── scripts/
    ├── make_figs.py        # fig3–fig9，直接读 DNN_Aggresvation100–109 的落盘产物
    └── make_schematics.py  # fig1–fig2 示意图
```

## 图表与实验编号的对应

| 图表 | 内容 | 数据来源 |
| --- | --- | --- |
| Fig. 1 | 审计管线示意 | — |
| Fig. 2 | 两阶段估计器示意 | — |
| Fig. 3 | V 保真度 + 按规模分层 | 103 |
| Fig. 4 | M 点估计 → 认证区间 | 104、108 |
| Fig. 5 | K 递进曲线 | 104 |
| Fig. 7 | 发布审查六规则 | 106 |
| Fig. 8 | 效率与盈亏平衡 | 100 |
| Fig. 9 | 攻击者能力 sensitivity + winner's curse | 109 |
| Table 1 | 数据与威胁模型 | FINAL_PROTOCOL |
| Table 2 | 估计器基线与消融 | 103 |
| Table 3 | M 区间与覆盖率 | 108 |
| Table 4 | τ-critical 检测 PR-AUC | 105 |
| Table 5 | 效率 | 100 |
| Table 6 | 发布审查 | 106 |
| Table 7 | 字段分级 | 106 |
| Table 8 | 攻击者能力 sensitivity | 109 |

（Fig. 6 witness 案例图尚未画，见下方待办。）

## 写作口径（与实验记录一致，不得擅自放宽）

**可主张**：摊销层次的实证（3.2×/4.6×）、M^(K) 的三条定理、35/41 与 38/41 的组合风险、
发布审查的 19.1%/56.4% vs 零危险放行、风险相对基底、全量模型归因不适合做推断风险排序。

**不可主张**：首次提出随机掩码 / 闭式读出 / 最大边际 / witness（均有前作，§2 已归属）；
摊销是 41 字段 K≤2 的唯一解（§7.5 已自证否）；M 全面优于传统指标（§8.1 已写明排序层
与单字段 M^(0) 接近，增益在决策层）；witness 可跨数据复现（§8.2 已写明 12–37%）。

## 待办

1. **Fig. 6（witness 案例）**：需从 104 号 M_table_official.csv 挑 2–3 个可解释案例
   （PJM/CAISO 各一），画"单字段安全 → 加入见证背景后越过 τ"的示意。
2. 作者单位、基金号、通讯方式待补（`\address{}` 现为占位）。
3. 投稿目标期刊确定后，按其模板调整（现为 elsarticle preprint 单栏）。
4. 可选补强（见 `DNN_Aggresvation107/outputs/analysis/GAPS.md` B 档）：p≥100 大字段空间、
   第三个电力市场、XGBoost/LightGBM 攻击器。
