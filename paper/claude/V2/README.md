# 论文 V1（claude 版）

《摊销层次与闭式读出：面向数据推断泄露的可认证集合敏感度审计》（作者：杜浩存）

## 与 classPaper 系列的关系

classPaper/V1–V3 讲的是「子集条件图 oracle」（GNN + 随机子集失活 + K 步微调）。
本版本讲的是其后 91→95 号实验收束出的**新故事线**，与旧版的关系是：

| 维度 | classPaper V3 | 本版 V1 |
|---|---|---|
| 主角 | 图 oracle 本身（能估任意集合） | 摊销**层次**错配的诊断与修复 |
| 核心主张 | oracle 又快又准，可支撑扫描 | oracle 在**尾部失明**，须把摊销从「值」上移到「表示」 |
| 图结构 | 核心卖点 | 已被 74 号证伪（六种聚合跨度仅 0.008），本版不再出现 |
| 结论强度 | 结构化估计即结论 | 结构化估计只是候选，**重训认证是唯一裁判** |
| 数据集 | PJM | PJM + CAISO（完整复现） |

## 结构

```
V1/
├── main.tex        # 正文（CjC 单栏模板，XeLaTeX）
├── ref.bib         # 22 篇（原 13 篇 + 本版新增 8 篇摊销/闭式/FDR 文献）
├── scripts/
│   └── make_figs.py  # 全部配图的生成脚本（直接读实验落盘文件）
└── figures/        # 9 张图
```

## 章节与图表对应

| 节 | 内容 | 图表 |
|---|---|---|
| 1 引言 | 尾部失明现象引入 | 图1 |
| 3 形式化 | $v_c(S)$、$\syn_k$、三分割、方法学红线 | 式1–2 |
| 4 诊断 | 尾部失明 + 三个排除性对照 + 值函数机制 | 图1、式3 |
| 5 方法 | 摊销表示 + 闭式读出、L0/L1、统一视角、三层管线 | 图2、式4–6、算法1 |
| 6.2 RQ1 | 方法阶梯 + 单变量受控对照 | 表1、图3 |
| 6.3 RQ2 | 成本—保真权衡 | 表2、图4 |
| 6.4 RQ3 | 高阶认证对决（字典证伪） | 图5 |
| 6.5 RQ4 | 衰减律 + 反层级 | 表3、图6 |
| 6.6 RQ5 | 配对 FDR | 图7 |
| 6.7 RQ6 | 跨种子秩过滤 | 图8 |
| 6.8 RQ7 | CAISO 第二数据集 | 图9、图3(b)、表3 |
| 6.9 | 零成本共线基线 | （文字） |

## 数据溯源（每个数字的出处）

**方法阶梯（表1、图3b）**
- PJM：`DNN_Aggresvation91/outputs/order2_truth.csv`（oracle 0.606/0.130、affine
  0.676/0.199、poly2 0.680/0.760、last 0.732/0.781）+
  `DNN_Aggresvation93/outputs/eval_l1_order2_aug8.csv`（L1 0.808/0.858）
- CAISO：`DNN_Aggresvation95_caiso/outputs/eval_l1_order2.csv` +
  `eval_l1_order2_aug8s{0,1,2}.csv`（L1 三种子 0.969/0.973/0.965）
- 三阶：`91/outputs/order3_truth.csv`、`95_caiso/outputs/eval_l1_order3*.csv`

**单变量受控对照（图3a）**：`91/outputs/readout_vs_feature_summary.csv`
（960 条分层样本 × 8 方法；oracle gap_strong 0.124、L0 0.016、ft50 0.036；
rho oracle 0.686 / L0 0.904 / ft50 0.899）

**成本（表2、图4）**：`91/outputs/bench_cost_o3.csv`（0.321 / 55.491 / 2.475 ms，
22.4× 加速），ρ 取自 readout_vs_feature_summary 同批

**高阶认证（图5、图9）**
- PJM full o5 top-40：`93/outputs/certify_o5_s*of[24].csv` + `certify_summary.csv`
  （认证均值 0.031、0 个过 0.10、子集反超 26/40=65%、MW p=0.058）
- PJM L1 o5 top-21：`93/outputs/certify_o5_l1top_s*.csv`（10/21 过 0.10，max 0.154）
- PJM L1 o4 top-20：`93/outputs/certify_o4_l1top_s*.csv`（9/20，max 0.196）
- PJM full o4 top-20：`93/outputs/certify_o4_fulltop_s*.csv`（0/20，max 0.091）
- CAISO 四组：`95_caiso/outputs/certify_o{3,4,5}_{fdrtop,l1top,fulltop}_s*.csv`
  （o4 L1 19/20=95%、o5 L1 结构化 0.0763→认证 0.0753、o5 full 0.0926→0.0534=1.73×、
  o3 FDR-top 10/10，MW p 均 <0.001）

**衰减律（表3、图6）**：`94/outputs/decay_law_certified.csv`（PJM）、
`95_caiso/outputs/decay_law_certified.csv`（CAISO，由 `scripts/cross_checks.py` 生成）

**反层级（6.5 节）**：`94/outputs/hierarchy_check_o5.csv`（10 个真协同的最好四阶
父集排名 164–2429；B=100 全漏、B=1000 漏 4、B=5000 全覆盖）

**配对 FDR（图7）**：PJM 1924→66 见 93 号 CHANGELOG；
CAISO `95_caiso/outputs/paired_fdr_o3_cat3.csv`（τ=0.10：1526→122；τ=0：9254/13244=69.9%）

**跨种子过滤（图8）**：`94/outputs/ensemble_filter_o4.csv`（真中位 105 / 假 27647，
分离 263×，阈值 500 精确率 45%→100%）、`95_caiso/outputs/ensemble_filter_o4.csv`
（真中位 20 / 假 809，分离 41×）

**共线基线（6.9 节）**：`95_caiso/outputs/closed_form_o{3,4,5}.csv` +
`scripts/closed_form_check.py` 输出（o2 尾部 0.713；AUC 1.000/0.962/0.825）

**CAISO 字段设计**：`DNN_Aggresvation95_caiso/fields_design.md`（三轮泄露审计，
最大线性泄露 0.9894 vs PJM 0.9984）

## 诚实性检查清单（正文已如实标注，未夸大）

1. L1 单种子候选精确率 PJM 仅 48%（图5 中面板的双峰形态直接画出），已在 6.4 节末与
   局限 (2) 明说；
2. 跨种子过滤阈值是在同一批认证集合上选的，需前瞻验证——6.6 节已标注；
3. 四、五阶密度是「超阈计数 × 认证精确率」的外推估计，非全认证——表3 脚注与局限 (3)；
4. φ 非通用基（合成信号上输给手工字典）写入局限 (1)，未回避；
5. CAISO 的 total_gen / interchange 只有日前计划口径，写入局限 (5)；
6. 前置工作两次错误结论（poly2 断崖、full 不衰减）在 6.5 节公开承认并解释成因；
7. 置换检验的结构盲点（曾据此误判）写入 7.1 节，未隐去。

## 编译

服务器无 LaTeX。到 Overleaf 打开 CjC 模板
（zhong-wen-ji-zhu-bao-gao-latexmo-ban-dan-lan-cjc-xelatex），用本目录 `main.tex`
替换主文件，上传 `ref.bib` 与整个 `figures/`，编译器选 XeLaTeX。
图为中文标注（matplotlib + Noto Sans CJK）；如需英文版改 `scripts/make_figs.py`
里的中文字符串重跑即可。

## 重新生成图

```bash
cd paper/claude/V1
/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python scripts/make_figs.py
```

## V1.1 增补（2026-07-30）：RQ8 可认证边界

新增 §6.8「RQ8：可认证边界 $k^\ast$ 与它的三段式成因」+ 表 4，并相应修改：
摘要（中/英）、RQ4 加「尺子随阶衰减」段、局限 (1)(4) 改写并新增 (4')、结论加末段。

**数据出处**（全部来自 `DNN_Aggresvation96/`）：

| 论文数字 | 文件 |
|---|---|
| 真值端回收率、单调性违反率 | `outputs/inject_truth*_{pjm,caiso}_dnn128_s*.csv` |
| 搜索端 φ / full 百分位 | `outputs/inject_search_m6_{pjm,caiso}_{last,full}_s0of1.csv` |
| 攻击者模型族对比（0.64/0.62/0.23） | `outputs/inject_truth*_{dnn512,hgb}_*.csv` |
| 样本量缩放（n=1200..4589） | `outputs/inject_truth*_n{1200,2400,3600}_*.csv` |
| 注入构造更正（峰度 1127→18、自检 0.028→0.29） | `scripts/inject_highorder.py` 文件头 + `WORKLOG.md` |
| 汇总与判读 | `WORKLOG.md` 末节「字典对照」 |

**诚实性检查（本次新增 3 条）**：
8. 表 4 的 full 字典对合成注入信号是**配基**（显式含生成单项式），不可外推为
   "真实数据上 full 更好"——RQ3 已认证否决。论文正文已显式标注该不可外推方向。
9. 衰减律（表 3）未重测，只加口径标注；数字本身仍来自 94/95 号认证结果。
10. $k^\ast=5$ 的搜索端证据基于每阶 6 个注入元组 × 400 个随机对照，样本偏小，
    结论强度限于"方向明确"，未做置换检验。
