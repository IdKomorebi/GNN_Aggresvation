# 《高级图论》课程论文 V3（当前推荐版本）

《面向数据推断泄露的任意字段集合敏感度评估图 oracle》（作者：杜浩存）。

## V3 相对 V2 的核心改动（按用户反馈）

1. **重新定位主线**：论文核心是"任意集合敏感度的 oracle 模型"，协同发现降为 oracle 的
   两个下游应用之一。标题、摘要、贡献、章节全部围绕 oracle 组织。
2. **方法讲透**：引言后直接进方法。方法章（第 4 节）为最大篇幅，含框架总览图（先放）、
   模型结构示意图（可见性感知消息传递）、11 个编号公式、算法 1 伪代码，逐步骤讲清
   输入编码 / 可见性注意力重归一化 / 随机子集失活训练 / K 步查询微调 / 扫描认证。
3. **实验大幅扩充**：新增 4 张真实数据性能图，实验按 RQ1–RQ4 组织，RQ1（oracle 保真度
   与成本）为最大实验节，含校准散点、逐规模贴合、成本-保真度权衡、种子稳健性。
4. GNN vs MLP 结构对比压缩为 RQ2 消融的一段 + 一张图。
5. 作者名改为**杜浩存**。

## 目录结构

```
V3/
├── main.tex   # 论文正文（CjC 单栏模板，XeLaTeX）
├── ref.bib    # 参考文献 14 篇（gbt7714）
└── figures/   # 7 个图位、8 个文件
    ├── framework_zh.png          # 图1 方法总览（oracle 为中心，新绘制）
    ├── model_arch_zh.png         # 图2 可见性感知消息传递结构示意（新绘制）
    ├── oracle_calibration_zh.png # 图3 oracle 估计 vs 重训真值 校准散点（★核心性能图，新）
    ├── fidelity_by_size_zh.png   # 图4(a) oracle 紧贴逐规模攻击上限（新）
    ├── cost_seed_zh.png          # 图4(b) 成本-保真度权衡 + 种子稳健性（新）
    ├── attacker_compare_zh.png   # 图5 图 vs MLP 消融
    ├── synergy2_heatmap.png      # 图6 二阶协同热力图（68号原图）
    └── triple_certify_zh.png     # 图7 三阶认证稀疏性
```

## 论文结构

1. 引言（威胁背景、集合函数视角、评估难题、oracle 核心思想、贡献）
2. 相关工作（简洁，3 段）
3. 威胁模型与问题形式化（定义 1–2、置零口径、噪声底）
4. **方法（核心大章）**：4.1 总览（图1）；4.2 字段相关图；4.3 子集条件图 oracle（图2 +
   式 3–6）；4.4 随机子集失活训练（式 7）；4.5 K 步查询微调（式 8）；4.6 扫描—认证协议（算法1）
5. **实验**：RQ1 oracle 保真度与成本（图3/图4/表1，最大节）；RQ2 消融图vsMLP（图5）；
   RQ3 应用一二阶协同 + 置零否定（图6/表2）；RQ4 应用二三阶扫描认证（图7）
6. 讨论（适用边界、迁移性、低阶稀疏、局限）
7. 结论

## 编译

服务器无 LaTeX。到 Overleaf 打开 CjC 模板
（https://www.overleaf.com/latex/templates/zhong-wen-ji-zhu-bao-gao-latexmo-ban-dan-lan-cjc-xelatex/tcnttxfsqykx），
用本目录 `main.tex` 替换主文件，上传 `ref.bib` 与整个 `figures/`，编译器选 XeLaTeX。
注：V3 用到 `algorithm` + `algpseudocode` 宏包（Overleaf 自带），若模板报冲突可把算法 1
环境替换为普通 itemize 列表。

## 数据溯源（全部数字来自真实实验输出）

- 校准散点（图3）：`DNN_Aggresvation69/outputs/est/gnn/*.json`（逐条估计）+ `truth_long.csv`（真值），
  池化 K0: MAE 0.098 / Spr 0.897；K200: MAE 0.039 / Spr 0.948
- 保真度表/逐规模/成本（表1、图4）：`oracle_fidelity.csv`（arch=gnn, reference=dnn_full）
- 种子稳健性：`k0_seed_robustness.csv`
- 图 vs MLP（图5）：`attacker_upper_bound.csv`
- 二阶协同（图6、表2）、置零对比（0.351 / 3.4×）：`DNN_Aggresvation68/outputs/synergy2_top_pairs.csv` + 68号 CHANGELOG
- 三阶扫描认证（图7；13244、0.795、314/2364 vs 19/2400、0.455）：68号 `triples_top.csv` / `triples_certified.csv`
- 超参数：`DNN_Aggresvation69/base.yaml`、`src/oracle.py`

## 口径诚实性说明（未夸大）

- oracle 角色下 MLP 绝对误差更低这一事实，在 RQ2 与讨论中如实承认；图结构优势限定在
  "≥5 字段专用推断" 与 "排序保真度足够支撑扫描"。
- 迁移性无实验，写为分析性论证 + 未来工作。
- 图3 校准散点 K=200 的池化 Spearman=0.948（全类别混合），与表1 分类别 Spearman（0.97+）
  口径不同：前者跨类别混合、后者类别内排序，均为真实计算值，正文已分别标注。
```
