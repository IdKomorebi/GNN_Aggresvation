# 《高级图论》课程论文 V2

基于图结构的电网数据任意字段集合敏感度评估与高阶协同泄露推断（作者：杜浩存）。

V2 相对 V1 的改动：按正式学术论文标准重写（参照 IGNN/IEEE TSG 与《计算机学报》论文结构），
方法章为核心（含 12 个编号公式、算法 1 伪代码、框架总览图），实验按 RQ1–RQ4 组织；
GNN vs MLP 结构对比压缩为 RQ4 一段 + 一张图。

## 目录结构

```
V2/
├── main.tex            # 论文正文（CjC 单栏模板，XeLaTeX）
├── ref.bib             # 参考文献 14 篇（gbt7714）
└── figures/
    ├── framework_zh.png          # 图1 三阶段框架总览（新绘制）
    ├── synergy2_heatmap.png      # 图2 二阶协同热力图（68号 outputs/ 原图）
    ├── oracle_fidelity_zh.png    # 图3 oracle 微调保真度（69号 architecture_comparison.csv, dnn_full, gnn 列）
    ├── triple_certify_zh.png     # 图4 三阶认证稀疏性（68号 triples_certified.csv）
    └── attacker_compare_zh.png   # 图5 GCN vs MLP 攻击者（69号 attacker_upper_bound.csv）
```

## 论文结构

1. 引言（威胁背景、动机实例、三大挑战 C1–C3、贡献 4 条、路线图）
2. 相关工作（4 小节 + 表1 方法族对比）
3. 威胁模型、问题形式化与预备知识（定义 1–4、置零口径、噪声底、GAT 预备）
4. 方法（核心章）：4.1 框架总览；4.2 字段相关图（五度量融合 τ=0.15/top-8）；
   4.3 可见性感知图推断模型（式 7–10）；4.4 子集条件 oracle（式 11）；
   4.5 K 步查询微调（式 12）；4.6 扫描—认证协议（算法 1 + 两个可检验前提 + 复杂度）
5. 实验（RQ1 协同存在性与置零失效；RQ2 oracle 保真度；RQ3 三阶扫描认证；RQ4 图结构收益）
6. 讨论（适用边界、迁移性、低阶稀疏猜想、局限 3 条）
7. 结论

## 编译方法

服务器无 LaTeX。到 Overleaf 打开 CjC 模板
（https://www.overleaf.com/latex/templates/zhong-wen-ji-zhu-bao-gao-latexmo-ban-dan-lan-cjc-xelatex/tcnttxfsqykx），
用本目录 `main.tex` 替换主文件，上传 `ref.bib` 与整个 `figures/`，编译器选 XeLaTeX。

## 数据溯源与口径说明（同 V1，全部数字来自真实实验输出）

- 攻击者对比 / 胜率 / 包络增益：`DNN_Aggresvation69/outputs/attacker_upper_bound.csv`
- oracle MAE / Spearman / 耗时：`DNN_Aggresvation69/outputs/architecture_comparison.csv`（reference=dnn_full）
- 二阶协同 top 对、置零口径（0.351 / 3.4×）：`DNN_Aggresvation68/outputs/synergy2_top_pairs.csv` + 68号 CHANGELOG
- 三阶扫描认证（13244、0.795、314/2364 vs 19/2400、0.455）：68号 `triples_top.csv`、`triples_certified.csv`
- 超参数（相关图 5 度量/τ=0.15/top-8；攻击者 4 层 64 维、Adam 1e-3、wd 5e-4、batch 128、
  500 epoch、patience 150；oracle 掩码拼接）：`DNN_Aggresvation69/base.yaml` 与 `src/oracle.py`
- 迁移性无实验，论文写为分析性论证 + 未来工作；oracle 角色下 MLP 绝对误差更低的事实在 6.1 节如实承认。
