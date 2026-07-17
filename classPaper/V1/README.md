# 《高级图论》课程论文 V1

基于图结构的电网数据任意字段集合敏感度评估与高阶协同泄露推断。

## 目录结构

```
V1/
├── main.tex            # 论文正文（CjC 单栏模板，XeLaTeX）
├── ref.bib             # 参考文献（gbt7714）
└── figures/
    ├── attacker_compare_zh.png   # 图1 GCN vs MLP 专用攻击者（69号 attacker_upper_bound.csv）
    ├── oracle_fidelity_zh.png    # 图2 图oracle 微调保真度（69号 architecture_comparison.csv, dnn_full口径, gnn列）
    ├── synergy2_heatmap.png      # 图3 二阶协同热力图（直接复制自 68号 outputs/）
    └── triple_certify_zh.png     # 图4 三阶认证稀疏性（68号 triples_certified.csv）
```

## 编译方法

服务器无 LaTeX 环境。到 Overleaf 打开 CjC 模板
（https://www.overleaf.com/latex/templates/zhong-wen-ji-zhu-bao-gao-latexmo-ban-dan-lan-cjc-xelatex/tcnttxfsqykx），
用本目录 `main.tex` 替换模板的主文件，上传 `ref.bib` 与 `figures/` 整个目录，
编译器选 XeLaTeX。

## 数据溯源（论文中全部数字均来自真实实验输出）

- 攻击者对比、胜率、包络增益：`DNN_Aggresvation69/outputs/attacker_upper_bound.csv`
- 图 oracle MAE / Spearman / 耗时：`DNN_Aggresvation69/outputs/architecture_comparison.csv`（reference=dnn_full）
- 二阶协同 top 对、置零口径对比（0.351 / 3.4×）：`DNN_Aggresvation68/outputs/synergy2_top_pairs.csv` 与 68号 CHANGELOG
- 三阶扫描认证（13244 全扫、Spearman 0.795、314/2364 vs 19/2400、最强 0.455）：
  `DNN_Aggresvation68/outputs/triples_top.csv`、`triples_certified.csv`
- 噪声底 σ≈0.002–0.004、显著标尺：60号
- 图生成脚本：会话 scratchpad `gen_figures.py`（如需改图可向 Claude 索要或重新生成）

## 论文口径说明（重要）

- “图结构比 MLP 准”指的是**专用攻击者**角色（≥5 字段随机集合平均 R² 为正、
  胜率随规模升至 90%），字段对/三元组审计样本上 GCN 更差的数字在表 1 中如实保留。
- 摊销 oracle 角色下 MLP 绝对误差更低这一事实，在第 6 节讨论中以一句承认，
  论文主体报告图 oracle 自身的保真度（排序 Spearman ≥0.97，足以支撑扫描协议）。
- **迁移性没有实验**，论文中写为分析性论证 + 未来工作，不是实验结论。
