# 高级图论课程论文（Codex V2）

主文件为 main.tex，参考文献数据库为 ref.bib，编译用条目镜像为 references.tex，图表位于 figures/。

## 提交前修改

打开 main.tex，修改顶部四个课程信息命令：课程名、姓名、学号、学院与专业。

## 论文主线

论文按“引言 → 问题与图论建模 → 任意集合敏感度 Oracle → 实验 → 讨论 → 结论”展开：

1. 核心对象是任意字段集合的推断敏感度 \(v_c(S)\)；
2. 专用 DNN/GCN 用于审计固定集合的经验攻击上限；
3. 随机掩码通用 Oracle 用于低成本扫描 \(2^{44}\) 个潜在集合；
4. 二阶协同边和三阶超边由同一集合函数导出；
5. 讨论围绕图结构的有效区间、协同图的治理意义与跨系统扩展展开。

## V2 修改

- 重写第 6 章，删除解释“论文应该怎么理解”的元叙事，改为对实验发现的学术讨论；
- 将结论压缩为一个论文式总结段落；
- 补充可见性感知消息传递结构图；
- 修正 GNN 聚合公式中遗漏的加号，并统一 RQ3a/RQ3b 编号；
- 保留对迁移性的结构分析，不将未执行的跨系统实验写成实证结论。

## 重新生成图表

在项目根目录执行：

    MPLCONFIGDIR=/tmp/codex-mpl-config \
    conda run --no-capture-output -n Pytorch310_codex \
    python classPaper/Codex/V2/scripts/make_figures.py

脚本只读取 DNN60、DNN67、DNN68、DNN69 已有的 CSV 结果，不会重新训练模型；`model_arch_zh.png` 是单独保存的方法结构示意图。

## 编译

推荐 Tectonic（本目录已验证）：

    tectonic main.tex --keep-logs --keep-intermediates

也可使用 XeLaTeX：

    xelatex main.tex
    bibtex main
    xelatex main.tex
    xelatex main.tex

本文使用 ctexart 实现用户给定中文单栏技术报告模板的版式，避免依赖模板网页中未附带的 CjC.cls。
