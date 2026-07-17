# 高级图论课程论文（Codex V1）

主文件为 main.tex，参考文献数据库为 ref.bib，编译用条目镜像为 references.tex，图表位于 figures/。

## 提交前修改

打开 main.tex，修改顶部四个课程信息命令：课程名、姓名、学号、学院与专业。

## 论文主线

论文按“引言 → 问题与图论建模 → 任意集合敏感度 Oracle → 实验 → 讨论”展开：

1. 核心对象是任意字段集合的推断敏感度 \(v_c(S)\)；
2. 专用 DNN/GCN 用于审计固定集合的经验攻击上限；
3. 随机掩码通用 Oracle 用于低成本扫描 \(2^{44}\) 个潜在集合；
4. 二阶协同边和三阶超边由同一集合函数导出；
5. 迁移性仅写为结构潜力，未伪造跨系统实验。

## 重新生成图表

在项目根目录执行：

    MPLCONFIGDIR=/tmp/codex-mpl-config \
    conda run --no-capture-output -n Pytorch310_codex \
    python classPaper/Codex/V1/scripts/make_figures.py

脚本只读取 DNN60、DNN67、DNN68、DNN69 已有的 CSV 结果，不会重新训练模型。

## 编译

推荐 XeLaTeX：

    xelatex main.tex
    bibtex main
    xelatex main.tex
    xelatex main.tex

本文使用 ctexart 实现用户给定中文单栏技术报告模板的版式，避免依赖模板网页中未附带的 CjC.cls。
