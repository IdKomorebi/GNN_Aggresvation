# 高级图论课程论文（Codex V3）

主文件为 main.tex，参考文献数据库为 ref.bib，编译用条目镜像为 references.tex，图表位于 figures/。

## 作者信息

作者与单位已按 Claude V3 的版式填写为“杜浩存 / 西安交通大学 / 西安 710049”，可在 `main.tex` 顶部修改。

## 论文主线

论文按“引言 → 问题与图论建模 → 任意集合敏感度 Oracle → 实验 → 讨论 → 结论”展开：

1. 核心对象是任意字段集合的推断敏感度 \(v_c(S)\)；
2. 专用 DNN/GCN 用于审计固定集合的经验攻击上限；
3. 随机掩码通用 Oracle 用于低成本扫描 \(2^{44}\) 个潜在集合；
4. 二阶协同边和三阶超边由同一集合函数导出；
5. 讨论围绕图结构的有效区间、协同图的治理意义与跨系统扩展展开。

## V3 修改

- 采用 Claude V3 的 CjC 风格：黑色章节标题、页首信息栏、紧凑作者区和手工摘要版式；
- 使用 TikZ 重新绘制子集条件图模型结构，PDF 中保持矢量清晰度；
- 删除“复现实验口径”附录，参考文献后直接结束全文；
- 继承 Codex V2 已重写的讨论与结论。

## 重新生成图表

在项目根目录执行：

    MPLCONFIGDIR=/tmp/codex-mpl-config \
    conda run --no-capture-output -n Pytorch310_codex \
    python classPaper/Codex/V3/scripts/make_figures.py

脚本只读取 DNN60、DNN67、DNN68、DNN69 已有的 CSV 结果，不会重新训练模型；模型结构图由 `figures/model_arch.tex` 绘制。

## 编译

推荐 Tectonic（本目录已验证）：

    tectonic main.tex --keep-logs --keep-intermediates

也可使用 XeLaTeX：

    xelatex main.tex
    bibtex main
    xelatex main.tex
    xelatex main.tex

本文使用 `ctexart` 复现 Claude V3 的 CjC 单栏视觉格式，避免依赖当前工作区中缺失的 `CjC.cls`。
