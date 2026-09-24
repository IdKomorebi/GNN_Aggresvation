# Codex V1_en：基于 Claude V4_en 的叙述与图形优化版

本版本由 **Codex 基于 `paper/claude/V4_en/main.tex`、各章节源文件及 `main.pdf` 优化形成**。原稿的研究工作、实验数据与作者署名予以保留；Codex 的工作范围是论文结构、英文写作、图形设计及排版整理。本目录是独立版本，未覆盖 Claude 原稿。

- 阅读论文：[main.pdf](main.pdf)
- LaTeX 入口：[main.tex](main.tex)
- 修改说明：[REVISION_NOTES.md](REVISION_NOTES.md)
- 全部图形预览：[figure_gallery.html](figure_gallery.html)
- 图形与数据来源：[FIGURE_SOURCES.md](FIGURE_SOURCES.md)
- 原稿文件校验：[source_manifest.json](source_manifest.json)
- 构建与交付检查：[VALIDATION.md](VALIDATION.md)

## 本次优化

标题改为 **Budgeted Marginal Inference for Field-Level Risk Grading of Power-System Data**。用 Power-System Data 覆盖电网算例、实际机组出力和系统运行记录；标题突出核心度量，计算加速作为支撑方法在摘要和正文展开。

全文围绕“组合泄露被单字段评价漏判 → 用背景下的最大边际贡献评价字段 → 用共享表示与逐子集读出提高计算效率 → 通过实验检验实际作用”展开。原稿四个独立 Results 章节合并为一个实验结果章，保留四个清楚的分节。六项性质对照移至附录，全列重构预训练作为扩展实验呈现。重写摘要、引言、讨论和结论，并清理研究日志式、自我评价式及重复表达。

全部九张图重新绘制。方法图采用电网拓扑、风机/负荷/保护符号、字段组合、矩阵、神经网络、读出参数等图形元素；实证图采用统一的配色、字形、坐标和面板层次。图形以矢量 PDF 嵌入论文，同时保留可编辑 SVG、Python 图源和 PNG 预览。各图的数据来自现有实验输出，没有重新训练模型。示意拓扑、波形和网络节点不是实测数据，图注已区分。

采用 IEEEtran journal 双栏版式。图形画布宽 7.16 英寸，正文以约等宽的双栏宽度置入；以 9 pt 为基础字号，部分刻度和图例使用 8–8.5 pt。PDF 字体嵌入，正文使用 Times 字体族，图中使用 Arial 度量兼容的 Liberation Sans，并辅以数学字形。

本次参考 IEEE 官方图形说明，重点落实矢量格式、双栏宽度、字体嵌入和一致的字号；这是一份期刊风格的完整修订稿，未指定某一本期刊的投稿页数及额外模板要求：

- [IEEE：Resolution and Size](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/)
- [IEEE：File Formatting](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/file-formatting/)

## 范围

沿用 V4_en 的四个数据集、十个目标、124,425 个字段集合及主要实验结论。**不纳入 120、121 号实验，也不新增实验。** 未改动数据、训练或评估代码。对少量超出实际所述范围的句子作简洁措辞调整；不增加方法问题讨论、补救实验或新的理论论证。具体文字调整见修改说明。

原文中的作者姓名予以保留，单位与联系方式未虚构；正式投稿时可在 `main.tex` 补充。

## 编译和重画

仅重新编译 PDF（不需要实验数据或 Python）：

```bash
cd paper/codex/V1_en
./build.sh
```

需要安装 Tectonic；首次运行可能需要联网下载 IEEEtran 等 TeX 资源。脚本固定进行两次 TeX 重跑，以避免 Tectonic 0.15 对 IEEEtran `.bbl` 的重复变更提示。保留 `main.bbl` 便于移交，编译时仍从 `ref.bib` 生成参考文献。

在完整项目中重新生成全部图形后编译：

```bash
PAPER_PYTHON=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python \
  bash paper/codex/V1_en/build.sh --figures
```

也可将 `PAPER_PYTHON` 设置为具备 NumPy、pandas、Matplotlib、scikit-learn、SciPy 和 PyTorch 的其他 Python。估计器图复用原项目 `pipe.py` 的闭包和边际表计算函数，仅处理保存的数组，不训练网络。脚本自动从本目录向上定位项目根目录，不依赖当前工作目录；`PYTHONDONTWRITEBYTECODE=1` 避免在原实验目录写入缓存。

已有 PDF/SVG 图形和 LaTeX 源文件足以独立编译；只有重画实证图时才需要完整实验输出。PNG 是阅读预览，论文使用矢量 PDF。
