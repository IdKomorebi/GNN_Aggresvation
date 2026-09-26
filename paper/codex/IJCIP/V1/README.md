# IJCIP 定向稿 V1

**Protecting power-system information releases from background-dependent inference**

本稿由 Codex 依据 `paper/codex/V2_en` 总体优化版独立改写，研究内容原始来源是 `paper/claude/V6_ieee`。面向 *International Journal of Critical Infrastructure Protection*，重点是电力基础设施的信息保护及发布复核。

[论文 PDF](main.pdf) · [LaTeX 主文件](main.tex) · [图形预览](figure_gallery.html) · [Highlights](highlights.docx) · [调研依据](../../JOURNAL_ADAPTATION_RESEARCH.md) · [数据与图形来源](../../V2_en/FIGURE_SOURCES.md)

## 相对通用版的实质调整

开篇从运营方为什么发布价格、预测和运行计划讲起，再说明出力、负荷、网络约束和价格共同描述运行状态，单项披露如何变成组合暴露。摘要首先说明被保护的信息和披露任务，计算方法作为使审计可反复实施的手段。

贡献顺序为：字段画像与组合证据、物理解释及跨系统实证、发布复核及计算可行性。没有把应用论文改写成法规解释文章，也没有引入未经实验支持的实际部署效果。

受控机制位于结果开头，新增原理图解释“出力识别报价段，价格约束私有加价，拥塞改变远端价格关系”。真实数据的背景放大接在该机制后面，发布决策应用并入同一证据章节。完整计算比较放在后续独立章节，使读者先理解保护价值，再判断重复审计是否可行。

Discussion 重写为三个实际问题：字段等级如何附带共发布条件；画像如何进入发布复核；不同审计规模下如何选择估计配置。保留主要解释性定理，将总体统计定义和额外读出分析放到附录。外部估计器比较保持完整数值表和图，缩短正文中各替代模型的实现描述。

图 1 使用发电、负荷和电网连接等对象；方法图保留必要的模型结构。所有图是可编辑矢量图，数值沿用通用版。增加两篇直接相关的 IJCIP 论文作为应用背景，未为了刊名大量堆砌自刊引用。

## 本稿结构

1. Introduction
2. Information Protection and Related Work
3. A Field Profile for Release Review
4. A Practical Audit Procedure
5. Experimental Setup
6. Physical Mechanism and Disclosure Evidence
7. Computational Feasibility of Repeated Audits
8. Implications for Infrastructure Information Protection
9. Conclusion

附录保留实现、定理扩展、完整设计诊断、额外指标及最小不安全集合统计，另增加主文移出的总体定义与估计分析。章节顺序是针对本稿及样刊观察作出的安排，不是声称期刊强制要求。

## 投稿材料状态与复用

采用 Elsevier `elsarticle` 单栏预印本格式，摘要少于 250 词、6 个关键词、数字参考文献；单栏稿方便审阅，不仿造出版后的双栏成品。独立 Highlights 提供 TXT 和 DOCX，两者内容相同。`submission_sources.zip` 将 TeX、BibTeX 和矢量图扁平打包，便于后续 Editorial Manager 使用。

作者指南页面在本环境返回 403；当前期刊专属字数限制、匿名要求、Highlights 是否强制等尚未完整核实，详见[调研记录](../../JOURNAL_ADAPTATION_RESEARCH.md)。本稿为完整内容初稿，作者单位、联系方式、基金、利益冲突、贡献说明和最终数据代码链接仍需据实补齐。没有提交期刊。

```bash
cd paper/codex/IJCIP/V1
bash build.sh
python scripts/package_sources.py
```

重绘可用 `PYTHON=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python bash redraw.sh`，需完整项目的既有实验输出。普通论文编译只需随稿图和 TeX 环境。

`VALIDATION.json` 记录本版编译、引用、摘要和图片检查。后续 IJCIP 修订请从本目录派生 `IJCIP/V2`，同时记录与通用母版的研究内容差异。
