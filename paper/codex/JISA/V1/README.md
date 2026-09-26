# JISA 定向稿 V1

**Auditing background-dependent inference leakage with field-level risk profiles**

本稿由 Codex 依据 `paper/codex/V2_en` 总体优化版独立改写，研究内容原始来源是 `paper/claude/V6_ieee`。面向 *Journal of Information Security and Applications*，重点是带背景信息的推断泄露及可验证的字段审计。

[论文 PDF](main.pdf) · [LaTeX 主文件](main.tex) · [图形预览](figure_gallery.html) · [Highlights](highlights.docx) · [调研依据](../../JOURNAL_ADAPTATION_RESEARCH.md) · [数据与图形来源](../../V2_en/FIGURE_SOURCES.md)

## 相对通用版的实质调整

摘要和引言从“接收者把发布字段与已有信息结合之后能够推断什么”展开，明确本文不是普通预测任务或固定模型特征解释。核心贡献首先是带预算的字段泄露画像，随后是共享特征、逐集合适配与重训练验证，最后是电力系统的充分应用实证。

主文先给出数据持有方、攻击者辅助历史、已公开字段、候选字段与受保护目标。预算、风险增量、放大和 witness 围绕同一个威胁模型展开，保留关键定理解释其安全含义。相关工作移到评估与应用之后，避免引言后的大量文献细节打断问题—方法主线。

结果先展示跨数据的背景放大和单字段遗漏，再比较评分与估计器，最后用受控电力机制解释现象。读出为何逐集合求解、误差如何传播，以及现成模型替代方案的优势和代价在主文中较充分展开。

Discussion 分别讨论画像的安全含义、审计应该衡量哪些指标及适用性。尤其区分画像误差、认证恢复量、关键字段召回：TabPFN v2 在高阈值下的召回优势被保留；本文对放大量的精度及大批集合查询的成本优势也被明确陈述。固定配置理由按作者意见写为“当前毫秒级开销可接受，采用整体画像更准确的配置；更大工作量可选重建单分支”。

图 1 的入口改为数据持有方、发布矩阵、辅助历史和受保护目标；图 2 显示重建/随机分支、按集合拼接的特征、ridge 读出和候选验证。物理机制图用于解释应用，不将论文主标题限制为电力市场。引入两篇与属性关联、实用安全计算效率相关的 JISA 论文，保持与已有研究的明确关系。

## 本稿结构

1. Introduction
2. Threat Model and Field-Level Risk
3. Amortized Scanning with Retraining-Based Verification
4. Experimental Setup
5. Evaluation Results
6. Application to Data-Release Decisions
7. Related Work
8. Discussion
9. Conclusion

附录容纳进一步性质、实现细节、目标复用、合成案例、完整设计诊断、附加指标及最小不安全集合统计。保留的技术内容围绕审计方法及其解释，不新增实验分支。

## 投稿材料状态与复用

采用 Elsevier `elsarticle` 单栏预印本格式，摘要少于 250 词、6 个关键词及数字参考文献；独立 Highlights 提供内容一致的 TXT 和 DOCX。`submission_sources.zip` 将源码、书目和 PDF 图扁平打包。这里使用技术细节较充分的第一版，为以后压缩篇幅保留完整母稿。

已核实的 JISA scope 重视原创信息安全贡献，明确排除仅把现成 AI/ML 模型用于常规安全数据集的工作；本稿因此把安全问题、测度和验证证据放在首位。此定位依据[官方期刊介绍](https://shop.elsevier.com/journals/journal-of-information-security-and-applications/2214-2126)。

作者指南页面在本环境返回 403；期刊专属字数、匿名审稿要求及附件强制性尚未完整核实，详见[调研记录](../../JOURNAL_ADAPTATION_RESEARCH.md)。单位、通信信息、基金、利益冲突、贡献说明和最终公开链接待作者据实补齐。本次没有投稿。

```bash
cd paper/codex/JISA/V1
bash build.sh
python scripts/package_sources.py
```

重绘可用 `PYTHON=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python bash redraw.sh`，需完整项目数据。普通编译只使用已交付图形与 TeX 文件。

`VALIDATION.json` 记录最终检查。后续目标期刊修订使用 `JISA/V2` 等目录，通用研究改动回溯到相应母版，避免把 IJCIP 的应用叙述直接套入 JISA。
