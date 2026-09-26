# IJCIP / JISA 定向改写依据

调研与改写日期：2026-09-26。共同母版为 `V2_en`；原始论文为 `paper/claude/V6_ieee/main.pdf` 及其 LaTeX 源码。两篇定向稿是同一研究的备选投稿版本。

## 1. 期刊定位怎样落实到论文

| 项目 | IJCIP | JISA |
|---|---|---|
| 已核实的 scope | 关键基础设施保护，包含能源部门，关注可实际应用的保护方法及系统依赖关系 | 信息安全原创研究及实践应用，重视技术贡献、隐私、安全管理与实证 |
| 本文切入点 | 电力基础设施的信息披露如何保护非公开运行信息 | 结合背景信息后的推断泄露如何被量化、解释和验证 |
| 开篇对象 | 运营方、发布目录、机组与网络耦合、披露时序 | 数据持有方、接收者已有信息、受保护目标、字段增量泄露 |
| 核心故事 | 物理关系产生组合暴露 → 可解释字段画像 → 支持发布复核 | 单字段评估遗漏上下文 → 有预算的风险定义 → 快速扫描与验证 → 安全效果 |
| 证据组织 | 先物理机制，再真实数据和发布应用，最后集中说明计算可行性 | 先背景放大及评分，再估计器精度与成本，随后解释物理机制 |
| 方法篇幅 | 保留应用所需定义与算法；总体统计定义、读出误差分析移入附录 | 主文保留威胁模型、核心定义、读出机制、误差关系及较充分的外部比较 |
| 图的视角 | 电网、发电与负荷图形；出力—报价段—价格的关系 | 发布矩阵、目标保护、辅助信息；特征分支、逐集合读出与验证 |
| 讨论落点 | 共发布条件、发布复核记录、计算可行性 | 风险画像的安全含义、误差与召回的区别、估计成本和适用性 |

Scope 来源：[IJCIP 官方介绍](https://shop.elsevier.com/journals/international-journal-of-critical-infrastructure-protection/1874-5482)、[JISA 官方介绍](https://shop.elsevier.com/journals/journal-of-information-security-and-applications/2214-2126)。上表中的改写选择是针对本文作出的编辑判断，并非期刊规定的固定章节模板。

JISA 当前介绍明确排除主要贡献仅为在常规安全数据集上套用现成 AI/ML 模型的工作。因此，JISA 稿把“带背景预算的泄露画像及其验证”放在贡献首位，估计器服务于大规模组合评估，电力数据提供有物理结构的应用证据。IJCIP 稿则把信息保护任务和运行机制贯穿前后，不通过增加法规讨论来制造适配性。

## 2. 样刊阅读记录与具体借鉴

以下区分完整阅读、版面观察与仅摘要阅读，避免把摘要检索当成完整样刊分析。样刊用于学习组织方式，未复制其图或文字；下载的参考论文没有放进本项目交付目录。

| 论文及来源 | 阅读范围 | 对本文有用的写法 |
|---|---|---|
| Farooq et al., **Securing the green grid: A data anomaly detection method for mitigating cyberattacks on smart meter measurements**, IJCIP 46 (2024), 100694. [机构库全文](https://vbn.aau.dk/ws/portalfiles/portal/768424036/1-s2.0-S1874548224000350-main.pdf)；[DOI](https://doi.org/10.1016/j.ijcip.2024.100694) | 全文、章节及实际 PDF 图页 | 从智能电表和状态估计的运行作用引出安全问题；方法前后持续连接具体电网。图中将系统信号流与电气拓扑分开，每幅图回答一个问题。 |
| Syrmakesis et al., **DAR-LFC**, IJCIP 45 (2024), 100678. [出版方页面](https://www.sciencedirect.com/science/article/pii/S1874548224000192)；[DOI](https://doi.org/10.1016/j.ijcip.2024.100678) | 出版方提供的摘要、引言、章节安排及实验/结论节选；未取得完整 PDF 版面 | 从负荷频率控制的运行功能引出受损通信下的恢复任务，按背景—方法—实验—讨论组织。借鉴功能先行的叙述，不仿照其安全场景。 |
| Itäpelto et al., **Digital twin application in lifecycle security of critical infrastructure**, IJCIP 50 (2025), 100783. [机构库记录](https://research.utwente.nl/en/publications/digital-twin-application-in-lifecycle-security-of-critical-infras/)；[DOI](https://doi.org/10.1016/j.ijcip.2025.100783) | 摘要和书目信息；PDF 获取失败 | 用于核对近期基础设施与生命周期语境。它是综述，未将其结构当作本研究论文的模板。 |
| Ibanez-Lissen et al., **LPASS: Linear probes are stepping stones for vulnerability detection using compressed LLMs**, JISA 93 (2025), 104125. [机构库全文](https://e-archivo.uc3m.es/bitstreams/4b69ea5d-3369-4fd7-a9ff-da8e48feeddc/download)；[DOI](https://doi.org/10.1016/j.jisa.2025.104125) | 全文、研究问题与贡献、章节组织、实验讨论和 PDF 图页 | 在方法前明确任务和计算瓶颈；区分效果与资源代价；以数据流、分支和模型组件解释流程。其相关工作后置说明这种安排可行，并不表示 JISA 强制后置。 |
| Jiang et al., **Exploiting attribute correlation for reconstruction attacks on differentially private multi-attribute data**, JISA 94 (2025), 104224. [出版方页面](https://www.sciencedirect.com/science/article/pii/S2214212625002613)；[DOI](https://doi.org/10.1016/j.jisa.2025.104224) | 出版方摘要、部分正文和威胁模型图的说明；未取得完整 PDF | 用于核对属性关联、数据持有方与攻击者背景信息这一安全问题的近期表述。本文不继承其差分隐私设置或攻击结论。 |
| **Statistical privacy protection for secure data access control in cloud**, JISA 84 (2024), 103823. [机构库记录](https://publications.polymtl.ca/58893/)；[出版方页面](https://www.sciencedirect.com/science/article/pii/S2214212624001261) | 摘要和元数据；作者站 PDF 获取失败 | 补充确认面向数据使用过程的隐私保护属于该刊应用语境；未据此宣称掌握其全文结构。 |

语言方面，IJCIP 稿让段落围绕“保护什么运行信息、为什么会暴露、证据如何帮助复核”展开；JISA 稿让段落围绕“安全问题、量化对象、方法作用、评估结论”展开。共同删除过程性实验记录口吻，避免把配置尝试逐项叙述为主线。样刊中的框图仍然存在，值得借鉴的是明确的对象和关系、图形化组件与简洁标注，而不是简单禁止方框。

## 3. 作者要求：已核实与尚待核实

两个 ScienceDirect **Guide for Authors** 页面在本环境中返回 HTTP 403：

- [IJCIP 作者指南](https://www.sciencedirect.com/journal/international-journal-of-critical-infrastructure-protection/publish/guide-for-authors)
- [JISA 作者指南](https://www.sciencedirect.com/journal/journal-of-information-security-and-applications/publish/guide-for-authors)

因此，没有把第三方网站列出的字数上限、关键词上限、匿名审稿模式或附件强制性写成已核实事实。稿件使用下列已核实的 Elsevier 通用准备方式；期刊特定硬性限制仍需在实际投稿时对照可访问的指南/投稿系统确认。

| 项目 | 官方依据 | 本次落实 |
|---|---|---|
| LaTeX | [Elsevier LaTeX instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions) 提供 `elsarticle`，要求按具体期刊核对参考文献样式 | 两篇使用 `elsarticle` 单栏稿件版及数字引用，保留可编辑 LaTeX；数字引用也与已读样刊一致 |
| 源文件组织 | 同一官方说明明确 Editorial Manager 不处理子目录结构 | 工作目录按模块维护，另附脚本生成扁平的 `submission_sources.zip` |
| Highlights | [Elsevier Highlights 说明](https://www.elsevier.support/publishing/answer/how-do-i-include-highlights-with-my-manuscript)：提供时为 3–5 条，每条不超过 85 字符（含空格），作为独立可编辑文件 | 每刊独立写 4 条，交付 TXT 和 DOCX，实际字符数见本版 `VALIDATION.json` |
| 图 | [Elsevier artwork sizing](https://www-prod.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing) 强调最终尺寸下的文字可读性及一致性 | 输出矢量 PDF、可编辑 SVG 和预览 PNG；统一颜色、面板号、轴标和标记，检查嵌入正文后的版面 |
| 图文摘要 | [Elsevier graphical abstract](https://www.elsevier.com/researcher/author/tools-and-resources/graphical-abstract) 为通用说明，不能据此判定每刊必须提供 | 当前不额外制造图文摘要；概览图已可作为后续制作的基础 |
| 摘要、关键词与篇幅 | 本次未完整核实期刊各自的硬性数字限制 | 自主采用摘要少于 250 英文词、6 个关键词；主文和附录充分展开，不声称存在已核实的页数豁免 |

作者姓名沿用原文。单位、通信作者信息、基金、利益冲突、作者贡献以及最终数据/代码链接未编造。这些投稿元信息与实际使用 AI 的披露应由作者在选定期刊后据实补齐。两个定向版本用于择刊比较，不代表两项独立研究，也未进行任何投稿操作。

## 4. 文件入口

- [通用 V2 说明](V2_en/README.md)
- [IJCIP V1 说明](IJCIP/V1/README.md)
- [JISA V1 说明](JISA/V1/README.md)
- [图形与实验来源对应](V2_en/FIGURE_SOURCES.md)

每版的 `VALIDATION.json` 记录最终编译、引用、图数、摘要长度和数据继承检查结果。原始 V6 文件哈希记录在 `V2_en/source_manifest.json`。
