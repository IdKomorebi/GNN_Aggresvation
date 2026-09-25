# V6_ieee：IEEE 双栏版（在 V5.2_en 基础上的文字修订与版式迁移）

**《Background-Aware Grading of Inference Risk in Power Market Data: Budgeted Marginal Risk and an Amortized Attacker》**

编译稿：`main.pdf`（IEEEtran journal 双栏，21 页，含 4 个附录与参考文献）。
编译：`tectonic --reruns 2 main.tex`。改数字后先 `cd scripts && python make_tables.py && python make_figs.py <图名…> && python make_schematics.py`。
模板参照 `paper/codex/V1_en`（只用版式，不用内容）。

## 一、全文检查发现并修正的问题

| 位置 | 问题 | 修正 |
| --- | --- | --- |
| 讨论 | "every decision reported here is certified by retraining"说得过满：只凭估计值选扣留字段的两种策略并未经重训认证 | 改为"扫描—认证标出的每个字段都经重训确认"；应用节标明那两种策略"without certification" |
| 结果 | 两个放大例子（SA 风电 0.19→0.60、SP15 电价 0.14→0.51）在两段里重复出现 | 第一段只保留 RTS 母线 303 的例子，其余留在"Amplification is common" |
| 结果 | "a query costs … one linear solve"不准确（本文三个种子各解一次） | 改为"three linear solves, one per seed, each shared by all targets" |
| 结果 | "V 误差下降 32%（≤3）和 32%（规模 4）"写法重复 | 合并为一句 |
| 附录 | 最小命中集引用指向排序对比节（应为应用节）；残留"disposal""escalation" | 改正引用，术语统一为 withholding / amplification |
| 全文 | "We report this plainly""keeps it honest""anyway""exactly the argument"等作者口吻 | 删去或改为中性陈述 |
| 全文 | disposal / escalation 与 withholding / amplification 混用 | 统一为 withholding、amplification（框架图同步修改） |
| 结论、局限 | 估计器描述仍是旧版；没有提到在训练行最少的数据集上系统低估 | 结论按最终估计器重写并加入外部对比结论；局限 (iii) 补"在训练行最少的数据集上偏低" |

## 二、TabPFN 这一 baseline 的描述（核查过的事实与叙述角度）

事实均已核实：
- 12 层 transformer、约 1,100 万参数、默认 8 个集成成员（从加载的模型直接统计）；
- 每个（集合，目标）单独拟合一个回归器；
- 耗时、精度来自 127 号；
- 认证后两者恢复的风险相同（前 1：0.959 对 0.961；前 3：0.979 对 0.978）。

叙述角度（按你的意见）：
- **两者"摊销"的对象不同。** TabPFN 在数据集之间摊销学习，所以每个字段集合都是一个新的预测任务：该集合对应的训练行组成新的上下文，要逐个目标送入 8 × 12 层的 transformer，前一个集合算过的东西对下一个集合毫无用处。本文在同一数据集的所有字段集合之间摊销：表示只学一次，每次查询只需几次小网络前向和三次线性求解（所有目标共用）。
- **审计恰恰需要成千上万次查询。** 每个目标、每张表都要成千上万个集合，每次重新定级还要重复一遍，所以 13–110 倍的耗时差是决定性的：NEM 的 K=2 全表要 2.7 小时，本文 5 分钟；CAISO 的规模 4 集合要 12 小时，本文 54 分钟。
- **扫描只需要排序，决策由重训认证。** 认证之后两者没有差别。

为了不被审稿人反驳，也如实写出三点：
- TabPFN 认证前的 V、M 略好；
- 它的优势集中在训练行最少的 PJM，本文在那里有系统低估；
- 它有时比重训攻击者族更强，文中把它定位为"更适合作为攻击者族的一员，只在认证背景上评估"，并在局限 (i) 中说明报告的风险对拥有此类模型的攻击者是下界。

## 三、结构与语言

- 五个 Results 章合并为一个 **Results** 节，下设五个小节：机制算例、背景放大、打分对比、摊销攻击者、稳健性与 K。这是 IEEE 期刊常见的组织方式。
- 估计器小节的顺序：保真度 → Γ 与背景召回 → **设计选择（表 VI）** → 预训练诊断 → 由诊断到设计 → **与外部方法对比（表 VII、图 9）** → 成本。"一个主干服务所有目标定义"属扩展实验，移入附录 C。
- 摘要从约 360 词压到约 240 词（IEEE 一般不超过 250 词）；引言用 \IEEEPARstart，计算动机段写清"要评估大量集合，而免训练的通用预测器每个集合都要重新处理数据"，为外部对比埋下伏笔。
- 公式按单栏宽度拆行（式 (2)(3)、预训练目标、读出特征）。

## 四、版式与图

| 项目 | 规格 |
| --- | --- |
| 模板 | IEEEtran journal，双栏，Times 正文 |
| 图 | 全部跨栏（figure*，排版宽 7.16 英寸）；统计图画布 6.5 英寸、放大约 1.1 倍，等效基础字号约 8.8 pt；示意图保持 6.9 英寸画布；PDF 内嵌 TrueType 字体 |
| 表 | 宽表（风险表、设计表、对比表、主干表、字段表）跨栏；窄表（机制、排序、决策）单栏按栏宽缩放 |
| 附录 | A 其余性质与证明；B 实现细节（含各 baseline）；C 一个主干服务所有目标定义；D 合成机理例子 |

## 图表与数据来源

与 V5.2_en 相同（见该目录 README），另：图 1 框架图术语更新；表 VI、VII 与图 9 来自 118/122/125/126/127 号。
