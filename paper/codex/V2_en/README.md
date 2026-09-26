# 通用英文稿 V2

这是 Codex 根据 `paper/claude/V6_ieee/main.pdf` 及其源码制作的总体优化版。它吸收 `paper/codex/V1_en` 的矢量绘图风格，但实验与结果以更新的 Claude V6 为准。V2 不绑定投稿期刊，作为后续 IJCIP、JISA 等定向版本的共同母版。

入口：[论文 PDF](main.pdf) · [主文件](main.tex) · [图形预览](figure_gallery.html) · [图与实验来源](FIGURE_SOURCES.md) · [期刊研究记录](../JOURNAL_ADAPTATION_RESEARCH.md)。

## 本版完成的优化

1. **集中故事线。** 以背景信息如何改变一个字段的推断风险为主线，将单字段风险、背景放大、支撑背景组织为一个风险画像。结果依次解释受控物理机制、真实数据中的放大现象、评分和估计效果、预算与发布应用。
2. **分配概念层级。** 核心定义和两条主要解释性结果保留正文；要求对照、进一步性质、表示学习诊断和附加指标移至附录。减少引言、方法和实验之间重复提出问题和重复论证计算必要性的段落。
3. **固定估计器的理由。** 全文采用重建特征＋随机特征、三种子预测平均的配置。每集合 48–64 ms（均值约 58 ms）在当前规模已可接受，它对完整风险画像的误差更小。重建单分支约 21 ms，保留为更大数据/查询量下的低成本选择，不增加实验。
4. **按结果组织方法比较。** 区分整体风险恢复、关键字段召回和计算成本。保留 TabPFN v2 的真实优势与本文效率优势；不把不同指标压缩成“完全相同”的结论。
5. **重画展示。** 概览使用电网对象、字段矩阵、增量分布与画像；模型图画出训练阶段、两个特征分支、读出和验证；新增出力—报价段—价格原理图。密集统计图改为上下或 2×2 面板；主表突出核心指标，剩余数据保留附录。
6. **改为完整论文表达。** 减少“尝试后发现”的研究日志口吻，摘要和结论围绕问题、方法、证据和用途展开。标题使用 power-system data，覆盖电价、预测、出力、潮流、负荷等数据。

本版沿用既有理论框架、数据角色、实验协议和结果，没有新增 120/121 的内容，也没有运行新的训练或科学实验。图形重绘读取已有实验输出。原始目录和旧版本保留。

## 结构

Introduction → Related Work → 风险定义与解释 → 估计及验证 → Experimental Setup → Results → 发布应用 → Discussion / Conclusion → 附录 → References。

各章节存为独立 `sec_*.tex`、`res_*.tex`；表格保留可编辑 LaTeX。具体来源和本版验证结果见 `source_manifest.json`、`FIGURE_SOURCES.md`、`VALIDATION.json`。图文件名沿用语义标识，不等于最终 PDF 中的自动编号。

## 构建与重绘

已有图片随稿交付，编译论文不需要原始数据或 Python：

```bash
cd paper/codex/V2_en
bash build.sh
```

需要 Tectonic（首次使用可能下载 TeX 依赖）；也可在含 `elsarticle-num.bst` 的 TeX Live/Overleaf 环境用 XeLaTeX、BibTeX 编译。

仅当需要重绘时，在完整项目中运行：

```bash
PYTHON=/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python bash redraw.sh
```

重绘需要 NumPy、pandas、Matplotlib、PyTorch 及原有实验输出，路径由脚本向上查找项目根目录。该命令重算展示统计并输出图片，不启动训练。`scripts/make_schematics.py` 手工构造矢量示意图，未使用生成式位图。

## 版本关系

```text
paper/claude/V6_ieee（研究内容来源）
    └── paper/codex/V2_en（通用总体优化）
          ├── paper/codex/IJCIP/V1（基础设施信息保护）
          └── paper/codex/JISA/V1（推断泄露与安全审计）
```

未来通用改进继续使用 `V3_en` 等版本；期刊稿在各自目录内使用 `V2`、`V3`，避免混淆通用研究修订与期刊适配。
