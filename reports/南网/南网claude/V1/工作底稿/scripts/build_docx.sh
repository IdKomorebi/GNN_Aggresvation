#!/bin/bash
# 合并 00/01/02 三份 Markdown 为 Word（在 南网claude/ 目录下运行：bash 工作底稿/scripts/build_docx.sh）
set -e
cd "$(dirname "$0")/../.."
pandoc 00_分类分级体系方案.md 01_师兄论文分类分级内容梳理.md 02_南网文件总结.md \
  --from markdown+tex_math_dollars --to docx \
  --metadata title="推断风险感知的电力数据分类分级体系" \
  --metadata subtitle="南网规则 · 师兄论文方法 · 近期研究（2026-09-29）" \
  --toc --toc-depth=2 --resource-path=. \
  -o 南网分类分级体系方案_完整版.docx
echo done
