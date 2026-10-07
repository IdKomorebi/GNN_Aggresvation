#!/bin/bash
# 合并 00/01/02 为 Word（任意目录运行：bash V2/scripts/build_docx.sh）
set -e
cd "$(dirname "$0")/.."
pandoc 00_分类分级体系方案.md 01_师兄论文分类分级内容梳理.md 02_南网文件总结.md \
  --from markdown+tex_math_dollars --to docx \
  --metadata title="面向南网的电力数据分类分级体系方案（V2）" \
  --metadata subtitle="南网规则为基线 · 师兄论文体系 · 近期研究（2026-10-07）" \
  --toc --toc-depth=2 --resource-path=. \
  -o 南网分类分级体系方案_V2.docx
echo done
