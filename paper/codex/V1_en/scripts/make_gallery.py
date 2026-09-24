"""Create a local gallery of the nine editable SVG figures; no external dependencies."""
from pathlib import Path
from html import escape
D=Path(__file__).resolve().parents[1]
FIGURES=[('fig1_framework','Fig. 1 · 从电力系统字段到组合风险审计'),('fig3_model','Fig. 2 · 共享表示与逐子集读出'),('fig4_combination_risk','Fig. 3 · 单字段评价漏掉的组合风险'),('fig5_profile','Fig. 4 · 字段风险画像与增益矩阵'),('fig2_mechanism','Fig. 5 · 协同、替代与上下文贡献'),('fig6_definition','Fig. 6 · 字段判别、排名与扣留'),('fig7_estimator','Fig. 7 · 估计精度与见证恢复'),('fig8_ablation','Fig. 8 · 关键消融'),('fig9_robustness','Fig. 9 · 鲁棒性与背景预算')]
head='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Codex V1 — 论文图形预览</title><style>body{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;background:#f2f5f7;color:#172b3a}main{max-width:1060px;margin:auto;padding:32px 20px}h1{font-size:26px}p{line-height:1.7;color:#425563}article{background:white;padding:22px;margin:24px 0;border:1px solid #dfe6eb;border-radius:6px}h2{font-size:18px;margin:0 0 12px}img{display:block;width:100%;height:auto}a{color:#0072b2;margin-right:18px}.links{padding-top:12px}nav{margin:20px 0}nav a{line-height:2.2}</style><main><h1>Codex V1 · 论文图形预览</h1><p>基于 paper/claude/V4_en 的表达与展示优化。下方按正文图号排列；PDF 为论文插图，SVG 可继续编辑。示意图与实证图的具体来源见 FIGURE_SOURCES.md。</p><p><a href="main.pdf">阅读完整论文</a><a href="README.md">版本说明</a></p><nav>'''
html=head+''.join(f'<a href="#fig{i}">Fig. {i}</a>' for i in range(1,10))+'</nav>'
for i,(name,title) in enumerate(FIGURES,1):
    html+=f'<article id="fig{i}"><h2>{escape(title)}</h2><img src="figures/{name}.svg" alt="{escape(title)}"><div class="links">'+''.join(f'<a href="figures/{name}.{ext}">{ext.upper()}</a>' for ext in ['pdf','svg','png'])+'</div></article>'
(D/'figure_gallery.html').write_text(html+'</main></html>',encoding='utf-8')
