"""Build six offline, self-contained HTML slide decks and their source index."""
from pathlib import Path
from html import escape as esc
import json,re,hashlib
from diagrams import DIAGRAMS

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
FIG=OUT/'assets'/'figures'
MANIFEST=json.loads((OUT/'assets'/'figure_manifest.json').read_text())
DECKS=[]

def log(n):return f'DNN_Aggresvation{n}/CHANGELOG.md'
def card(head,body,num='',small='',accent=False):
    return f'<article class="card{" accent" if accent else ""}"><div class="num">{num}</div><h3>{head}</h3><p>{body}</p>'+ (f'<p class="small">{small}</p>' if small else '')+'</article>'
def cards(items):return '<div class="cards">'+''.join(card(*x) for x in items)+'</div>'
def insights(items):
    return '<div class="insights">'+''.join(f'<article class="insight{" orange" if i==len(items)-1 and len(items)>2 else ""}"><h3>{h}</h3><p>{b}</p></article>' for i,(h,b) in enumerate(items))+'</div>'
def table(headers,rows,compact=False):
    return '<div class="tablewrap'+(' compact' if compact else '')+'"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{c}</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def unique_svg(svg,prefix):
    svg=re.sub(r'<\?xml.*?\?>','',svg,flags=re.S);svg=re.sub(r'<!DOCTYPE.*?>','',svg,flags=re.S)
    ids=re.findall(r'\bid="([^"]+)"',svg)
    for ident in sorted(set(ids),key=len,reverse=True):
        new=prefix+'_'+ident
        svg=svg.replace(f'id="{ident}"',f'id="{new}"').replace(f'url(#{ident})',f'url(#{new})').replace(f'href="#{ident}"',f'href="#{new}"')
    return svg
def graphic(name):return '<div class="panel chart">'+(FIG/f'{name}.svg').read_text()+'</div>'
def diag(name):return '<div class="diagram">'+DIAGRAMS[name]()+'</div>'
def split(a,b,equal=False):return f'<div class="split{" equal" if equal else ""}">{a}{b}</div>'
def completed(items,side):
    a='<div class="done-list">'+''.join(f'<div class="done-item"><span class="check">✓</span><div><h3>{h}</h3><p>{b}</p></div></div>' for h,b in items)+'</div>'
    return '<div class="completion">'+a+insights(side)+'</div>'
def deck(no,title,short,range_,subtitle,deliverables):
    d={'no':no,'title':title,'short':short,'range':range_,'subtitle':subtitle,'slides':[]};DECKS.append(d)
    d['slides'].append({'kind':'本批工作','title':title,'cover':True,'body':'','takeaway':'','sources':[], 'notes':[subtitle,'本次主要交付：'+'；'.join(deliverables)+'。'],'deliverables':deliverables})
    return d
def add(d,kind,title,body,takeaway,sources=None,notes=None,figure=None):
    src=list(sources or [])
    if figure:
        src=MANIFEST[figure]['sources']+src
        if not body:body=graphic(figure)
    d['slides'].append({'kind':kind,'title':title,'body':body,'takeaway':takeaway,'sources':list(dict.fromkeys(src)),'notes':notes or [takeaway], 'figure':figure})
def chart(d,kind,title,name,points,takeaway,notes=None,sources=None):
    add(d,kind,title,split(graphic(name),insights(points)),takeaway,sources,notes,figure=name)

def build_content():
    # Content definitions follow in content.py to keep the rendering code reviewable.
    exec((OUT/'src'/'content.py').read_text(),globals())

def render(d):
    css=(OUT/'src'/'theme.css').read_text();js=(OUT/'src'/'viewer.js').read_text();N=len(d['slides']);pages=[]
    for i,s in enumerate(d['slides']):
        foot=f'<footer class="foot"><div class="source">'+(f'<button type="button" data-sources>数据与讲述提示</button>　' if s['sources'] else '')+esc('；'.join(dict.fromkeys(x.split('/')[0].replace('DNN_Aggresvation','实验 ') for x in s['sources'])))+f'</div><div class="folio">{d["no"]:02d} · {i+1:02d} / {N:02d}</div></footer>'
        if s.get('cover'):
            h=f'<section class="slide cover" data-title="{esc(s["title"])}"><div class="kicker"><span>研究进展汇报 · {d["no"]:02d}</span><span class="tag">电力数据推断风险研究</span></div><div class="cover-main"><div><div class="eyebrow">本批完成的工作</div><h1>{d["title"]}</h1><p class="subtitle">{d["subtitle"]}</p></div><div><div class="giant">{d["no"]:02d}</div><div class="range">{d["range"]}</div></div></div><div class="cover-cards">'+''.join(f'<div><b>{j+1:02d}</b>{x}</div>' for j,x in enumerate(s['deliverables']))+'</div>'+foot+'</section>'
        else:
            body=unique_svg(s['body'],f'd{d["no"]}s{i}')
            h=f'<section class="slide" data-title="{esc(s["title"])}"><div class="kicker"><span>{esc(s["kind"])}</span><span class="tag">进展 {d["no"]:02d} · {esc(d["short"])}</span></div><h2>{s["title"]}</h2><div class="content">{body}</div><div class="takeaway">{s["takeaway"]}</div>{foot}</section>'
        pages.append(h)
    metadata=json.dumps([{'title':re.sub('<[^>]+>','',s['title']),'kind':s['kind'],'notes':s['notes'],'sources':s['sources']} for s in d['slides']],ensure_ascii=False).replace('</','<\/')
    title=re.sub('<[^>]+>','',d['title']).replace('\n',' ')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>{d['no']:02d}｜{title}</title><style>{css}</style></head><body><nav class="toolbar" aria-label="演示工具"><div class="name"><a href="index.html">← 六次汇报目录</a>　 /　 {d['no']:02d} {d['short']}</div><div class="tools"><button id="overview-toggle" aria-expanded="false">总览 O</button><button id="notes-toggle" aria-expanded="false">讲稿 N</button><button id="fullscreen">全屏 F</button><button id="print">打印 / PDF</button></div></nav><main class="viewport"><div class="stage">{''.join(pages)}</div></main><nav class="bottom" aria-label="翻页"><button id="prev" aria-label="上一页">← 上一页</button><span id="count"></span><input id="seek" type="range" min="1" max="{N}" value="1" aria-label="选择页码"><button id="next" aria-label="下一页">下一页 →</button><span class="hint">← → 翻页 · O 总览 · N 讲稿 · F 全屏</span></nav><aside class="drawer" id="notes" aria-label="讲述提示与来源"><button class="drawer-close" id="notes-close" aria-label="关闭讲稿">×</button><div id="note-content"></div></aside><div class="overview" id="overview" aria-label="幻灯片总览"><div class="overview-grid"></div></div><div class="toast" id="toast"></div><div id="live" class="sr-only" aria-live="polite"></div><script type="application/json" id="slide-metadata">{metadata}</script><script>{js}</script></body></html>'''

def main():
    build_content()
    all_sources=set();source_lines=['# 六次组会汇报：逐页来源与口径','', '所有图为基于已有实验输出的新绘图；不运行训练。历史阶段与最终配置分别标注。','']
    manifest=[]
    for d in DECKS:
        filename=f'{d["no"]:02d}_{d["short"]}.html';d['filename']=filename
        (OUT/filename).write_text(render(d),encoding='utf-8')
        source_lines += [f'## {d["no"]:02d} {d["short"]}','']
        notes_lines=[f'# 第 {d["no"]} 次：{d["short"]}', '', f'对应：{d["range"]}', '', d['subtitle'], '']
        for i,s in enumerate(d['slides']):
            clean_title=re.sub('<[^>]+>',' ',s['title'])
            notes_lines += [f'## 第 {i+1} 页 · {clean_title}', '', *[x+'\n' for x in s['notes']]]
            if s['takeaway']:notes_lines += ['本页落点：'+s['takeaway'],'']
            source_lines += [f'### 第 {i+1} 页：{s["title"]}','']
            if s.get('figure'):
                source_lines += [f'重绘图：`assets/figures/{s["figure"]}.svg`', '', MANIFEST[s['figure']]['note'],'']
            source_lines += [f'- `{x}`' for x in s['sources']]+['']
            all_sources.update(s['sources'])
        (OUT/'讲稿').mkdir(exist_ok=True)
        (OUT/'讲稿'/f'{d["no"]:02d}_{d["short"]}.md').write_text('\n'.join(notes_lines),encoding='utf-8')
        manifest.append({'number':d['no'],'title':d['short'],'file':filename,'slides':len(d['slides']),'figures':[s['figure'] for s in d['slides'] if s.get('figure')]})
    # Preserve newly drawn diagram SVGs as editable standalone assets as well.
    for name,fn in DIAGRAMS.items():(FIG/f'diagram_{name}.svg').write_text(fn(),encoding='utf-8')
    source_manifest=[]
    for src in sorted(all_sources):
        p=ROOT/src
        if not p.is_file():raise FileNotFoundError(src)
        source_manifest.append({'path':src,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (OUT/'数据与图表来源.md').write_text('\n'.join(source_lines),encoding='utf-8')
    (OUT/'assets'/'source_manifest.json').write_text(json.dumps(source_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'assets'/'deck_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    index_cards=''.join(f'<a class="deck-card" href="{d["filename"]}"><div class="number">{d["no"]:02d}<span>{len(d["slides"])} 页</span></div><h2>{d["short"]}</h2><p>{d["subtitle"]}</p><footer>{d["range"]}<b>进入演示 →</b></footer></a>' for d in DECKS)
    (OUT/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>六次组会汇报｜电力数据推断风险研究</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f3;color:#203544;font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif}}main{{max-width:1340px;margin:auto;padding:58px 34px}}.eyebrow{{font-size:14px;letter-spacing:.18em;color:#147d78;font-weight:700}}h1{{font-size:49px;line-height:1.3;margin:20px 0}}.intro{{font-size:18px;color:#607580;line-height:1.8;max-width:960px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin-top:36px}}.deck-card{{display:flex;flex-direction:column;background:white;border:1px solid #d8e3df;border-radius:15px;padding:27px;text-decoration:none;color:inherit;transition:transform .2s,box-shadow .2s}}.deck-card:hover{{transform:translateY(-4px);box-shadow:0 15px 38px #20354412}}.number{{font-size:38px;font-weight:700;color:#147d78}}.number span{{float:right;font-size:13px;color:#71878a;font-weight:400;padding-top:17px}}h2{{font-size:23px;line-height:1.5;margin:20px 0 10px}}.deck-card p{{font-size:15px;line-height:1.85;color:#607580;flex:1}}.deck-card footer{{border-top:1px solid #e3ebe6;padding-top:17px;margin-top:16px;font-size:12px;color:#71878a;line-height:1.8}}.deck-card footer b{{display:block;font-size:14px;color:#147d78;margin-top:10px}}.help{{margin-top:32px;padding:24px 28px;background:#e5efeb;border-radius:12px;line-height:1.9;font-size:15px}}.help a{{color:#147d78}}@media(max-width:1000px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:650px){{main{{padding:32px 20px}}h1{{font-size:34px}}.grid{{grid-template-columns:1fr}}}}</style></head><body><main><div class="eyebrow">RESEARCH PROGRESS · SIX MEETINGS</div><h1>电力数据推断风险研究<br>六次组会汇报</h1><p class="intro">从论文方向调整到实验补全、估计器改进与投稿准备。每份汇报完整呈现一批工作的背景、实施过程、实验结果和阶段交付。</p><div class="grid">{index_cards}</div><div class="help"><strong>演示：</strong>方向键翻页，F 全屏，O 总览，N 显示讲述提示与数据来源；支持浏览器打印为 PDF。<br>六份 HTML 的图表、样式与脚本均已内嵌，可单独复制后离线演示。<br><a href="README.md">使用说明</a> · <a href="数据与图表来源.md">逐页来源记录</a></div></main></body></html>''',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
