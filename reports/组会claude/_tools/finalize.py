"""按正文引用顺序把 figures/ 中的图重命名为 图1.png、图2.png…，更新 report.tex，
并用内容哈希在各编号实验目录中找回原图路径，写出 图片来源.md。"""
import hashlib, re, sys, glob, os
ROOT = '/data1/duhaocun/projects/GNN_Aggresvation'
d = sys.argv[1]
tex_p = os.path.join(d, 'report.tex')
tex = open(tex_p, encoding='utf-8').read()
refs = re.findall(r'\\fig(?:\[[^\]]*\])?\{figures/([^}]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}', tex)
h = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
index = {}
for p in glob.glob(ROOT + '/DNN_Aggresvation*/figures/*.png') + glob.glob(ROOT + '/paper/**/*.png', recursive=True):
    index.setdefault(h(p), p)
rows, seen = [], {}
tmp = {}
for name, cap in refs:
    if name in seen: continue
    k = len(seen) + 1
    new = f'图{k}.png'
    seen[name] = new
    src = os.path.join(d, 'figures', name)
    orig = index.get(h(src))
    note = '原图直接复制' if orig else '重新绘制（matplotlib），见下方说明'
    rows.append((new, os.path.relpath(orig, ROOT) if orig else name, note, re.sub(r'\s+', ' ', cap)))
    tmp[name] = src
for name, new in seen.items():
    os.rename(tmp[name], os.path.join(d, 'figures', '__' + new))
for name, new in seen.items():
    os.rename(os.path.join(d, 'figures', '__' + new), os.path.join(d, 'figures', new))
    tex = tex.replace('figures/' + name + '}', 'figures/' + new + '}')
for new, src, note, cap in rows:
    if '原图' in note:
        tag = '\\protect\\newline{\\footnotesize 原图：\\texttt{%s}}' % src.replace('_', '\\_')
    else:
        tag = '\\protect\\newline{\\footnotesize 重新绘制，数据来源：\\texttt{%s}}' % src.replace('_', '\\_')
    m = re.search(r'(\\fig(?:\[[^\]]*\])?\{figures/' + re.escape(new) + r'\}\{)((?:[^{}]|\{[^{}]*\})*)(\})', tex)
    if m and '原图：' not in m.group(2) and '重新绘制' not in m.group(2):
        tex = tex[:m.start(2)] + m.group(2) + tag + tex[m.end(2):]
open(tex_p, 'w', encoding='utf-8').write(tex)
with open(os.path.join(d, '图片来源.md'), 'w', encoding='utf-8') as f:
    f.write('# 图片来源\n\n| 图 | 来源（项目根目录相对路径） | 处理 | 报告中的图题 |\n| --- | --- | --- | --- |\n')
    for r in rows: f.write('| %s | `%s` | %s | %s |\n' % r)
print('\n'.join(f'{a} <- {b} [{c}]' for a, b, c, _ in rows))
