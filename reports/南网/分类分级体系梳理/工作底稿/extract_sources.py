"""Extract the supplied local sources without modifying originals."""
from pathlib import Path, PurePosixPath
import csv
import hashlib
import json
import posixpath
import subprocess
import zipfile
from lxml import etree as ET

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / '分类分级体系梳理'
RAW = BASE / '南网文件_解压'
TEXT = OUT / '工作底稿' / '提取文本'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      's': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}

def words(el):
    return ''.join(el.xpath('.//w:t/text() | .//m:t/text()', namespaces=NS))

def docx_text(path, target):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read('word/document.xml'))
        lines = [f'# {path.name}', '', '定位符按正文 XML 顺序生成；表格按行展开，公式为文本提取，需结合原文核对。', '']
        for idx, el in enumerate(root.find('w:body', NS), 1):
            tag = ET.QName(el).localname
            if tag == 'p':
                s = words(el)
                style = el.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
                media = len(el.xpath('.//w:drawing | .//w:object | .//w:pict', namespaces=NS))
                if s or media:
                    lines.append(f'[P{idx:04d}; style={",".join(style)}] {s}' + (f' [图形/对象:{media}]' if media else ''))
            elif tag == 'tbl':
                lines.append(f'\n[T{idx:04d}]')
                for rowidx, row in enumerate(el.findall('w:tr', NS), 1):
                    cells = [words(cell).replace('\n', ' ') for cell in row.findall('w:tc', NS)]
                    lines.append(f'R{rowidx:03d}\t' + '\t'.join(cells))
                lines.append('')
    target.write_text('\n'.join(lines), encoding='utf-8')

def xlsx_text(path, target):
    with zipfile.ZipFile(path) as z:
        ss = []
        if 'xl/sharedStrings.xml' in z.namelist():
            ss = [''.join(i.xpath('.//s:t/text()', namespaces=NS)) for i in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels = {i.get('Id'): i.get('Target') for i in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        lines = [f'# {path.name}', '']
        for sheet in wb.xpath('.//s:sheet', namespaces=NS):
            name = sheet.get('name')
            rel = rels[sheet.get('{'+NS['r']+'}id')]
            xmlpath = rel.lstrip('/') if rel.startswith('/') else posixpath.normpath('xl/'+rel)
            root = ET.fromstring(z.read(xmlpath))
            lines.append(f'\n## 工作表：{name}\n')
            for row in root.xpath('.//s:sheetData/s:row', namespaces=NS):
                vals = []
                for c in row.findall('s:c', NS):
                    v = c.findtext('s:v', '', namespaces=NS)
                    if c.get('t') == 's': v = ss[int(v)] if v else ''
                    elif c.get('t') == 'inlineStr': v = ''.join(c.xpath('.//s:t/text()', namespaces=NS))
                    if v: vals.append(c.get('r')+'='+v.replace('\n', ' / '))
                if vals: lines.append(' | '.join(vals))
            merges = root.xpath('.//s:mergeCell/@ref', namespaces=NS)
            if merges: lines.append('合并单元格：'+', '.join(merges))
    target.write_text('\n'.join(lines), encoding='utf-8')

def main():
    RAW.mkdir(exist_ok=True)
    TEXT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE / '南网文件.zip') as z:
        for item in z.infolist():
            name = item.filename
            if not item.flag_bits & 0x800:
                try: name = name.encode('cp437').decode('gb18030')
                except (UnicodeEncodeError, UnicodeDecodeError): pass
            pp = PurePosixPath(name)
            if pp.is_absolute() or '..' in pp.parts: raise ValueError(name)
            dest = RAW / name
            if item.is_dir(): dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = z.read(item)
                if dest.exists() and dest.read_bytes() != data: raise ValueError(f'Existing file differs: {dest}')
                dest.write_bytes(data)
    sources = [BASE / '邢之熠_毕业论文.docx'] + sorted(p for p in RAW.rglob('*') if p.is_file())
    manifest = []
    for i, path in enumerate(sources, 1):
        key = f'S{i:02d}'
        target = TEXT / f'{key}_{path.stem}.md'
        if path.suffix == '.docx': docx_text(path, target)
        elif path.suffix == '.xlsx': xlsx_text(path, target)
        elif path.suffix == '.pdf':
            result = subprocess.run(['pdftotext','-layout',str(path),'-'], check=True, capture_output=True)
            pages = result.stdout.decode('utf-8').split('\f')
            target.write_text(f'# {path.name}\n\n'+ '\n'.join(f'\n## PDF物理页 {j}\n\n{p}' for j,p in enumerate(pages,1) if p.strip()), encoding='utf-8')
        else: continue
        manifest.append({'id':key,'source':str(path.relative_to(BASE)), 'bytes':path.stat().st_size, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'extracted':str(target.relative_to(OUT))})
        print(key, path.name, len(target.read_text(encoding='utf-8')))
    (OUT / '工作底稿' / '来源清单.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
