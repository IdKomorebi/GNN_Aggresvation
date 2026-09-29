from pathlib import Path
from collections import Counter
from lxml import etree as ET
import csv, json, zipfile

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'分类分级体系梳理'
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
path=next((BASE/'南网文件_解压').rglob('*附件1：*.xlsx'))
with zipfile.ZipFile(path) as z:
    ss=[''.join(i.xpath('.//s:t/text()',namespaces=NS)) for i in ET.fromstring(z.read('xl/sharedStrings.xml'))]
    root=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    raw=[]
    for row in root.xpath('.//s:sheetData/s:row',namespaces=NS):
        vals={}
        for c in row.findall('s:c',NS):
            v=c.findtext('s:v','',namespaces=NS)
            if c.get('t')=='s': v=ss[int(v)] if v else ''
            elif c.get('t')=='inlineStr': v=''.join(c.xpath('.//s:t/text()',namespaces=NS))
            vals[''.join(x for x in c.get('r') if x.isalpha())]=v
        raw.append((int(row.get('r')),vals))
    merges=root.xpath('.//s:mergeCell/@ref',namespaces=NS)
headers=next(v for n,v in raw if n==3)
records=[]
for n,v in raw:
    if n>3 and v.get('A','').isdigit() and v.get('D'):
        records.append({'Excel行号':n,**{headers[k]:v.get(k,'') for k in headers}})
target=OUT/'工作底稿'/'开放目录_逐项核对.csv'
with target.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
stats={'文件':str(path.relative_to(BASE)), '工作表':'数据开放目录', '数据行数':len(records),
       '序号范围':[records[0]['序号'],records[-1]['序号']], '合并单元格':merges,
       '字段':list(headers.values()), '按业务域':dict(Counter(r['所属业务域'] for r in records)),
       '按开放策略':dict(Counter(r['开放策略'] for r in records)),
       '空白单元格数':{h:sum(not r[h] for r in records) for h in headers.values()}}
(OUT/'工作底稿'/'开放目录_统计.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
print('\n抽样项目：')
for r in records:
    if any(s in r['数据资源名称'] for s in ['负荷预测','实际负荷','节点电价','机组','客户用电','企业用电','居民','行业用电','潮流','新能源出力']):
        print(json.dumps({k:r[k] for k in ['Excel行号','序号','所属业务域','数据资源名称','更新频率','开放对象','开放策略','开放条件']},ensure_ascii=False))
