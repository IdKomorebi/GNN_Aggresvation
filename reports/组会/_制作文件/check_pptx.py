"""Check real PPTX packages, unchanged experimental images and rendered pages."""
from pathlib import Path
from zipfile import ZipFile
import json, hashlib, subprocess, os, re, sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
ROOT=OUT.parents[1]
TMP=Path('/tmp/meeting-pptx-check')
TMP.mkdir(exist_ok=True)
MAN=json.loads((HERE/'ppt_manifest.json').read_text())
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
records=[]
for report in MAN:
    path=OUT/report['file']
    with ZipFile(path) as z:
        slidefiles=[s for s in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',s)]
        assert len(slidefiles)==report['page_count']
        assert len([s for s in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',s)])==report['page_count']
        for sf in slidefiles:
            xml=ET.fromstring(z.read(sf))
            assert xml.find('p:cSld/p:bg/p:bgPr/a:solidFill/a:srgbClr',NS).get('val')=='FFFFFF', sf
        expected={hashlib.sha256((ROOT/i['source']).read_bytes()).hexdigest() for s in report['slides'] for i in s['images']}
        actual={hashlib.sha256(z.read(p)).hexdigest() for p in z.namelist() if p.startswith('ppt/media/')}
        assert expected==actual, (path,expected-actual,actual-expected)
    v2=path.parent.parent/'V2'
    assert v2.is_dir() and not list(v2.iterdir()),v2
    records.append({'file':report['file'],'slides':report['page_count'],
     'white_backgrounds':True,'speaker_notes_present':True,'image_bytes_match_sources':True,'v2_reserved_empty':True})

def render(report):
    no=report['number']; dest=TMP/f'{no:02d}';dest.mkdir(exist_ok=True)
    path=OUT/report['file']
    env=os.environ.copy();env['GSETTINGS_BACKEND']='memory'
    result=subprocess.run(['libreoffice',f'-env:UserInstallation=file:///tmp/meeting-lo-check-{no:02d}',
      '--headless','--convert-to','pdf','--outdir',str(dest),str(path)],env=env,
      check=True,text=True,capture_output=True,timeout=180)
    pdf=dest/path.with_suffix('.pdf').name
    if not pdf.exists():raise RuntimeError(result.stdout+'\n'+result.stderr)
    info=subprocess.run(['pdfinfo',str(pdf)],check=True,capture_output=True,text=True).stdout
    pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
    assert pages==report['page_count'],(no,pages)
    subprocess.run(['pdftoppm','-scale-to','1600','-png',str(pdf),str(dest/'slide')],check=True,capture_output=True,timeout=180)
    subprocess.run(['pdftotext','-bbox-layout',str(pdf),str(dest/'text_bounds.html')],check=True,capture_output=True)
    images=sorted(dest.glob('slide-*.png'),key=lambda p:int(p.stem.split('-')[-1]))
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
    for start in range(0,len(images),8):
        group=images[start:start+8];cw=640;ch=388;sheet=Image.new('RGB',(cw*2,ch*((len(group)+1)//2)),'#e8e8e8')
        draw=ImageDraw.Draw(sheet)
        for j,p in enumerate(group):
            x=(j%2)*cw;y=(j//2)*ch
            im=Image.open(p).convert('RGB');im.thumbnail((632,356));sheet.paste(im,(x+4,y+26))
            draw.text((x+8,y+2),f'{no:02d} — 第{start+j+1}页',font=font,fill='black')
        sheet.save(TMP/f'contact_{no:02d}_{start//8+1}.jpg',quality=92)
    print(f'{no:02d}: rendered {pages} pages',flush=True)
    return {'number':no,'rendered_pages':pages,'pdf':str(pdf)}

if __name__=='__main__':
    selected={int(x) for x in sys.argv[1:]}
    subset=[r for r in MAN if not selected or r['number'] in selected]
    with ThreadPoolExecutor(max_workers=2) as pool:
        rendered=list(pool.map(render,subset))
    if selected:
        previous=json.loads((HERE/'VALIDATION.json').read_text())
        rendered+= [r for r in previous['rendering'] if r['number'] not in selected]
        rendered.sort(key=lambda r:r['number'])
    result={'reports':records,'rendering':rendered,
      'total_pages':sum(r['slides'] for r in records),
      'original_image_placements':sum(not i.get('derived',False) for r in MAN for s in r['slides'] for i in s['images']),
      'targeted_matplotlib_redraws':sum(i.get('derived',False) for r in MAN for s in r['slides'] for i in s['images']),
      'editable_schematics':2,'visual_review':'pending'}
    (HERE/'VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
