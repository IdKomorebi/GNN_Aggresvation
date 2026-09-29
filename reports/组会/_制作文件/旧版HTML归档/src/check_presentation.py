"""Render and inspect every slide in Chromium; no network or experiments."""
from pathlib import Path
import json, math, hashlib
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parents[1]
QA=OUT/'assets'/'previews'; QA.mkdir(exist_ok=True)
MAN=json.loads((OUT/'assets'/'deck_manifest.json').read_text())
FONT=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
issues=[]; reports=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1600,'height':992},device_scale_factor=1)
    errors=[]; requests=[]
    page.on('pageerror',lambda err:errors.append(str(err)))
    page.on('request',lambda req:requests.append(req.url) if req.url.startswith(('http://','https://')) else None)
    for d in MAN:
        page.goto((OUT/d['file']).as_uri(),wait_until='load')
        page.evaluate('document.fonts.ready')
        assert page.evaluate('deck.count')==d['slides']
        thumbs=[]
        for i in range(d['slides']):
            page.evaluate('(i)=>deck.go(i)',i)
            overflow=page.evaluate('''()=>{
              const s=document.querySelector('.slide.active');
              const problems=[];
              for(const e of s.querySelectorAll('.content,.tablewrap,.stack,.insights,.done-list,.card,.formula,.cover-main,.cover-cards,.takeaway')){
                if(e.scrollHeight>e.clientHeight+3 || e.scrollWidth>e.clientWidth+3)
                  problems.push({type:e.className,sw:e.scrollWidth,cw:e.clientWidth,sh:e.scrollHeight,ch:e.clientHeight});
              }
              const c=s.querySelector('.content');
              if(c){const cb=c.getBoundingClientRect();for(const e of c.querySelectorAll('table,.insight,.formula,.done-item')){
                const b=e.getBoundingClientRect();if(b.top<cb.top-3||b.bottom>cb.bottom+3)
                  problems.push({type:'outside-content',class:e.className,top:b.top-cb.top,bottom:b.bottom-cb.bottom});
              }}
              return problems;
            }''')
            if overflow:issues.append({'deck':d['number'],'slide':i+1,'issues':overflow})
            path=QA/f'{d["number"]:02d}_{i+1:02d}.png'
            page.locator('.slide.active').screenshot(path=str(path))
            im=Image.open(path).convert('RGB');im.thumbnail((480,270));thumbs.append(im)
        # Visible interaction checks on every deck.
        page.keyboard.press('Home');assert page.evaluate('deck.current')==0
        page.keyboard.press('ArrowRight');assert page.evaluate('deck.current')==1
        page.keyboard.press('n');assert page.locator('#notes').evaluate('(e)=>e.classList.contains("open")')
        assert page.locator('#note-content a').count()>0
        page.keyboard.press('Escape')
        page.keyboard.press('o');assert page.locator('#overview').evaluate('(e)=>e.classList.contains("open")')
        page.locator('[data-slide="3"]').click();assert page.evaluate('deck.current')==3
        assert not page.locator('#overview').evaluate('(e)=>e.classList.contains("open")')
        page.locator('#seek').fill(str(d['slides']));assert page.evaluate('deck.current')==d['slides']-1
        page.keyboard.press('Tab')
        page.goto((OUT/d['file']).as_uri()+'#4');assert page.evaluate('deck.current')==3
        # All slides must be visible in print, regardless of current page.
        page.emulate_media(media='print')
        assert page.locator('.slide:visible').count()==d['slides']
        page.emulate_media(media='screen')
        sheet=Image.new('RGB',(1480,math.ceil(len(thumbs)/3)*306+16),'#dfe8e5');draw=ImageDraw.Draw(sheet)
        for i,im in enumerate(thumbs):
            x=10+(i%3)*490;y=10+(i//3)*306;sheet.paste(im,(x,y));draw.text((x+8,y+276),f'{d["number"]:02d} / {i+1:02d}',font=FONT,fill='#203544')
        sheet.save(QA/f'{d["number"]:02d}_总览.jpg',quality=88)
        reports.append({'file':d['file'],'slides':d['slides'],'navigation':'passed','notes':'passed','overview':'passed','hash':'passed','print_visibility':'passed'})
    # Responsive directory and offline script/resource checks.
    page.set_viewport_size({'width':390,'height':844})
    page.goto((OUT/'index.html').as_uri());assert page.locator('.deck-card').count()==6
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(QA/'index_mobile.png'),full_page=True)
    page.set_viewport_size({'width':1600,'height':1050})
    page.screenshot(path=str(QA/'index_desktop.png'),full_page=True)
    browser.close()
result={'decks':reports,'total_slides':sum(x['slides'] for x in MAN),'overflow_issues':issues,'javascript_errors':errors,'network_requests':requests}
(OUT/'assets'/'presentation_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
