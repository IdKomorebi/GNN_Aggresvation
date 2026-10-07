// Render the same scene data as editable PowerPoint shapes and text boxes.
// NODE_PATH=/data1/duhaocun/envs/node_pptx_work/node_modules /data1/duhaocun/envs/node_pptx/bin/node src/build_ppt.js
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const root = path.resolve(__dirname,'..');
const data = JSON.parse(fs.readFileSync(path.join(__dirname,'scenes.json'),'utf8'));
const ppt = new pptxgen();
ppt.defineLayout({name:'CSG',width:13.333333,height:7.5});
ppt.layout='CSG';
ppt.author='项目组';
ppt.subject='南网项目分类分级研究体系：规则、组合推断证据、GRPO 等级建议与审批复评';
ppt.title=data.title;
ppt.company='项目研究';
ppt.lang='zh-CN';
ppt.theme={headFontFace:data.font,bodyFontFace:data.font,lang:'zh-CN'};
const I=v=>v/96;
let id=0;
for(const p of data.slides){
  const s=ppt.addSlide();s.background={color:'FFFFFF'};
  for(const o of p.items){
    const name=`s${p.number}-${o.kind}-${++id}`;
    if(o.kind==='rect'||o.kind==='circle'){
      const type=o.kind==='circle'?ppt.ShapeType.ellipse:(o.r?ppt.ShapeType.roundRect:ppt.ShapeType.rect);
      s.addShape(type,{x:I(o.x),y:I(o.y),w:I(o.w),h:I(o.h),radius:I(o.r||0),rectRadius:I(o.r||0),
        fill:{color:o.fill},line:o.stroke?{color:o.stroke,width:o.sw*.75}:{color:o.fill,transparency:100},
        objectName:name});
    }else if(o.kind==='line'){
      s.addShape(ppt.ShapeType.line,{x:I(Math.min(o.x,o.x2)),y:I(Math.min(o.y,o.y2)),
        w:I(Math.abs(o.x2-o.x)),h:I(Math.abs(o.y2-o.y)),
        flipH:o.x2<o.x,flipV:o.y2<o.y,
        line:{color:o.color,width:o.width*.75,...(o.arrow?{beginArrowType:'none',endArrowType:'triangle'}:{}),...(o.dash?{dashType:'dash'}:{})},objectName:name});
    }else{
      s.addText(o.text,{x:I(o.x),y:I(o.y),w:I(o.w),h:I(o.h),
        fontFace:data.font,fontSize:o.size*.75,bold:!!o.bold,color:o.color,
        margin:0,breakLine:false,paraSpaceAfterPt:0,lineSpacing:o.lineHeight*.75,
        align:o.align,valign:'top',wrap:false,isTextBox:true,
        lang:'zh-CN',objectName:name});
    }
  }
  s.addNotes(p.notes);
}
(async()=>{
  const dst=path.join(root,'南网数据分类分级研究体系_白底简约版.pptx');
  await ppt.writeFile({fileName:dst});
  const JSZip=require(require.resolve('jszip',{paths:[require.resolve('pptxgenjs')]}));
  const zip=await JSZip.loadAsync(fs.readFileSync(dst));
  for(const name of Object.keys(zip.files).filter(n=>/^ppt\/theme\/theme\d+\.xml$/.test(n))){
    let text=await zip.file(name).async('string');
    text=text.replace(/<a:ea typeface="[^"]*"\s*\/>/g,`<a:ea typeface="${data.font}"/>`);
    zip.file(name,text);
  }
  fs.writeFileSync(dst,await zip.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
  console.log(dst);
})();
