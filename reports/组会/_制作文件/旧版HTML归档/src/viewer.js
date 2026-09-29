(()=>{
  const slides=[...document.querySelectorAll('.slide')], stage=document.querySelector('.stage');
  const range=document.getElementById('seek'),counter=document.getElementById('count'),drawer=document.getElementById('notes'),overview=document.getElementById('overview');
  let current=0;
  const metadata=JSON.parse(document.getElementById('slide-metadata').textContent);
  function fit(){const v=document.querySelector('.viewport');const scale=Math.min(v.clientWidth/1600,v.clientHeight/900);stage.style.transform=`translate(-50%,-50%) scale(${scale})`;}
  function notes(){const m=metadata[current];document.getElementById('note-content').innerHTML=`<h2>${current+1}. ${m.title}</h2><h3>讲述提示</h3>${m.notes.map(t=>`<p>${t}</p>`).join('')}<h3>数据与图表来源</h3><ul>${m.sources.map(s=>`<li><a href="../../${encodeURI(s)}" target="_blank" rel="noopener">${s}</a></li>`).join('')}</ul><p class="mini-label">图表已嵌入本 HTML；来源链接指向项目原始记录。复制单个 HTML 后仍能演示。</p>`;}
  function go(n){current=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>{s.classList.toggle('active',i===current);s.setAttribute('aria-hidden',String(i!==current));});range.value=current+1;counter.textContent=`${current+1} / ${slides.length}`;history.replaceState(null,'',`#${current+1}`);notes();document.querySelectorAll('.overview-grid button').forEach((b,i)=>b.classList.toggle('current',i===current));document.getElementById('live').textContent=`第 ${current+1} 页：${metadata[current].title}`;}
  function toggleNotes(){drawer.classList.toggle('open');document.getElementById('notes-toggle').setAttribute('aria-expanded',drawer.classList.contains('open'));}
  function toggleOverview(){overview.classList.toggle('open');document.getElementById('overview-toggle').setAttribute('aria-expanded',overview.classList.contains('open'));}
  document.getElementById('prev').onclick=()=>go(current-1);document.getElementById('next').onclick=()=>go(current+1);
  range.max=slides.length;range.oninput=()=>go(+range.value-1);
  document.getElementById('notes-toggle').onclick=toggleNotes;document.getElementById('notes-close').onclick=toggleNotes;
  document.getElementById('overview-toggle').onclick=toggleOverview;
  document.getElementById('fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch(e){const t=document.getElementById('toast');t.textContent='浏览器未允许全屏，可使用浏览器 F11。';t.style.display='block';setTimeout(()=>t.style.display='none',2800);}};
  document.getElementById('print').onclick=()=>window.print();
  document.querySelectorAll('[data-sources]').forEach(b=>b.onclick=()=>{drawer.classList.add('open');document.getElementById('notes-toggle').setAttribute('aria-expanded','true');notes();});
  document.querySelector('.overview-grid').innerHTML=metadata.map((m,i)=>`<button type="button" data-slide="${i}"><small>${String(i+1).padStart(2,'0')} / ${m.kind}</small>${m.title}</button>`).join('');
  document.querySelectorAll('[data-slide]').forEach(b=>b.onclick=()=>{go(+b.dataset.slide);overview.classList.remove('open');document.getElementById('overview-toggle').setAttribute('aria-expanded','false');});
  document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(['ArrowRight','ArrowDown','PageDown',' '].includes(e.key)){e.preventDefault();go(current+1);}else if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){e.preventDefault();go(current-1);}else if(e.key==='Home'){e.preventDefault();go(0);}else if(e.key==='End'){e.preventDefault();go(slides.length-1);}else if(e.key.toLowerCase()==='n')toggleNotes();else if(e.key.toLowerCase()==='o')toggleOverview();else if(e.key.toLowerCase()==='f')document.getElementById('fullscreen').click();else if(e.key==='Escape'){drawer.classList.remove('open');overview.classList.remove('open');document.getElementById('notes-toggle').setAttribute('aria-expanded','false');document.getElementById('overview-toggle').setAttribute('aria-expanded','false');}});
  let sx=0,sy=0;stage.addEventListener('touchstart',e=>{sx=e.changedTouches[0].clientX;sy=e.changedTouches[0].clientY;},{passive:true});stage.addEventListener('touchend',e=>{const dx=e.changedTouches[0].clientX-sx,dy=e.changedTouches[0].clientY-sy;if(Math.abs(dx)>55&&Math.abs(dx)>Math.abs(dy)*1.5)go(current+(dx<0?1:-1));},{passive:true});
  window.addEventListener('resize',fit);window.addEventListener('hashchange',()=>go((parseInt(location.hash.slice(1))||1)-1));document.addEventListener('fullscreenchange',fit);fit();go((parseInt(location.hash.slice(1))||1)-1);
  window.deck={go,get current(){return current;},count:slides.length};
})();
