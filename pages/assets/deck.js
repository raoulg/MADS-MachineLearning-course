// Runs a lesson deck: scaling the stage, moving between slides and build steps,
// speaker notes, crumbs, and the links back to the course pages.
(function(){
  var slides=[].slice.call(document.querySelectorAll('.deck>section'));
  // slide numbers: "MADS-ML · N" in a footer, the bare number elsewhere
  slides.forEach(function(s,k){
    [].forEach.call(s.querySelectorAll('.n'),function(el){
      el.textContent=(el.closest('.foot')&&!el.dataset.plain)?'MADS-ML · '+(k+1):String(k+1);
    });
  });
  // breadcrumbs from the sections that start a part
  var nav=document.getElementById('crumbs');
  slides.forEach(function(s,k){
    if(!s.dataset.crumb)return;
    var a=document.createElement('a');a.href='#'+(k+1);a.textContent=s.dataset.crumb;a.dataset.from=k;nav.appendChild(a);
  });
})();
(function(){
  document.documentElement.classList.add('js');
  var slides=[].slice.call(document.querySelectorAll('.deck>section'));
  var stage=document.getElementById('stage'), notes=document.getElementById('notes');
  var i=0;
  function fit(){stage.style.setProperty('--scale',Math.min(innerWidth/1920,innerHeight/1080));}
  function builds(s){return [].slice.call(s.querySelectorAll('[data-build-in]'));}
  function next(){var left=builds(slides[i]).filter(function(b){return !b.classList.contains('built');});if(left.length){left[0].classList.add('built');}else if(i<slides.length-1){show(i+1);}}
  function show(n,back){
    i=Math.max(0,Math.min(slides.length-1,n));
    builds(slides[i]).forEach(function(b){b.classList.toggle('built',!!back);});
    slides.forEach(function(s,k){s.classList.toggle('active',k===i);});
    history.replaceState(null,'','#'+(i+1));
    var a=slides[i].querySelector('aside');
    notes.querySelector('.count').textContent='SLIDE '+(i+1)+' / '+slides.length;
    notes.querySelector('.text').textContent=a?a.textContent:'';
  }
  addEventListener('keydown',function(e){
    if(e.metaKey||e.ctrlKey||e.altKey)return;
    var k=e.key;
    if(['ArrowRight','ArrowDown','PageDown',' ','Enter'].indexOf(k)>=0){e.preventDefault();next();}
    else if(['ArrowLeft','ArrowUp','PageUp','Backspace'].indexOf(k)>=0){e.preventDefault();show(i-1,true);}
    else if(k==='Home'){show(0);}
    else if(k==='End'){show(slides.length-1);}
    else if(k==='n'||k==='N'){document.body.classList.toggle('show-notes');}
    else if(k==='f'||k==='F'){
      if(!document.fullscreenElement){document.documentElement.requestFullscreen&&document.documentElement.requestFullscreen();}
      else{document.exitFullscreen&&document.exitFullscreen();}
    }
  });
  addEventListener('click',function(e){
    if(e.target.closest('#notes,#crumbs'))return;
    if(e.clientX<innerWidth/3){show(i-1,true);}else{next();}
  });
  var x0=null;
  addEventListener('touchstart',function(e){x0=e.touches[0].clientX;},{passive:true});
  addEventListener('touchend',function(e){
    if(x0===null)return;
    var dx=e.changedTouches[0].clientX-x0; x0=null;
    if(Math.abs(dx)>50){if(dx<0){next();}else{show(i-1,true);}}
  });
  addEventListener('hashchange',function(){
    var h=parseInt(location.hash.slice(1),10);
    if(isFinite(h)&&h-1!==i)show(h-1);
  });
  addEventListener('resize',fit);
  fit();
  var h=parseInt(location.hash.slice(1),10);
  show(isFinite(h)?h-1:0);
  setTimeout(function(){document.getElementById('hint').classList.add('gone');},4000);
})();
(function(){
  var slides=[].slice.call(document.querySelectorAll('.deck>section'));
  var links=[].slice.call(document.querySelectorAll('#crumbs a'));
  var starts=links.map(function(a){return +a.dataset.from;});
  function mark(){
    var i=slides.findIndex(function(s){return s.classList.contains('active');});
    var here=null;
    links.forEach(function(a,k){var end=k+1<starts.length?starts[k+1]:slides.length;if(starts[k]<=i&&i<end)here=a;});
    links.forEach(function(a){a.classList.toggle('here',a===here);});
  }
  new MutationObserver(mark).observe(document.querySelector('.deck'),{attributes:true,subtree:true,attributeFilter:['class']});
  mark();
})();
(function(){
  // links from a deck back to the course pages; the lesson comes from <body data-lesson>
  var lesson=document.body.dataset.lesson;
  var nav=document.createElement('nav');nav.id='site-links';
  var links=[['../index.html','\u2190 MADS-ML start']];
  if(lesson)links.push(['../leerdoelen/les'+lesson+'.html','learning goals'],['../huiswerk/your-study.html#lesson-'+lesson,'your research']);
  links.forEach(function(l){var a=document.createElement('a');a.href=l[0];a.textContent=l[1];nav.appendChild(a);});
  nav.addEventListener('click',function(e){e.stopPropagation();});
  nav.addEventListener('keydown',function(e){e.stopPropagation();});
  document.body.appendChild(nav);
})();
