const $=id=>document.getElementById(id);
let songs=[],state={jobs:[],tokenizers:[]},uploading=false,submitting=false,refreshing=false;
const fields=['steps','save_every','window_frames','learning_rate_head','learning_rate_lora','learning_rate_io','accumulation','regularizer_songs','regularizer_steps','regularizer_batch','regularizer_weight','generated_flow_percent','seed'];
function error(s=''){$('error').textContent=s;$('error').hidden=!s;}
async function api(path,options={}){const r=await fetch(path,options);const d=await r.json();if(!r.ok)throw new Error(d.error||'Request failed');return d;}
function post(path,data){return api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});}
function node(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function buttons(){$('train-button').disabled=uploading||submitting||!songs.length||!state.semantic_audio_ready;$('train-button').firstChild.textContent=state.busy?'Add training to queue ':'Start training ';}
function renderSongs(){
    $('songs').replaceChildren(...songs.map((s,i)=>{const row=node('div',undefined,'label-row');row.style.margin='14px 0';row.append(node('span',s.name));const b=node('button','Remove','text-button');b.type='button';b.disabled=uploading;b.onclick=async()=>{try{songs=await post('/api/training/songs',{songs:songs.filter((_,j)=>i!==j)});renderSongs();buttons();}catch(e){error(e.message);}};row.append(b);return row;}));
}
async function upload(files){
    if(uploading)return;uploading=true;error();renderSongs();buttons();
    try{for(let i=0;i<files.length;i++){$('upload-status').textContent=`Uploading ${i+1}/${files.length}: ${files[i].name}`;const r=await api('/api/upload',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:files[i]});songs=await post('/api/training/songs',{songs:[...songs,{id:r.id,name:files[i].name}]});renderSongs();}}
    catch(e){error(e.message);}finally{uploading=false;$('upload-status').textContent=`${songs.length} songs in collection`;renderSongs();buttons();}
}
for(const type of ['dragenter','dragover'])$('songs-drop').addEventListener(type,e=>{e.preventDefault();$('songs-drop').classList.add('dragging');});
for(const type of ['dragleave','drop'])$('songs-drop').addEventListener(type,e=>{e.preventDefault();$('songs-drop').classList.remove('dragging');});
$('songs-drop').addEventListener('drop',e=>upload([...e.dataTransfer.files]));$('songs-file').onchange=()=>{upload([...$('songs-file').files]);$('songs-file').value='';};
$('songs-drop').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('songs-file').click();}};
$('training-form').onsubmit=async e=>{e.preventDefault();submitting=true;buttons();error();const parameters=Object.fromEntries(fields.map(k=>[k,Number($(k).value)]));parameters.train_head=$('train_head').checked;parameters.train_nar=$('train_nar').checked;try{await post('/api/jobs',{kind:'train',name:$('name').value,tokenizer_choice:$('base').value,parameters});await refresh();}catch(e){error(e.message);}finally{submitting=false;buttons();}};
function lossChart(rows,key,label,color){
    const points=rows.filter(r=>Number.isFinite(r[key]));const box=node('figure');box.style.margin='16px 0';box.append(node('figcaption',label));
    if(!points.length){box.append(node('small','No measurements yet.'));return box;}
    const smooth=key==='flow_loss',windowSize=20;let sum=0;
    const averages=points.map((p,i)=>{sum+=p[key];if(i>=windowSize)sum-=points[i-windowSize][key];return sum/Math.min(i+1,windowSize);});
    const ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 500 180');svg.setAttribute('role','img');svg.setAttribute('tabindex','0');svg.setAttribute('aria-label',`${label}. Hover or use left/right arrow keys for exact values.`);svg.style.cssText='width:100%;display:block;cursor:crosshair';
    function shape(tag,attrs,text){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e;}
    let lo=Infinity,hi=-Infinity;for(const p of points){lo=Math.min(lo,p[key]);hi=Math.max(hi,p[key]);}
    const pad=(hi-lo)*.05||Math.max(Math.abs(hi)*.05,.001);lo=Math.max(0,lo-pad);hi+=pad;
    const first=points[0].step,last=points.at(-1).step;
    const x=step=>68+(step-first)/Math.max(1,last-first)*416,y=value=>144-(value-lo)/(hi-lo)*128;
    for(let i=0;i<5;i++){const value=lo+(hi-lo)*i/4,yy=y(value);shape('line',{x1:68,x2:484,y1:yy,y2:yy,stroke:'currentColor',opacity:.12});shape('text',{x:60,y:yy+4,'text-anchor':'end',fill:'currentColor','font-size':11},value.toFixed(4));}
    shape('text',{x:68,y:170,fill:'currentColor','font-size':11},`Step ${first}`);
    if(last!==first)shape('text',{x:484,y:170,'text-anchor':'end',fill:'currentColor','font-size':11},`Step ${last}`);
    shape('polyline',{points:points.map(p=>`${x(p.step)},${y(p[key])}`).join(' '),fill:'none',stroke:color,'stroke-width':1.5,opacity:smooth?.35:1,'data-series':'raw'});
    if(smooth)shape('polyline',{points:points.map((p,i)=>`${x(p.step)},${y(averages[i])}`).join(' '),fill:'none',stroke:color,'stroke-width':2.5,'data-series':'moving-average'});
    const cursor=shape('line',{x1:0,x2:0,y1:16,y2:144,stroke:'currentColor',opacity:.4,'stroke-dasharray':'3 3'});
    const dot=shape('circle',{cx:0,cy:0,r:3,fill:color});
    const readout=node('div',undefined,'loss-readout');readout.style.cssText='font-size:12px;font-variant-numeric:tabular-nums;min-height:2.8em';
    let selected=points.length-1;
    function show(i){selected=i;const p=points[i];cursor.setAttribute('x1',x(p.step));cursor.setAttribute('x2',x(p.step));dot.setAttribute('cx',x(p.step));dot.setAttribute('cy',y(p[key]));readout.textContent=`Step ${p.step} · loss ${p[key].toFixed(6)}`+(smooth?` · mean (${Math.min(i+1,windowSize)} updates): ${averages[i].toFixed(6)}`:'');}
    svg.onpointermove=e=>{const point=svg.createSVGPoint();point.x=e.clientX;point.y=e.clientY;const px=point.matrixTransform(svg.getScreenCTM().inverse()).x;let nearest=0;for(let i=1;i<points.length;i++)if(Math.abs(x(points[i].step)-px)<Math.abs(x(points[nearest].step)-px))nearest=i;show(nearest);};
    svg.onpointerleave=()=>{if(document.activeElement!==svg)show(points.length-1);};
    svg.onkeydown=e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();show(e.key==='Home'?0:e.key==='End'?points.length-1:Math.max(0,Math.min(points.length-1,selected+(e.key==='ArrowLeft'?-1:1))));}};
    show(selected);
    box.append(svg,readout,node('small',(smooth?'Faint: raw · bold: trailing 20-update mean (shorter initially). ':'')+'Independent vertical scale. Hover for values; chart refresh pauses while hovered/focused.'));return box;
}
async function trainingProgress(card,j){
    try{const p=await api('/api/training/'+j.id);if(!card.isConnected)return;
        const charts=node('div');for(const [key,label,color] of [['flow_loss','Flow loss','#74b3ff'],['regularizer_loss','Weighted tokenizer retention loss','#c692ff'],['validation_loss','Held-out flow loss','#69cfa6']])charts.append(lossChart(p.metrics||[],key,label,color));card.append(charts);
        if(p.resumable){const resume=node('button',`Resume from step ${p.step}`,'text-button');resume.onclick=async()=>{resume.disabled=true;try{await post('/api/jobs',{kind:'train',resume_job:j.id});await refresh();}catch(e){error(e.message);resume.disabled=false;}};card.append(resume);}
        else if(!['queued','running','cancelling'].includes(j.status))card.append(node('small','No unfinished resumable checkpoint. Older weights can still be used for a fresh run.'));
    }catch(e){card.append(node('small','Loss history unavailable: '+e.message));}
}
async function refresh(){
    if(refreshing)return;refreshing=true;
    try{
        state=await api('/api/state');$('status').textContent=state.busy?'● Processing local jobs':'● Ready';
        const options=state.tokenizers||[];const signature=JSON.stringify(options);
        if($('base').dataset.options!==signature){const selected=$('base').value;$('base').replaceChildren(...options.map(o=>new Option(o.name,o.id)));if(options.some(o=>o.id===selected))$('base').value=selected;$('base').dataset.options=signature;
            $('checkpoints').replaceChildren(...options.map(o=>{const p=node('p',o.name);return p;}));}
        // Keep the inspected point stable instead of replacing it every two seconds.
        if($('runs').querySelector('svg:hover,svg:focus')){buttons();return;}
        const jobs=state.jobs.filter(j=>j.kind==='train');const active=jobs.filter(j=>['queued','running','cancelling'].includes(j.status));const past=jobs.filter(j=>!active.includes(j)).slice(0,$('show-older').checked?undefined:5);
        const expanded=new Set([...$('runs').querySelectorAll('details[open]')].map(d=>d.closest('.job').dataset.id));
        $('runs').replaceChildren(...[...active,...past].map(j=>{const card=node('article',undefined,'job');card.dataset.id=j.id;card.append(node('h3',j.title),node('p',j.status+(j.queue_position?` · queue #${j.queue_position}`:'')),node('p',j.progress,'progress-text'));
            const details=node('details');details.open=expanded.has(j.id);details.append(node('summary','Saved training settings'));const pre=node('pre',JSON.stringify(j.settings,null,2));pre.style.cssText='white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px';details.append(pre);card.append(details);
            if(['queued','running'].includes(j.status)){const stop=node('button',j.status==='queued'?'Remove from queue':'Stop and save','text-button');stop.onclick=async()=>{try{await post('/api/cancel',{id:j.id});await refresh();}catch(e){error(e.message);}};card.append(stop);}return card;}));buttons();
        await Promise.all([...$('runs').children].map(card=>trainingProgress(card,jobs.find(j=>j.id===card.dataset.id))));
    }catch(e){$('status').textContent='Studio disconnected';$('train-button').disabled=true;}finally{refreshing=false;}
}
$('show-older').onchange=refresh;
async function init(){try{songs=await api('/api/training/songs');renderSongs();await refresh();}catch(e){error(e.message);}}
init();setInterval(refresh,2000);
