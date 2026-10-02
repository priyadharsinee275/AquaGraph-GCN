const API = 'http://127.0.0.1:8000';
let network = null, current = null, selected = null, liveTimer = null;
let view = {k:1, x:0, y:0}, dragging = false, dragStart = null, focusEnabled = false;
const $ = s => document.querySelector(s);
const graph = $('#graph'), stage = $('#graphStage'), select = $('#snapshotSelect'), toast = $('#toast');
const NS = 'http://www.w3.org/2000/svg';

async function get(path){const r=await fetch(API+path); if(!r.ok) throw new Error(await r.text()); return r.json();}
function pct(v){return `${(v*100).toFixed(1)}%`}
function showToast(msg){toast.textContent=msg;toast.classList.add('show');clearTimeout(showToast.t);showToast.t=setTimeout(()=>toast.classList.remove('show'),1700)}
function riskClass(v){return v>.66?'high':v>.33?'mid':'low'}
function riskColor(v){return v>.66?'#ff6482':v>.33?'#ffd36b':'#55cdc1'}
function svgEl(tag,attrs={}){const el=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));return el}
function applyView(){graph.style.transform=`translate(${view.x}px,${view.y}px) scale(${view.k})`;graph.style.transformOrigin='50% 50%';}
function resetView(){view={k:1,x:0,y:0};applyView()}

function renderGraph(data){
  const byId=Object.fromEntries(data.predictions.map(p=>[p.node_id,p]));
  graph.innerHTML='';
  const edgeG=svgEl('g');
  network.edges.forEach(e=>{
    const a=network.nodes.find(n=>n.node_id===e.source), b=network.nodes.find(n=>n.node_id===e.target);
    const line=svgEl('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:`edge ${selected&&focusEnabled&&(e.source===selected||e.target===selected)?'active':''}`});
    line.dataset.a=e.source;line.dataset.b=e.target;edgeG.appendChild(line);
  });
  graph.appendChild(edgeG);
  network.nodes.forEach(n=>{
    const p=byId[n.node_id];
    const related=!selected||n.node_id===selected||network.neighbors[selected]?.includes(n.node_id);
    const g=svgEl('g',{class:`node risk-${riskClass(p.risk)} ${selected===n.node_id?'selected ':''}${focusEnabled&&!related?'dim':''}`});g.dataset.node=n.node_id;
    const c=svgEl('circle',{cx:n.x,cy:n.y,r:n.type==='tank'?10:7});g.appendChild(c);
    if(p.risk>.78){g.appendChild(svgEl('circle',{cx:n.x,cy:n.y,r:10,class:'ring'}));}
    if(selected===n.node_id){g.appendChild(svgEl('circle',{cx:n.x,cy:n.y,r:10,class:'halo'}));}
    const t=svgEl('text',{x:n.x,y:n.y+17,class:'node-label'});t.textContent=n.node_id;g.appendChild(t);
    g.addEventListener('click',()=>inspect(n.node_id));
    g.addEventListener('mouseenter',e=>showNodeTip(n,e));g.addEventListener('mouseleave',hideNodeTip);
    g.addEventListener('mousemove',moveNodeTip);
    graph.appendChild(g);
  });
  applyView();
}
function showNodeTip(n,e){const p=current?.predictions?.find(x=>x.node_id===n.node_id); if(!p)return;const tip=$('#nodeTooltip');tip.innerHTML=`<strong>${n.node_id}</strong><span>Leak risk · ${pct(p.risk)}</span><span>Pressure · ${p.pressure_m} m</span>`;tip.classList.add('show');moveNodeTip(e)}
function moveNodeTip(e){const r=stage.getBoundingClientRect();$('#nodeTooltip').style.left=`${e.clientX-r.left+10}px`;$('#nodeTooltip').style.top=`${e.clientY-r.top+10}px`}
function hideNodeTip(){$('#nodeTooltip').classList.remove('show')}
function renderSuspects(data){
  $('#truthBadge').textContent=data.leak_present?`Leak · ${data.ground_truth_node}`:'Healthy snapshot';
  $('#suspectList').innerHTML=data.top_suspects.map((p,i)=>`<div class="suspect" data-node="${p.node_id}"><div class="rank">0${i+1}</div><div><div class="sname">${p.node_id}</div><div class="subscore">model leak probability</div><div class="bar"><div class="fill" style="width:${Math.max(3,p.risk*100)}%"></div></div></div><div class="score">${pct(p.risk)}</div></div>`).join('');
  document.querySelectorAll('.suspect').forEach(el=>el.addEventListener('click',()=>inspect(el.dataset.node)));
}
function renderModelSummary(detail){
  $('#selectedMini').textContent=detail.node_id;$('#selectedRiskLarge').textContent=pct(detail.risk);$('#selectedTitle').textContent=`Node ${detail.node_id}`;
  $('#layer2Output').textContent=detail.layer2_logits.map(v=>v.toFixed(1)).join(' / ');
  $('#activeUnits').textContent=`${detail.layer1.active_units} / 36 active`;
  const vals=detail.layer1.top_channels;
  $('#layer1Bars').innerHTML=Array.from({length:18},(_,i)=>`<i style="height:${10+((vals[i%vals.length]?.activation||.4)*9)}px"></i>`).join('');
  $('#activationBars').innerHTML=Array.from({length:18},(_,i)=>`<i title="Channel ${i+1}"></i>`).join('');
  const bars=[...$('#activationBars i')];vals.forEach((v,i)=>{if(bars[i])bars[i].style.height=`${Math.max(8,Math.min(120,(v.activation+1)*46))}px`;});
  $('#neighborList').innerHTML=detail.neighbors.map(n=>`<span>${n}</span>`).join('')||'<span>none</span>';
}
async function inspect(nodeId){
  if(!current)return; selected=nodeId;
  renderGraph(current); const p=current.predictions.find(x=>x.node_id===nodeId); if(!p)return;
  const detail=await get(`/api/node-detail/${current.snapshot_id}/${nodeId}`);
  $('#nodeAvatar').textContent=nodeId.replace('N','');$('#profileNode').textContent=`Junction ${nodeId}`;$('#inspectorTitle').textContent=`Junction ${nodeId} · signal profile`;
  $('#profileStatus').textContent=p.ground_truth?'Ground truth leak in this snapshot':'No ground-truth leak at this node';$('#profileRisk').textContent=pct(p.risk);
  const items=[['Pressure',`${p.pressure_m} m`,'local head'],['Demand',`${p.demand_lps} L/s`,'consumer demand'],['Flow in',`${p.flow_in_lps} L/s`,'incoming flow'],['Flow out',`${p.flow_out_lps} L/s`,'outgoing flow'],['Pressure gradient',`${p.pressure_gradient}`,'local change'],['Acoustic',`${p.acoustic_index}`,'sensor index'],['Vibration',`${p.vibration_index}`,'sensor index'],['Risk score',pct(p.risk),'GCN output']];
  $('#detailGrid').innerHTML=items.map(([a,b,c])=>`<div class="signal-card"><span>${a}</span><b>${b}</b><small>${c}</small></div>`).join('');
  renderModelSummary(detail); showToast(`Inspecting ${nodeId}`);
  document.querySelector('#inspector').scrollIntoView({behavior:'smooth',block:'start'});
}
async function loadPrediction(id,scroll=false){
  const glow=$('#inferenceGlow');glow.classList.add('show');current=await get(`/api/predict/${id}`);renderGraph(current);renderSuspects(current);
  if(selected&&current.predictions.some(x=>x.node_id===selected)) await inspectWithoutScroll(selected); else await inspectWithoutScroll(current.top_suspects[0].node_id);
  showToast(`Snapshot ${id} scored by 2-layer GCN`);setTimeout(()=>glow.classList.remove('show'),650);if(scroll)document.querySelector('#network').scrollIntoView({behavior:'smooth',block:'start'});
}
async function inspectWithoutScroll(nodeId){if(!current)return;selected=nodeId;renderGraph(current);const p=current.predictions.find(x=>x.node_id===nodeId);if(!p)return;const detail=await get(`/api/node-detail/${current.snapshot_id}/${nodeId}`);$('#selectedMini').textContent=nodeId;$('#selectedRiskLarge').textContent=pct(detail.risk);$('#selectedTitle').textContent=`Node ${nodeId}`;$('#nodeAvatar').textContent=nodeId.replace('N','');$('#profileNode').textContent=`Junction ${nodeId}`;$('#profileStatus').textContent=p.ground_truth?'Ground truth leak in this snapshot':'Model-selected candidate';$('#profileRisk').textContent=pct(detail.risk);const items=[['Pressure',`${p.pressure_m} m`,'local head'],['Demand',`${p.demand_lps} L/s`,'consumer demand'],['Flow in',`${p.flow_in_lps} L/s`,'incoming flow'],['Flow out',`${p.flow_out_lps} L/s`,'outgoing flow'],['Pressure gradient',`${p.pressure_gradient}`,'local change'],['Acoustic',`${p.acoustic_index}`,'sensor index'],['Vibration',`${p.vibration_index}`,'sensor index'],['Risk score',pct(p.risk),'GCN output']];$('#detailGrid').innerHTML=items.map(([a,b,c])=>`<div class="signal-card"><span>${a}</span><b>${b}</b><small>${c}</small></div>`).join('');renderModelSummary(detail);}

async function init(){
  network=await get('/api/network');$('#edgeCount').textContent=`${network.edges.length} pipes`;
  const snaps=await get('/api/snapshots'), m=await get('/api/metrics');$('#top1').textContent=pct(m.test.top1_localization);$('#top3').textContent=pct(m.test.top3_localization);$('#f1').textContent=pct(m.test.node_f1);
  snaps.forEach(s=>{const o=document.createElement('option');o.value=s.snapshot_id;o.textContent=`#${s.snapshot_id} · ${s.leak_present?'LEAK':'HEALTHY'}`;select.appendChild(o)});
  const demo=await get('/api/demo');select.value=demo.snapshot_id;await loadPrediction(demo.snapshot_id);
}

$('#runBtn').addEventListener('click',()=>loadPrediction(Number(select.value),true));$('#heroRun').addEventListener('click',()=>{const i=(select.selectedIndex+7)%select.options.length;select.selectedIndex=i;loadPrediction(Number(select.value),true)});select.addEventListener('change',()=>loadPrediction(Number(select.value)));
$('#inspectBest').addEventListener('click',()=>current&&inspect(current.top_suspects[0].node_id));$('#focusNeighbors').addEventListener('click',()=>{$('#focusNeighbors').classList.toggle('active');focusEnabled=!focusEnabled;if(current)renderGraph(current)});$('#resetView').addEventListener('click',resetView);
$('#liveBtn').addEventListener('click',()=>{if(liveTimer){clearInterval(liveTimer);liveTimer=null;$('#liveBtn').textContent='◉ Live';showToast('Live mode stopped');return}$('#liveBtn').textContent='■ Stop';let i=select.selectedIndex;liveTimer=setInterval(()=>{i=(i+1)%select.options.length;select.selectedIndex=i;loadPrediction(Number(select.value))},3200);showToast('Live mode started')});
$('#tourBtn').addEventListener('click',()=>$('#tourModal').classList.add('show'));$('#closeTour').addEventListener('click',()=>$('#tourModal').classList.remove('show'));$('#tourModal').addEventListener('click',e=>{if(e.target.id==='tourModal')$('#tourModal').classList.remove('show')});
// Pan + zoom on the graph canvas.
stage.addEventListener('wheel',e=>{e.preventDefault();const factor=e.deltaY<0?1.08:.92;view.k=Math.max(.65,Math.min(2.2,view.k*factor));applyView()},{passive:false});
stage.addEventListener('pointerdown',e=>{dragging=true;dragStart={x:e.clientX-view.x,y:e.clientY-view.y};stage.setPointerCapture(e.pointerId)});stage.addEventListener('pointermove',e=>{if(!dragging)return;view.x=e.clientX-dragStart.x;view.y=e.clientY-dragStart.y;applyView()});stage.addEventListener('pointerup',()=>dragging=false);stage.addEventListener('pointercancel',()=>dragging=false);
init().catch(err=>{console.error(err);showToast('Start FastAPI, then refresh this page')});
