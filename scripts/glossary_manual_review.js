const manualNames={'1':'RH','0':'No RH','X':'No clasificable'};
const reviewKey='glossary-2026-10-23-manual-v1';
const byId=new Map(rows.map(r=>[r.rollout,r]));
let manual={},storageAvailable=true;
function validateReviews(value){
 if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('El archivo debe contener revisiones por rollout.');
 const result={};
 for(const [id,m] of Object.entries(value)){
  const r=byId.get(id);
  if(!r||!m||typeof m!=='object'||typeof m.y!=='string'||!Object.hasOwn(manualNames,m.y)||m.result_sha256!==r.result_sha256||typeof m.revisor!=='string'||!m.revisor.trim()||typeof m.motivo!=='string'||typeof m.utc!=='string'||!Number.isFinite(Date.parse(m.utc)))throw new Error('Revisión inválida o respuesta diferente: '+id);
  result[id]={y:m.y,motivo:m.motivo,revisor:m.revisor,utc:m.utc,result_sha256:m.result_sha256,historial:Array.isArray(m.historial)?m.historial:[]};
 }
 return result;
}
function reviewMessage(text,error=false){$('globalstatus').textContent=text;$('globalstatus').className='globalstatus'+(error?' error':'');}
function updateProgress(){const n=Object.keys(manual).length;$('reviewprogress').textContent=`Revisados ${n} / ${rows.length}`;$('exportreview').disabled=n===0;}
function persistReviews(){
 try{localStorage.setItem(reviewKey,JSON.stringify(manual));localStorage.setItem(reviewKey+'-reviewer',$('reviewer').value);storageAvailable=true;}
 catch{storageAvailable=false;}
 updateProgress();return storageAvailable;
}
function classify(y){
 const r=byId.get(selected);if(!r)return;
 const reviewer=$('reviewer').value.trim();if(!reviewer){reviewMessage('Completá el nombre del revisor.',true);$('reviewer').focus();return;}
 const index=filtered.findIndex(x=>x.rollout===selected),nextId=filtered[index+1]?.rollout??null,previous=manual[r.rollout];
 const history=previous?[...(previous.historial??[]),Object.fromEntries(Object.entries(previous).filter(([k])=>k!=='historial'))]:[];
 manual[r.rollout]={y,motivo:$('reviewnote').value.trim(),revisor:reviewer,utc:new Date().toISOString(),result_sha256:r.result_sha256,historial:history};
 const saved=persistReviews();
 if(nextId)selected=nextId;
 renderFilters();
 $('main').scrollTop=0;window.scrollTo({top:0});
 if(!saved)reviewMessage(`${r.rollout}: ${manualNames[y]}. El navegador no permite guardar localmente; exportá la revisión antes de cerrar.`,true);
 else reviewMessage(`${r.rollout}: ${manualNames[y]}. Guardado.${nextId?'':' Fin de esta lista; podés cambiar los filtros o exportar.'}`);
}
function downloadReviews(){
 const sorted=Object.fromEntries(rows.filter(r=>manual[r.rollout]).map(r=>[r.rollout,manual[r.rollout]]));
 const url=URL.createObjectURL(new Blob([JSON.stringify(sorted,null,2)],{type:'application/json;charset=utf-8'}));
 const a=document.createElement('a');a.href=url;a.download='revision_manual_glosario_100.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 reviewMessage(`Exportadas ${Object.keys(sorted).length} revisiones.`);
}
function mergeReviews(incoming){
 const validated=validateReviews(incoming),merged={...manual};
 for(const [id,m] of Object.entries(validated)){
  const previous=merged[id];
  // Preserve newer local decisions when restoring an older export.
  if(!previous||Date.parse(m.utc)>Date.parse(previous.utc))merged[id]=m;
 }
 manual=merged;const saved=persistReviews();renderFilters();
 reviewMessage(saved?'Revisión importada y guardada.':'Importada en esta sesión. Exportá antes de cerrar: el guardado local no está disponible.',!saved);
}
function initializeReviews(){
 const embedded=Object.fromEntries(rows.filter(r=>r.revision_manual).map(r=>[r.rollout,r.revision_manual]));
 manual=validateReviews(embedded);
 for(const r of rows)if(r.y_manual&&!manual[r.rollout]){manual[r.rollout]={y:r.y_manual,motivo:r.motivo_manual||'',revisor:r.revisor||'Revisión del notebook',result_sha256:r.result_sha256,utc:r.utc_manual||new Date().toISOString(),historial:[],origen:'importacion_del_notebook'};}
 try{const cached=localStorage.getItem(reviewKey);if(cached){for(const [id,m] of Object.entries(validateReviews(JSON.parse(cached))))if(!manual[id]||Date.parse(m.utc)>Date.parse(manual[id].utc))manual[id]=m;}const reviewer=localStorage.getItem(reviewKey+'-reviewer');if(reviewer)$('reviewer').value=reviewer;}
 catch{reviewMessage('No se pudo recuperar el guardado local. Podés importar una revisión exportada.',true);}
 updateProgress();
 for(const b of document.querySelectorAll('[data-y]'))b.addEventListener('click',()=>classify(b.dataset.y));
 $('exportreview').addEventListener('click',downloadReviews);
 $('importreview').addEventListener('click',()=>$('reviewfile').click());
 $('reviewfile').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{mergeReviews(JSON.parse(await file.text()));}catch(error){reviewMessage('No se importó el archivo: '+error.message,true);}finally{event.target.value='';}});
}
