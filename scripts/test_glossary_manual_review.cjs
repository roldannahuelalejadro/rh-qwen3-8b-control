const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../analysis/glossary/explorador_glosario_100.html'),'utf8');
const actualPayload=JSON.parse(html.split('<script id="data" type="application/json">')[1].split('</script>')[0]);
const unreviewed=JSON.parse(JSON.stringify(actualPayload));
for(const r of unreviewed.rows){delete r.revision_manual;r.y_manual='';}
const payload=JSON.stringify(unreviewed);
const source=html.split('<script>')[1].split('</script>')[0];
new vm.Script(source);
const storage=new Map();
let failStorage=false;
function element(){return {textContent:'',className:'',value:'',hidden:false,disabled:false,dataset:{},children:[],listeners:{},
 append(...e){this.children.push(...e);},replaceChildren(){this.children=[];},setAttribute(k,v){this[k]=v;},
 addEventListener(k,f){this.listeners[k]=f;},focus(){},click(){this.listeners.click?.();}};}
function session(input=payload){
 const els=new Map();
 const get=id=>{if(!els.has(id))els.set(id,element());return els.get(id);};
 get('data').textContent=input;get('label').value='X';get('finish').value='all';get('reviewfilter').value='all';get('reviewer').value='Nahuel';
 const buttons=['1','0','X'].map(y=>Object.assign(element(),{dataset:{y}}));
 const context=vm.createContext({console,Map,Date,Blob,JSON,Object,URL,setTimeout,
  document:{getElementById:get,createElement:element,querySelectorAll:()=>buttons},
  window:{scrollTo(){}},localStorage:{getItem:k=>storage.get(k)??null,setItem(k,v){if(failStorage)throw Error('storage disabled');storage.set(k,v);}},navigator:{}});
 vm.runInContext(source,context);return {context,get,buttons,run:code=>vm.runInContext(code,context)};
}
let s=session();
assert.equal(s.run('filtered.length'),33);
assert.equal(s.run('selected'),'r0003');
const second=s.run('filtered[1].rollout'),third=s.run('filtered[2].rollout');
s.get('reviewnote').value='Revision manual de prueba';
s.buttons[0].click();
assert.equal(s.run("manual.r0003.y"),'1');
assert.equal(s.run("manual.r0003.motivo"),'Revision manual de prueba');
assert.equal(s.run('selected'),second);
assert.equal(s.run('rows[2].y_propuesto'),'X');
assert.match(s.get('reviewprogress').textContent,/1 \/ 100/);
s=session();
assert.equal(s.run('manual.r0003.y'),'1');
s.run("select('r0003');classify('0')");
assert.equal(s.run('manual.r0003.historial[0].y'),'1');
assert.equal(s.run('manual.r0003.y'),'0');
s.get('reviewfilter').value='pending';s.run('renderFilters()');
assert.equal(s.run('filtered.length'),32);
s.run("classify('X')");
assert.equal(s.run(`manual['${second}'].y`),'X');
assert.equal(s.run('selected'),third);
assert.equal(s.run('filtered.length'),31);
assert.throws(()=>s.run("validateReviews({r0003:{y:'1',result_sha256:'wrong'}})"));
assert.equal(s.run('manual.r0003.y'),'0');
const old=s.run('JSON.parse(JSON.stringify(manual.r0003))');old.utc='2020-01-01T00:00:00Z';old.y='1';
s.context.old=old;s.run('mergeReviews({r0003:old})');assert.equal(s.run('manual.r0003.y'),'0');
failStorage=true;s.run("classify('1')");
assert.match(s.get('globalstatus').textContent,/no permite guardar/);
assert.equal(s.run('Object.keys(manual).length'),3);
s.get('reviewfilter').value='all';s.run('renderFilters(); select(filtered.at(-1).rollout); classify("X")');
assert.match(s.get('globalstatus').textContent,/no permite guardar/);
failStorage=false;s.run('persistReviews();classify("0")');
assert.match(s.get('globalstatus').textContent,/Fin de esta lista/);
storage.clear();
if(actualPayload.rows.every(r=>r.revision_manual)){
 s=session(JSON.stringify(actualPayload));
 assert.equal(s.run('Object.keys(manual).length'),100);
 assert.equal(s.run('manual.r0001.historial.length'),actualPayload.rows[0].revision_manual.historial.length);
 assert.equal(s.run('manual.r0003.y'),'X');
}
console.log('OK: botones 1/0/X, avance, persistencia, historial, filtros, validacion de hashes, importacion y fallo de almacenamiento.');
