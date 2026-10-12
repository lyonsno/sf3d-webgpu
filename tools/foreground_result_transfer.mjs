// Exact uncapped JSON byte accounting without materializing the full history.
// Matches JSON data/toJSON semantics, including escaping, sparse arrays and
// nonfinite numbers. The actual encoded body is checked against this count.
function stringBytes(text){
 let n=2;
 for(let i=0;i<text.length;i++){
  const c=text.charCodeAt(i);
  if(c===34||c===92||[8,9,10,12,13].includes(c))n+=2;
  else if(c<32)n+=6;
  else if(c>=0xd800&&c<=0xdbff&&text.charCodeAt(i+1)>=0xdc00&&text.charCodeAt(i+1)<=0xdfff){n+=4;i++;}
  else if(c>=0xd800&&c<=0xdfff)n+=6;
  else n+=c<128?1:c<2048?2:3;
 }
 return n;
}
export function measureJsonUtf8(input){
 const active=new Set();
 const normalize=(v,key)=>{
  if(v!==null&&(typeof v==='object'||typeof v==='bigint')&&typeof v.toJSON==='function')v=v.toJSON(key);
  if(v instanceof Number||v instanceof String||v instanceof Boolean)v=v.valueOf();
  return v;
 };
 const count=(original,key='',array=false)=>{
  const v=normalize(original,key),type=typeof v;
  if(v===null)return 4;
  if(type==='string')return stringBytes(v);
  if(type==='number')return Number.isFinite(v)?String(Object.is(v,-0)?0:v).length:4;
  if(type==='boolean')return v?4:5;
  if(type==='bigint')throw Error('BigInt JSON value');
  if(type==='undefined'||type==='function'||type==='symbol')return array?4:undefined;
  if(active.has(v))throw Error('circular JSON value');
  active.add(v);let total=2,entries=0;
  try{
   if(Array.isArray(v)){for(let i=0;i<v.length;i++){total+=count(v[i],String(i),true)+(i?1:0);}}
   else for(const name of Object.keys(v)){const bytes=count(v[name],name);if(bytes!==undefined){total+=stringBytes(name)+1+bytes+(entries++?1:0);}}
  }finally{active.delete(v);}
  return total;
 };
 const bytes=count(input);if(!Number.isSafeInteger(bytes)||bytes<1)throw Error('complete safe JSON byte count required');
 return bytes;
}

export async function sendForegroundResult({value,runId,fetch,budget}){
 if(value?.runId!==runId||value.status!=='finished'||value.sameDevice!==true||value.sameQueue!==true||
  !Array.isArray(value.frames)||!value.serviceReport)throw Error('complete owned foreground result identity required');
 const bytes=measureJsonUtf8(value);
 const admission=await fetch('/foreground-result-admission',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({runId,bytes})});
 if(!admission.ok)throw Error('foreground result serialization refused: '+await admission.text());
 const {demand}=await admission.json();
 if(demand?.rangeCpuBytes!==4*bytes)throw Error('actual foreground serialization demand changed');
 const lease=budget.reserveCpu(demand.rangeCpuBytes,'complete-foreground-JSON-serialization-transfer');
 try{
  const body=new TextEncoder().encode(JSON.stringify(value));
  if(body.byteLength!==bytes)throw Error('foreground snapshot changed during encoding');
  const hash=await crypto.subtle.digest('SHA-256',body);
  const sha256=Array.from(new Uint8Array(hash),b=>b.toString(16).padStart(2,'0')).join('');
  const response=await fetch('/foreground-result',{method:'POST',headers:{'Content-Type':'application/json',
   'X-Run-Id':runId,'X-Body-Bytes':String(bytes),'X-Body-Sha256':sha256},body});
  if(!response.ok)throw Error('foreground full raw result persistence refused: '+await response.text());
  const ref=await response.json();
  if(ref.schema!=='sf3d.native-foreground-run-reference.v0'||ref.status!=='retained'||ref.runId!==runId||
   ref.bytes!==bytes||ref.sha256!==sha256||typeof ref.path!=='string'||!ref.path.startsWith('/'))
   throw Error('complete source-linked foreground result receipt mismatch');
  return ref;
 }finally{lease.release();}
}
