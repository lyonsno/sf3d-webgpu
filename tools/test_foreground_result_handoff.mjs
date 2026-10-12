import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
const source=fs.readFileSync(new URL('./smoke_resident_dino.mjs',import.meta.url),'utf8');
const full={status:'finished',runId:'owned-run',sameDevice:true,sameQueue:true,before:{frameCount:1},after:{frameCount:3},
 frames:[{runId:'owned-run',unknown:'retained-α'},{runId:'owned-run',value:[1,null,true]}],
 serviceReport:{schema:'observed-service',receipts:[{unknown:'all'}],services:[{id:'all'}]},future:{all:[1,2,3]}};
if(source.includes('foregroundRun:foregroundFinish}')){
 const literal=source.match(/return ({backend,dino,twoStream,[^\n]+foregroundRun:foregroundFinish});/)[1];
 const names=['backend','dino','twoStream','postProcessor','fullArtifact','decodedGlbImages','upstreamRetirement','inputConsumed','budget','foregroundFinish'];
 const hot=new Function(...names,'return '+literal)({},{},{},{},{},[],{},true,{snapshot:()=>({})},full);
 assert.equal(hot.foregroundRun?.frames,undefined,'complete native result must not return full foreground history over CDP while host remains live');
}
const {measureJsonUtf8,sendForegroundResult}=await import('./foreground_result_transfer.mjs');
const {createForegroundResultReceiver}=await import('./native_foreground_result.mjs');
const sha=b=>createHash('sha256').update(b).digest('hex'),body=Buffer.from(JSON.stringify(full));
assert.equal(measureJsonUtf8(full),body.length);
for(const v of [null,1,true,'é😀',{'future':undefined,a:[undefined,NaN,Infinity],nested:{x:'\ud800'}}])
 assert.equal(measureJsonUtf8(v),Buffer.byteLength(JSON.stringify(v)));
const circular={};circular.self=circular;assert.throws(()=>measureJsonUtf8(circular),/circular/);
for(const fault of [null,'partial','hash','run','duplicate','refusal']){
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'foreground-result-contract-'));
 const receiver=createForegroundResultReceiver({filename:path.join(dir,'full-foreground.json'),runId:'owned-run'});
 const admit=async(_phase,demand)=>({verdict:fault==='refusal'?'refused':'admitted',demand});
 if(fault==='refusal'){await assert.rejects(receiver.admit({runId:'owned-run',bytes:body.length},admit),/refused/);assert.equal(receiver.state.admission.verdict,'refused');continue;}
 await receiver.admit({runId:'owned-run',bytes:body.length},admit);
 const actual=fault==='partial'?body.subarray(0,body.length-1):body;
 const headers={runId:fault==='run'?'other':'owned-run',sha256:fault==='hash'?'0'.repeat(64):sha(actual),bytes:body.length};
 if(['partial','hash','run'].includes(fault)){
  await assert.rejects(receiver.receive(actual,headers),/partial|hash|identity/);
  assert.equal(receiver.state.deliveries.length,1);const f=receiver.state.deliveries[0];assert.equal(f.status,'failed');assert.equal(f.bytes,actual.length);assert.equal(f.sha256,sha(actual));assert.deepEqual(fs.readFileSync(f.path),actual);
  await assert.rejects(receiver.restore({exitObserved:true,ownedPid:123},123),/complete/);continue;
 }
 const ref=await receiver.receive((async function*(){yield body.subarray(0,3);yield body.subarray(3);})(),headers);
 assert.equal(ref.bytes,body.length);assert.equal(ref.sha256,sha(body));assert.equal(ref.runId,'owned-run');
 await assert.rejects(receiver.restore({exitObserved:false,ownedPid:123},123),/exact.*exit/);
 await assert.rejects(receiver.restore({exitObserved:true,ownedPid:999},123),/exact.*exit/);
 const restored=await receiver.restore({exitObserved:true,ownedPid:123},123);assert.deepEqual(restored,full,'unknown additive fields/all receipts retained unchanged');
 if(fault==='duplicate'){await assert.rejects(receiver.receive(body,headers),/duplicate/);assert.deepEqual(fs.readFileSync(ref.path),body);assert.equal(receiver.state.deliveries.length,2);}
}
let released=0,calls=[],posted;
// Exercise an actual positive storage prefix followed by a filesystem failure.
// Both received and successfully persisted bytes remain independently exact.
{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'foreground-result-storage-'));
 const receiver=createForegroundResultReceiver({filename:path.join(dir,'full.json'),runId:'owned-run'});
 await receiver.admit({runId:'owned-run',bytes:body.length},async()=>({verdict:'admitted'}));
 const realWrite=fs.writeSync;let writes=0;
 try{
  fs.writeSync=(fd,b,offset,length)=>{if(writes++)throw Error('observed simulated storage fault');return realWrite(fd,b,offset,1);};
  await assert.rejects(receiver.receive(body,{runId:'owned-run',bytes:body.length,sha256:sha(body)}),/storage fault/);
 }finally{fs.writeSync=realWrite;}
 const prefix=receiver.state.deliveries[0];assert.equal(prefix.receivedBytes,body.length);assert.equal(prefix.bytes,1);
 assert.equal(prefix.sha256,sha(body.subarray(0,1)));assert.deepEqual(fs.readFileSync(prefix.path),body.subarray(0,1));
 assert.equal(prefix.status,'failed');assert.equal(receiver.state.reference,undefined);
}
const response=v=>({ok:true,json:async()=>v,text:async()=>''});
const fetched=async(url,options)=>{
 calls.push(url);if(url==='/foreground-result-admission')return response({demand:{rangeCpuBytes:4*body.length}});
 posted=options.body;assert.equal(options.headers['X-Run-Id'],'owned-run');assert.equal(options.headers['X-Body-Sha256'],sha(body));
 return response({schema:'sf3d.native-foreground-run-reference.v0',status:'retained',runId:'owned-run',path:'/owned/full.json',bytes:body.length,sha256:sha(body)});
};
const sent=await sendForegroundResult({value:full,runId:'owned-run',fetch:fetched,budget:{reserveCpu(n){assert.equal(n,4*body.length);return{release(){released++;}};}}});
assert.deepEqual(Buffer.from(posted),body);assert.deepEqual(calls,['/foreground-result-admission','/foreground-result']);assert.equal(released,1);assert.equal(sent.schema,'sf3d.native-foreground-run-reference.v0');assert.equal(sent.frames,undefined);
await assert.rejects(sendForegroundResult({value:full,runId:'other',fetch:fetched,budget:{}}),/identity/);
console.log('PASS full foreground result file handoff, prospective serialization lease, complete bytes/hash/source, partial/duplicate/refusal retention, restore only after exact exit');
