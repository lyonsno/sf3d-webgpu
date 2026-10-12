import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {createNativeDinoObservation} from './native_dino_observation.mjs';
import http from 'node:http';
const dir=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-dino-stream-contract-'));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const bytes=Buffer.alloc(4);bytes.writeFloatLE(1);
const info={name:'camera_embedder.linear.weight',shape:[1],size:2,dtype:1,offset:0};
const phase={phase:'camera',verdict:'admitted',descriptor:{name:'camera',tensors:[info]}};
const request={kind:'weights',sourceETag:'"source"',tensors:[{...info,byteLength:4}]};
const create=leaf=>createNativeDinoObservation({table:new Map([[info.name,info]]),sourceETag:'"source"',
  expectedWeightHashes:new Map([[info.name,sha(bytes)]]),outputDir:path.join(dir,leaf)});
const admit=async(_p,d)=>({verdict:'admitted',demand:d});
if(!process.argv.includes('--reuse-only')){
 if(!process.argv.includes('--hash-only')){
  const c=create('partial');await c.admit(request,phase,admit);
  if(c.admitTensor)await c.admitTensor(0,0,admit);
  await assert.rejects(async()=>await c.persistTensor(0,0,bytes.subarray(0,1)),/partial/);
  assert.ok(c.observations[0].failedDeliveries?.length===1,'received partial bytes must have failed-evidence custody');
  const failed=c.observations[0].failedDeliveries[0];
  assert.equal(failed.claimedObservationIndex,0);assert.equal(failed.claimedTensorIndex,0);
  assert.equal(failed.expectedBytes,4);assert.equal(failed.bytes,1);
  assert.equal(failed.sha256,sha(bytes.subarray(0,1)));assert.deepEqual(fs.readFileSync(failed.path),bytes.subarray(0,1));
  assert.equal(c.observations[0].outputs.length,0);assert.equal(c.observations[0].status,'failed');
  assert.throws(()=>c.assertComplete(),/incomplete/);
 }
  const wrong=create('wrong-hash');await wrong.admit(request,phase,admit);
  if(wrong.admitTensor)await wrong.admitTensor(0,0,admit);
  const different=Buffer.alloc(4);different.writeFloatLE(2);
  await assert.rejects(async()=>await wrong.persistTensor(0,0,different),/canonical.*hash|hash.*canonical/);
  assert.equal(wrong.observations[0].outputs.length,0);assert.deepEqual(fs.readFileSync(wrong.observations[0].failedDeliveries[0].path),different);
  let receiving;
  const streamed=createNativeDinoObservation({table:new Map([[info.name,info]]),sourceETag:'"source"',
    expectedWeightHashes:new Map([[info.name,sha(bytes)]]),outputDir:path.join(dir,'streamed'),
    onEvent(kind,payload){if(kind==='dino-observation-tensor-receiving')receiving=payload.output.path;}});
  await streamed.admit(request,phase,admit);await streamed.admitTensor(0,0,admit);
  await streamed.persistTensor(0,0,(async function*(){yield bytes.subarray(0,1);
    assert.equal(fs.statSync(receiving).size,1,'actual bytes reach disk before the remaining body arrives');
    yield bytes.subarray(1,3);yield bytes.subarray(3);})());
  const complete=streamed.observations[0].outputs[0];assert.deepEqual(fs.readFileSync(complete.path),bytes);
  await assert.rejects(streamed.persistTensor(0,0,Buffer.alloc(8)),/duplicate/);
  assert.deepEqual(fs.readFileSync(complete.path),bytes,'successful bytes not overwritten by duplicate body');
  assert.equal(streamed.observations[0].failedDeliveries[0].bytes,8);
  assert.equal(streamed.observations[0].outputs.length,1);
  const freshDenied=create('fresh-denied');await freshDenied.admit(request,phase,admit);
  await assert.rejects(freshDenied.admitTensor(0,0,async()=>({verdict:'refused'})),/refused/);
  assert.equal(freshDenied.observations[0].tensorAdmissions[0].admission.verdict,'refused');
  assert.equal(freshDenied.observations[0].status,'refused');
}
const source=fs.readFileSync(new URL('./smoke_resident_dino.mjs',import.meta.url),'utf8');
const start=source.indexOf('const diagnostic=async(kind,observation)=>'),end=source.indexOf('    device.pushErrorScope',start);
assert.ok(start>=0&&end>start);const body=source.slice(start+'const diagnostic='.length,end).trim().replace(/;$/,'');
let allocations=0,destroys=0,release=0,gates=0,persisted=0;const ranges=[];
const device={queue:{submit(){},async onSubmittedWorkDone(){}},
  createBuffer({size}){allocations++;return {size,async mapAsync(){},getMappedRange(offset,length){ranges.push([offset,length]);return new ArrayBuffer(length??size);},unmap(){},destroy(){destroys++;}};},
  createCommandEncoder(){return {copyBufferToBuffer(_a,_b,_c,_d,size){assert.ok([4,8].includes(size));},finish(){return {};}};}};
const budget={reserveCpu(){return {release(){release++;}};}};
const fetch=async(url)=>{
  if(url==='/dino-observation-tensor-admission')gates++;
  if(url==='/dino-observation-tensor')persisted++;
  return {ok:true,json:async()=>({index:0,demand:{rangeCpuBytes:32,workGpuBytes:8}})};
};
const callback=new Function('device','budget','fetch','GPUBufferUsage','GPUMapMode','return ('+body+')')(
  device,budget,fetch,{COPY_DST:8,MAP_READ:1},{READ:1});
await callback('weights',{sourceETag:'"source"',tensors:[
  {name:'first',shape:[1],value:{size:4}},{name:'second',shape:[2],value:{size:8}}]});
assert.equal(allocations,1,'one real full-size staging must survive all tensor copies in a phase');
assert.equal(destroys,1);assert.equal(release,1);assert.equal(gates,2,'fresh gate before every copy, not retired physical credit');
assert.equal(persisted,2);assert.deepEqual(ranges,[[0,4],[0,8]],'complete actual tensor bytes, no oversized/default readback');
const transport=create('actual-node-http');await transport.admit(request,phase,admit);await transport.admitTensor(0,0,admit);
let receivedHeaders,transportError;
const server=http.createServer(async(req,res)=>{
  try{receivedHeaders=req.headers;await transport.persistTensor(0,0,req);res.writeHead(200).end('retained');}
  catch(error){transportError=error;res.writeHead(500).end(String(error));}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
try{
  const result=await new Promise((resolve,reject)=>{
    const req=http.request({host:'127.0.0.1',port:server.address().port,path:'/tensor',method:'POST',headers:{'Content-Length':4}},res=>{
      const chunks=[];res.on('data',c=>chunks.push(c));res.on('end',()=>resolve({status:res.statusCode,body:Buffer.concat(chunks).toString()}));
    });req.on('error',reject);req.end(bytes);
  });
  assert.equal(transportError,undefined);assert.deepEqual(result,{status:200,body:'retained'});
  assert.equal(receivedHeaders['content-length'],'4');assert.deepEqual(fs.readFileSync(transport.observations[0].outputs[0].path),bytes);
  fs.writeFileSync(path.join(dir,'actual-http-receipt.json'),JSON.stringify({route:'actual-Node-HTTP-IncomingMessage-stream',
    node:process.version,requestBytes:4,sourceSHA256:sha(bytes),response:result,outputs:transport.observations[0].outputs,
    claim:'actual local HTTP stream with deterministic synthetic bytes; not browser/GPU/model conformance'},null,2));
}finally{await new Promise(resolve=>server.close(resolve));}
console.log('PASS actual readback reuse/per-copy admission/whole bytes; partial and mismatched canonical raw evidence retained');
