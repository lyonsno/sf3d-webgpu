import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {weightFixture} from './fixtures/weight_resource_fixture.mjs';
import {parseFlatWeightHeader} from '../src/lib/flat_tensor_ranges.js';
import {createHash} from 'node:crypto';
const root=path.resolve(new URL('..',import.meta.url).pathname);
const dir=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-dino-observation-contract-'));
const reportPath=path.join(dir,'report.json');
const run=spawnSync(process.execPath,['--import','./tools/wgsl-raw-loader-register.mjs',
  'tools/smoke_resident_dino.mjs','--observe-dino','--report',reportPath],{cwd:root,encoding:'utf8'});
assert.equal(run.status,1);const report=JSON.parse(fs.readFileSync(reportPath));
assert.match(report.error.message,/DINO observation requires.*full-model.*guards/,
  'requested diagnostic must not silently fall through to an unguarded DINO route');
const {createNativeDinoObservation}=await import('./native_dino_observation.mjs');
// Execute the actual browser callback with a tiny device/transport fixture.
// No replacement readback implementation or native-route claim is used.
const harness=fs.readFileSync(path.join(root,'tools/smoke_resident_dino.mjs'),'utf8');
const start=harness.indexOf('const diagnostic=async(kind,observation)=>');
const end=harness.indexOf('    device.pushErrorScope',start);
assert.ok(start>=0&&end>start);
const callbackSource=harness.slice(start+'const diagnostic='.length,end).trim().replace(/;$/,'');
const callbackFixture=({transportFailure=false,drainFailure=false})=>{
  let destroyed=0,released=0,requests=0,allocations=0;
  const device={queue:{submit(){},async onSubmittedWorkDone(){if(drainFailure)throw null;}},
    createBuffer(){allocations++;return {async mapAsync(){},getMappedRange:()=>new ArrayBuffer(4),unmap(){},destroy(){destroyed++;}};},
    createCommandEncoder:()=>({copyBufferToBuffer(){},finish:()=>({})})};
  const budget={reserveCpu(){return {release(){released++;}};}};
  const fetch=async url=>{requests++;if(url==='/dino-observation-tensor'&&transportFailure)throw 0;return {ok:true,json:async()=>({index:0,demand:{rangeCpuBytes:16,workGpuBytes:4}})};};
  const callback=new Function('device','budget','fetch','GPUBufferUsage','GPUMapMode','return ('+callbackSource+')')(
    device,budget,fetch,{COPY_DST:8,MAP_READ:1},{READ:1});
  return {callback,counters:()=>({destroyed,released,allocations})};
};
const observed={tensors:[{name:'weight',shape:[1],size:2,dtype:1,offset:0,value:{size:4}}]};
const combined=callbackFixture({transportFailure:true,drainFailure:true});
await assert.rejects(combined.callback('weights',observed),e=>e instanceof AggregateError&&e.errors[0]===0&&e.errors[1]===null,
  'diagnostic transport rejection must survive an unresolved readback drain');
assert.deepEqual(combined.counters(),{destroyed:0,released:0,allocations:1});
const settled=callbackFixture({transportFailure:true});
await assert.rejects(settled.callback('weights',observed),e=>e===0);
assert.deepEqual(settled.counters(),{destroyed:1,released:1,allocations:1});
const table=new Map([['camera_embedder.linear.weight',{shape:[1],size:2,dtype:1,offset:0}]]);
const phase={phase:'camera',verdict:'admitted',descriptor:{name:'camera',tensors:[{name:'camera_embedder.linear.weight',...table.values().next().value}]}};
const request={kind:'weights',sourceETag:'"source"',tensors:[{name:'camera_embedder.linear.weight',...table.values().next().value,byteLength:4}]};
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const bytes=Buffer.alloc(4);bytes.writeFloatLE(1);
const create=leaf=>createNativeDinoObservation({table,sourceETag:'"source"',outputDir:path.join(dir,leaf),
  expectedWeightHashes:new Map([['camera_embedder.linear.weight',sha(bytes)]])});
const admitted=async(_p,d)=>({verdict:'admitted',demand:d});
const c=create('success');const a=await c.admit(request,phase,admitted);
assert.equal(a.index,0);assert.equal(a.demand.requiredBytes,20);
assert.equal(a.demand.workGpuBytes,4);assert.equal(a.demand.rangeCpuBytes,16);
await c.admitTensor(0,0,admitted);await c.persistTensor(0,0,bytes);assert.equal(c.observations[0].status,'complete');
assert.equal(fs.readFileSync(c.observations[0].outputs[0].path).length,4);
await assert.rejects(c.admit(request,phase,admitted),/kind|checkpoint/);
const checkpoint={kind:'checkpoint',tensors:[{name:'cameraEmbed',shape:[768],byteLength:3072}]};
await c.admit(checkpoint,phase,admitted);
await assert.rejects(c.admit(request,phase,admitted),/incomplete/);
let negative=0;
for(const mutation of [r=>r.sourceETag='"wrong"',r=>r.tensors=[],r=>r.tensors[0].dtype=0,
  r=>r.tensors[0].offset=7,r=>r.tensors[0].shape=[2],r=>r.tensors[0].byteLength=2]){
  const bad=JSON.parse(JSON.stringify(request));mutation(bad);let observed=false;
  await assert.rejects(create('bad-'+negative++).admit(bad,phase,async()=>{observed=true;return {verdict:'admitted'};}));
  assert.equal(observed,false);
}
const refused=create('refused');await assert.rejects(refused.admit(request,phase,async()=>({verdict:'refused'})),/refused/);
await assert.rejects(refused.persistTensor(0,0,bytes),/admitted/);
await assert.rejects(create('wrong-phase').admit(request,{...phase,phase:'dino-block-0'},admitted),/phase/);
await assert.rejects(create('wrong-route').admit(request,{...phase,verdict:'refused'},admitted),/phase/);
assert.throws(()=>c.assertComplete(),/incomplete/);
const fullFixture=weightFixture(),headerBytes=new DataView(fullFixture.bytes.buffer).getUint32(12,true);
const fullTable=parseFlatWeightHeader(fullFixture.bytes.buffer.slice(0,headerBytes),fullFixture.bytes.length).tensors;
const full=createNativeDinoObservation({table:fullTable,sourceETag:'"source"',outputDir:path.join(dir,'full'),
  expectedWeightHashes:new Map([...fullTable].map(([name,t])=>[name,sha(Buffer.alloc(t.size*(t.dtype===1?2:1)))]))});
for(const phaseName of ['camera',...Array.from({length:24},(_,i)=>'dino-block-'+i)]){
  const block=Number(phaseName.slice('dino-block-'.length));
  const selected=[...fullTable].filter(([name])=>phaseName==='camera'?name.startsWith('camera_embedder.linear.'):
    name.startsWith('image_tokenizer.model.encoder.layer.'+block+'.')||
    (block===0&&name.startsWith('image_tokenizer.model.embeddings.'))||
    (block===23&&name.startsWith('image_tokenizer.model.layernorm.'))).map(([name,t])=>({name,...t}));
  const phase={phase:phaseName,verdict:'admitted',descriptor:{name:phaseName,tensors:selected}};
  const weights=selected.map(t=>({...t,byteLength:t.size*(t.dtype===1?2:1)}));
  const row=await full.admit({kind:'weights',sourceETag:'"source"',tensors:weights},phase,admitted);
  for(const [i,t]of weights.entries()){await full.admitTensor(row.index,i,admitted);await full.persistTensor(row.index,i,Buffer.alloc(t.byteLength));}
  if(phaseName==='camera'||block<3||block===23){
    const names=phaseName==='camera'?['cameraEmbed']:block===0?['patchEmbed','block0']:['block'+block];
    const tensors=names.map(name=>({name,shape:name==='cameraEmbed'?[768]:[1297,1024],byteLength:name==='cameraEmbed'?3072:1297*1024*4}));
    const checkpoint=await full.admit({kind:'checkpoint',tensors},phase,admitted);
    for(const [i,t]of tensors.entries()){await full.admitTensor(checkpoint.index,i,admitted);await full.persistTensor(checkpoint.index,i,Buffer.alloc(t.byteLength));}
  }
}
full.assertComplete();assert.equal(full.observations.length,30);
assert.equal(full.observations.filter(o=>o.kind==='weights').flatMap(o=>o.outputs).length,536);
const nonfinite=create('nonfinite');await nonfinite.admit(request,phase,admitted);
await nonfinite.admitTensor(0,0,admitted);await nonfinite.persistTensor(0,0,bytes);
await nonfinite.admit(checkpoint,phase,admitted);await nonfinite.admitTensor(1,0,admitted);
const nan=Buffer.alloc(3072);nan.writeFloatLE(NaN);
await assert.rejects(nonfinite.persistTensor(1,0,nan),/nonfinite/);
assert.equal(fs.statSync(nonfinite.observations[1].failedDeliveries[0].path).size,3072,'raw failing bytes retained');
assert.equal(nonfinite.observations[1].status,'failed');
assert.throws(()=>nonfinite.assertComplete(),/incomplete/);
console.log('PASS opt-in full-route guard enforcement; diagnostic order/source/new-backing/partial/duplicate/refusal contracts, synthetic only');
