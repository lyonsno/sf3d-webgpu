import assert from 'node:assert/strict';
import {runResidentDino} from '../src/lib/resident_dino.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64};
const fixture=weightFixture({fp16Names:new Set(['camera_embedder.linear.weight','camera_embedder.linear.bias'])}),etag='"dino-observation"',prior=globalThis.fetch;
let fetches=0;
globalThis.fetch=async(_url,init)=>{
  fetches++;const [start,end]=init.headers.Range.slice(6).split('-').map(Number);
  return new Response(fixture.bytes.slice(start,end+1),{status:206,headers:{ETag:etag,
    'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`,'Content-Length':String(end-start+1)}});
};
function make(){
  const device=fakeWeightDevice(),dispatches=[];device.createBindGroup=descriptor=>descriptor;device.createCommandEncoder=()=>({
    beginComputePass(){return {setPipeline(){},setBindGroup(){},dispatchWorkgroups(...args){dispatches.push(args);},end(){}};},
    copyBufferToBuffer(){},finish(){return {};}});
  const budget=createLoaderMemoryBudget({cpuBytes:268435456,gpuBytes:1073741824,totalBytes:1342177280});
  return {device,budget,dispatches};
}
const prepare=async x=>{await x.budget.requestOwnedDevice({requestDevice:async()=>x.device});return {
  device:x.device,memoryBudget:x.budget,weightsUrl:'fixture',expectedWeightBytes:fixture.bytes.length,
  expectedSourceETag:etag,imageChw:new Float32Array(3*512*512),onBeforePhase:async()=>{},withResult:async()=>{}};};
try{
  const x=make(),base=await prepare(x),seen=[],weights=[];
  const result=await runResidentDino({...base,streamFp16Uploads:false,
    onAfterWeightsLoaded:async o=>{weights.push(o.tensors.map(t=>t.name));
      assert.ok(o.tensors.every(t=>t.value.size===t.size*(t.dtype===1?2:1)));},
    onDiagnosticCheckpoint:async o=>{
      assert.ok(o.tensors.every(t=>t.buffer.destroyed===0));
      assert.ok(o.tensors.every(t=>t.buffer.size===t.shape.reduce((n,v)=>n*v,1)*4));
      seen.push(...o.tensors.map(t=>t.name));
    },
    withResult:async r=>{
      assert.deepEqual(seen,['cameraEmbed','patchEmbed','block0','block1','block2','block23'],
        'existing checkpoint copies must be consumed before retirement');
      assert.equal(r.tokensBuf.destroyed,0);
    },
  });
  assert.equal(result.blocks,24);assert.equal(result.loadingReport.gpuFp16UploadMode,'direct-mapped-v1');
  assert.equal(weights.length,25);assert.equal(weights.flat().length,536);
  assert.equal(x.device.destroyed,0);assert.equal(x.budget.snapshot().cpu.liveBytes,0);
  assert.equal(x.budget.snapshot().gpu.liveBytes,0);x.budget.restore();
  const defaultRun=make(),defaultConfig=await prepare(defaultRun);
  const phases=[];const unchanged=await runResidentDino({...defaultConfig,onBeforePhase:async p=>phases.push(p.name)});
  assert.equal(unchanged.loadingReport.gpuFp16UploadMode,'queue-write-complete-source-prefix-v1');
  assert.deepEqual(phases,['camera',...Array.from({length:24},(_,i)=>'dino-block-'+i),'dino-output']);
  assert.deepEqual(defaultRun.dispatches,x.dispatches,'opt-in observation does not change model dispatches');
  assert.ok(x.dispatches.length>24);
  assert.deepEqual(defaultRun.device.buffers.map(b=>b.size),x.device.buffers.map(b=>b.size),
    'observation hooks allocate no duplicate model or snapshot buffers');
  assert.equal(defaultRun.budget.snapshot().cpu.liveBytes,0);assert.equal(defaultRun.budget.snapshot().gpu.liveBytes,0);defaultRun.budget.restore();
  for(const key of ['streamFp16Uploads','onAfterWeightsLoaded','onDiagnosticCheckpoint']){
    const y=make(),cfg=await prepare(y),before=fetches;
    await assert.rejects(runResidentDino({...cfg,[key]:0}),/exact.*boolean|observation callback|checkpoint callback/);
    assert.equal(fetches,before);assert.equal(y.device.buffers.length,0);y.budget.restore();
  }
  const y=make(),cfg=await prepare(y);let consumer=false;
  await assert.rejects(runResidentDino({...cfg,onDiagnosticCheckpoint:async()=>{throw 0;},
    withResult:async()=>{consumer=true;}}),e=>e===0);
  assert.equal(consumer,false);assert.equal(y.device.destroyed,0);
  assert.equal(y.budget.snapshot().cpu.liveBytes,0);assert.equal(y.budget.snapshot().gpu.liveBytes,0);y.budget.restore();
}finally{globalThis.fetch=prior;}
console.log('PASS complete24-block diagnostic checkpoint/weight custody, explicit mapped mode, invalid options and falsy observer denial; synthetic device only');
