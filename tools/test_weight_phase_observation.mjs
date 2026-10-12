import assert from 'node:assert/strict';
import {createWeightPhaseSource} from '../src/lib/weights.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64};
const fixture=weightFixture(),etag='"phase-observation"',prior=globalThis.fetch;
let requests=0;
globalThis.fetch=async(_url,init)=>{
  requests++;const [start,end]=init.headers.Range.slice(6).split('-').map(Number);
  return new Response(fixture.bytes.slice(start,end+1),{status:206,headers:{ETag:etag,
    'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`,'Content-Length':String(end-start+1)}});
};
const open=async(callback)=>{
  const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:1048576,gpuBytes:1048576});
  budget.bindOwnedDevice(device);
  const source=await createWeightPhaseSource(device,'fixture',{memoryBudget:budget,
    expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag,onAfterWeightsLoaded:callback});
  return {device,budget,source};
};
try{
  const events=[];const x=await open(async observation=>{
    events.push('observed');assert.equal(observation.storageKind,'gpu-buffer');
    assert.equal(observation.sourceETag,etag);assert.equal(observation.tensors.length,2);
    assert.deepEqual(observation.tensors.map(t=>t.name),['camera_embedder.linear.weight','camera_embedder.linear.bias']);
    assert.ok(observation.tensors.every(t=>t.value.destroyed===0&&t.value.size===t.size));
    assert.ok(x.budget.snapshot().gpu.liveBytes>0);
  });
  await x.source.withWeights(x.source.template.cameraEmbedder,()=>{
    assert.deepEqual(events,['observed'],'loaded weights must be observed before computation');
    events.push('consumer');
  });
  assert.deepEqual(events,['observed','consumer']);
  assert.equal(x.budget.snapshot().gpu.liveBytes,0);await x.source.dispose();x.budget.restore();
  const denied=await open(async()=>{throw 0;});let consumed=false;
  await assert.rejects(denied.source.withWeights(denied.source.template.cameraEmbedder,()=>{consumed=true;}),e=>e===0);
  assert.equal(consumed,false);assert.equal(denied.budget.snapshot().cpu.liveBytes,0);
  assert.equal(denied.budget.snapshot().gpu.liveBytes,0);assert.equal(denied.device.destroyed,0);
  assert.ok(denied.device.buffers.every(b=>b.destroyed===1));await denied.source.dispose();denied.budget.restore();
  const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:1048576,gpuBytes:1048576});budget.bindOwnedDevice(device);
  const before=requests;
  await assert.rejects(createWeightPhaseSource(device,'fixture',{memoryBudget:budget,
    expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag,onAfterWeightsLoaded:false}),/weight observation callback/);
  assert.equal(requests,before);budget.restore();
}finally{globalThis.fetch=prior;}
console.log('PASS opt-in exact loaded-weight observation precedes consumer; live scope, falsy denial cleanup, borrowed device and invalid callback');

