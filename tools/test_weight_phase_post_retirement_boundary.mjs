import assert from 'node:assert/strict';
import {createWeightPhaseSource} from '../src/lib/weights.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';

globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
const fixture=weightFixture(),header=new DataView(fixture.bytes.buffer).getUint32(12,true),etag='"post-retirement-boundary"';
const originalFetch=globalThis.fetch;let payloadFetches=0;
globalThis.fetch=async(url,init)=>{
  assert.equal(init.headers['If-Match'],etag);
  const [start,end]=init.headers.Range.slice(6).split('-').map(Number);
  if(start>=header)payloadFetches++;
  return new Response(fixture.bytes.slice(start,end+1),{status:206,headers:{
    ETag:etag,'Content-Range':'bytes '+start+'-'+end+'/'+fixture.bytes.length,'Content-Length':String(end-start+1)}});
};
const make=()=>{
  const device=fakeWeightDevice(),create=device.createBuffer.bind(device),pending=new Set();
  let rejectRetirement=false,postRetirementFences=0;
  device.createBuffer=descriptor=>{
    const buffer=create(descriptor),destroy=buffer.destroy.bind(buffer);
    buffer.destroy=()=>{destroy();pending.add(buffer);};
    return buffer;
  };
  device.queue.onSubmittedWorkDone=async()=>{
    if(pending.size){
      postRetirementFences++;
      if(rejectRetirement)throw Error('post-retirement boundary unresolved');
      pending.clear();
    }
  };
  const budget=createLoaderMemoryBudget({cpuBytes:2*header+4096,gpuBytes:4096});budget.bindOwnedDevice(device);
  return {device,budget,pending,get postRetirementFences(){return postRetirementFences;},setReject(value){rejectRetirement=value;}};
};
let current,source;
try{
  current=make();source=await createWeightPhaseSource(current.device,'fixture',{
    memoryBudget:current.budget,expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag});
  await source.withWeights(source.template.cameraEmbedder,selected=>{
    assert.equal(new Float32Array(selected.weight.data)[0],fixture.values.get('camera_embedder.linear.weight'));
    assert.equal(current.pending.size,0);assert(current.budget.snapshot().gpu.liveBytes>0);
  });
  assert.equal(current.pending.size,0,'a returned GPU phase must observe a queue boundary after its API destruction requests');
  assert.equal(current.postRetirementFences,1);
  assert.equal(source.phases.at(-1).retirementQueueBoundary,'observed-after-api-destroy');
  assert.equal(source.phases.at(-1).status,'completed-retired');
  assert.equal(current.budget.snapshot().gpu.liveBytes,0);
  assert.equal(current.budget.snapshot().gpu.physicalMemoryMeasured,false);
  await source.dispose();current.device.destroy();current.budget.restore();current=null;source=null;

  for(const failWork of [false,true]){
    current=make();source=await createWeightPhaseSource(current.device,'fixture',{
      memoryBudget:current.budget,expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag});
    current.setReject(true);
    await assert.rejects(source.withWeights(source.template.cameraEmbedder,()=>{
      if(failWork)throw 0;
    }),error=>failWork?error instanceof AggregateError&&error.errors[0]===0&&
      error.errors[1].message==='post-retirement boundary unresolved':error.message==='post-retirement boundary unresolved');
    assert.equal(source.phases.at(-1).status,'cleanup-unresolved');
    assert.equal(source.phases.at(-1).retirementQueueBoundary,'pending-after-api-destroy');
    assert.equal(current.pending.size,2);
    assert.equal(current.budget.snapshot().gpu.liveBytes,0,'successful API destructions already released logical leases; do not invent unretired charges');
    assert.equal(current.device.buffers.every(buffer=>buffer.destroyed===1),true);
    const beforeFetches=payloadFetches,beforeBuffers=current.device.buffers.length;
    await assert.rejects(source.withWeights(source.template.cameraEmbedder,()=>{}),/quarantined/);
    assert.equal(payloadFetches,beforeFetches);assert.equal(current.device.buffers.length,beforeBuffers);
    await assert.rejects(source.dispose(),/post-retirement boundary unresolved/);
    assert.equal(current.pending.size,2,'failed retry keeps the unresolved boundary');
    current.setReject(false);await source.dispose();assert.equal(current.pending.size,0);
    assert.equal(current.device.buffers.every(buffer=>buffer.destroyed===1),true,'retry does not recreate/double-destroy successfully retired resources');
    current.device.destroy();current.budget.restore();current=null;source=null;
  }
  console.log('PASS post-API-destruction queue observation before GPU phase return; failed boundary quarantines/no new fetch or allocation/retry recovery; successful logical retirement remains distinct from physical backing');
}finally{
  globalThis.fetch=originalFetch;
  if(current){current.setReject(false);await source?.dispose();current.device.destroy();current.budget.restore();}
}
