import assert from 'node:assert/strict';
import * as loader from '../src/lib/weights.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';

assert.equal(typeof loader.createWeightPhaseSource,'function','phase source must reuse the real weight builders without eager payload allocation');
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
const fixture=weightFixture(),header=new DataView(fixture.bytes.buffer).getUint32(12,true),etag='"phase-fixture-1"';
const original=globalThis.fetch,calls=[];
let wrongSource=false,onPayload=null;
globalThis.fetch=async(url,init)=>{
  assert.equal(init.headers['If-Match'],etag);
  const match=init.headers.Range.match(/^bytes=(\d+)-(\d+)$/);assert.ok(match);
  const start=Number(match[1]),end=Number(match[2]);calls.push({start,end});
  if(start>=header)onPayload?.();
  return new Response(fixture.bytes.slice(start,end+1),{status:206,headers:{ETag:wrongSource?'"changed"':etag,
    'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`,'Content-Length':String(end-start+1)}});
};
const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:2*header+4096,gpuBytes:4096});
budget.bindOwnedDevice(device);
let source;
try{
  source=await loader.createWeightPhaseSource(device,'fixture.bin',{memoryBudget:budget,expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag});
  assert.equal(calls.length,2,'construction reads only prefix and table');
  assert.equal(device.buffers.length,0,'metadata does not allocate model payload');
  assert.equal(source.template.imageTokenizer.blocks.length,24);
  assert.equal(source.template.backbone.mainBlocks.length,4);
  let completed=0;
  for(let index=0;index<24;index++){
    await source.withWeights({block:source.template.imageTokenizer.blocks[index]},async({block})=>{
      assert.equal(new Float32Array(block.norm1.weight.data)[0],fixture.values.get(`image_tokenizer.model.encoder.layer.${index}.norm1.weight`));
      assert.ok(budget.snapshot().gpu.liveBytes>0);
      await assert.rejects(()=>source.withWeights({camera:source.template.cameraEmbedder},async()=>{}),/active phase/);
      completed++;
    });
    assert.equal(budget.snapshot().gpu.liveBytes,0,'each finished block retires before the next loads');
    assert.equal(budget.snapshot().cpu.liveBytes,0);
  }
  assert.equal(completed,24,'residency changes must not truncate the encoder');
  const prior=calls.length;
  await assert.rejects(()=>source.withWeights({unknown:{tensorName:'not-authenticated'}},async()=>{}),/phase selection/);
  assert.equal(calls.length,prior,'foreign selection refuses before payload access');
  await assert.rejects(()=>source.withWeights({},async()=>{}),/nonempty/);
  const selection={camera:source.template.cameraEmbedder};
  onPayload=()=>{selection.camera=source.template.imageTokenizer.layernorm;};
  await source.withWeights(selection,async({camera})=>{
    assert.equal(new Float32Array(camera.weight.data)[0],fixture.values.get('camera_embedder.linear.weight'),'selection pins before asynchronous payload reads');
  });onPayload=null;

  let resolveFence,entered;
  const enteredPromise=new Promise(resolve=>entered=resolve);
  device.queue.onSubmittedWorkDone=async()=>{if(resolveFence===null){entered();await new Promise(resolve=>resolveFence=resolve);}};
  // Upload fence completes first; hold the phase-terminal fence explicitly.
  let uploadDone=false;
  const held=source.withWeights({camera:source.template.cameraEmbedder},async()=>{uploadDone=true;resolveFence=null;});
  await enteredPromise;
  assert.equal(uploadDone,true);assert.ok(budget.snapshot().gpu.liveBytes>0,'submitted weights stay charged until the terminal fence');
  resolveFence();await held;assert.equal(budget.snapshot().gpu.liveBytes,0);
  device.queue.onSubmittedWorkDone=async()=>{};
  await assert.rejects(()=>source.withWeights({camera:source.template.cameraEmbedder},async()=>{throw Error('compute failed');}),/compute failed/);
  assert.equal(budget.snapshot().gpu.liveBytes,0);
  wrongSource=true;
  await assert.rejects(()=>source.withWeights({camera:source.template.cameraEmbedder},async()=>{}),/source identity mismatch/);
  assert.equal(budget.snapshot().gpu.liveBytes,0);assert.equal(budget.snapshot().cpu.liveBytes,0);
  wrongSource=false;
  const bufferCount=device.buffers.length;
  device.queue.onSubmittedWorkDone=async()=>{throw Error('queue drain failed');};
  await assert.rejects(()=>source.withWeights({camera:source.template.cameraEmbedder},async()=>{}),error=>
    error instanceof AggregateError&&error.errors.length===2&&error.errors.every(e=>e.message==='queue drain failed'));
  assert.ok(budget.snapshot().gpu.liveBytes>0,'an unresolved upload/drain keeps its charge until recovery');
  await assert.rejects(()=>source.withWeights({camera:source.template.cameraEmbedder},async()=>{}),/quarantined/);
  assert.equal(device.buffers.length,bufferCount+1,'quarantine permits no second phase allocation');
  device.queue.onSubmittedWorkDone=async()=>{};
  await source.dispose();
  assert.equal(budget.snapshot().gpu.liveBytes,0,'recovery drains before owned retirement');
  await assert.rejects(()=>source.withWeights({camera:source.template.cameraEmbedder},async()=>{}),/disposed/);
}finally{globalThis.fetch=original;device.queue.onSubmittedWorkDone=async()=>{};await source?.dispose();device.destroy();budget.restore();}
console.log('phase weight construction, all 24 blocks, pre-read selection/source refusal, fences and retirement pass');
