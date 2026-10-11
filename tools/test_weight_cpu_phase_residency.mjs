import assert from 'node:assert/strict';
import {createWeightPhaseSource} from '../src/lib/weights.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
const half='image_estimator.model.visual.class_embedding',full='image_estimator.model.visual.positional_embedding';
const fixture=weightFixture({fp16Names:new Set([half]),tensorShapes:new Map([[half,[2]]])}),header=new DataView(fixture.bytes.buffer).getUint32(12,true);
const etag='"synthetic-cpu-phase"',original=globalThis.fetch,reads=[];
let wrong=false;
globalThis.fetch=async(url,init)=>{
  assert.equal(init.headers['If-Match'],etag);
  const [,a,b]=init.headers.Range.match(/^bytes=(\d+)-(\d+)$/),start=Number(a),end=Number(b);
  reads.push({start,end});return new Response(fixture.bytes.slice(start,end+1),{status:206,headers:{
    ETag:wrong?'"wrong-source"':etag,'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`,
    'Content-Length':String(end-start+1)}});
};
const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:2*header+4096,gpuBytes:4096});
budget.bindOwnedDevice(device);let source;
try{
  source=await createWeightPhaseSource(device,'synthetic.bin',{memoryBudget:budget,
    expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag});
  assert.equal(typeof source.reference,'function','CPU-only lazy tensors require authentic canonical table references');
  assert.equal(typeof source.withCpuWeights,'function','CPU preparation/heads must not retain whole CLIP raw payload');
  const selection={half:source.reference(half),full:source.reference(full)};
  assert.deepEqual(source.describe(selection).map(t=>t.name),[half,full]);
  const before=reads.length;
  await source.withCpuWeights(selection,async tensors=>{
    assert.equal(tensors.half[0],1,'same exact FP16 converter');
    assert.equal(tensors.full[0],fixture.values.get(full));
    assert.equal(tensors.half.length,2);assert.equal(tensors.full.length,1);
    assert.equal(device.buffers.length,0,'CPU phase allocates no needless GPU weights');
    assert.equal(budget.snapshot().cpu.liveBytes,12,'only converted owned results stay charged');
    await assert.rejects(()=>source.withCpuWeights(selection,async()=>{}),/active phase/);
  });
  assert.equal(reads.length-before,2,'only selected full tensors read');
  assert.equal(budget.snapshot().cpu.liveBytes,0);assert.equal(source.phases.at(-1).status,'completed-retired');
  assert.equal(source.phases.at(-1).storageKind,'cpu-fp32');
  for(const value of [0,false,null,undefined,'']){
    await assert.rejects(()=>source.withCpuWeights(selection,async()=>{throw value;}),e=>e===value);
    assert.equal(source.phases.at(-1).status,'failed-retired');assert.equal(budget.snapshot().cpu.liveBytes,0);
  }
  const prior=reads.length;
  await assert.rejects(()=>source.withCpuWeights({foreign:{}},async()=>{}),/selection/);
  assert.throws(()=>source.reference('missing.canonical.name'),/Missing weight/);
  assert.equal(reads.length,prior,'foreign/absent identities reject before payload');
  wrong=true;await assert.rejects(()=>source.withCpuWeights(selection,async()=>{}),/source identity mismatch/);
  assert.equal(budget.snapshot().cpu.liveBytes,0);wrong=false;
}finally{globalThis.fetch=original;await source?.dispose();device.destroy();budget.restore();}
console.log('PASS CPU-only canonical phases, exact conversion, fresh source, complete selection and failure retirement; synthetic policy only');
