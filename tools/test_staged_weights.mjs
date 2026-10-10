import assert from 'node:assert/strict';
import {weightFixture, fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
import {loadWeights} from '../src/lib/weights.js';
import {loadWeightTensorUnit} from '../src/lib/weights.js';
import {parseFlatWeightHeader} from '../src/lib/flat_tensor_ranges.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';

globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
const name='image_tokenizer.image_mean';
const fixture=weightFixture({tensorShapes:new Map([[name,[8]]]),fp16Names:new Set([name])});
const h=new DataView(fixture.bytes.buffer).getUint32(12,true);
const words=[0,0x8000,1,0x3c00,0xbc00,0x7c00,0xfc00,0x7e00];
words.forEach((v,i)=>new DataView(fixture.bytes.buffer).setUint16(h+2*i,v,true));
const original=globalThis.fetch, etag='"fixture-revision-1"';
let calls=[], pendingDrain=false;
globalThis.fetch=async (url,init)=>{
  assert.equal(pendingDrain,false,'next tensor cannot be fetched before prior queue completion');
  assert.equal(init?.headers?.['If-Match'],etag);
  const match=init?.headers?.Range?.match(/^bytes=(\d+)-(\d+)$/);
  assert.ok(match,'staged mode must never request the whole checkpoint');
  const start=Number(match[1]),end=Number(match[2]);calls.push({start,end});
  return new Response(fixture.bytes.slice(start,end+1),{status:206,headers:{ETag:etag,'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`,'Content-Length':String(end-start+1)}});
};
const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:2*h+2048,gpuBytes:1_000_000});
device.queue.onSubmittedWorkDone=async()=>{pendingDrain=true;await Promise.resolve();pendingDrain=false;};
budget.bindOwnedDevice(device);
try{
  const weights=await loadWeights(device,'fixture.bin',null,{memoryBudget:budget,expectedWeightBytes:fixture.bytes.length,loadingMode:'tensor-ranges',expectedSourceETag:etag});
  const values=new Float32Array(weights.imageTokenizer.imageMean.data);
  assert.equal(values.length,8);assert.ok(Object.is(values[1],-0));assert.equal(values[2],2**-24);
  assert.deepEqual([...values.slice(3,7)],[1,-1,Infinity,-Infinity]);assert.ok(Number.isNaN(values[7]));
  assert.deepEqual(calls.slice(0,2),[{start:0,end:15},{start:0,end:h-1}]);
  assert.equal(calls.length,fixture.values.size+2);
  assert.equal(weights.loadingReport.mode,'tensor-ranges');
  assert.equal(weights.loadingReport.sourceETag,etag);
  assert.ok(weights.loadingReport.maximumSourceUnitBytes<fixture.bytes.length);
  const lazy='image_estimator.model.visual.ln_pre.weight';
  assert.equal(new Float32Array(weights._rawGet(lazy).data)[0],fixture.values.get(lazy));
  assert.equal(weights._rawGetCPU(lazy)[0],fixture.values.get(lazy));
  weights.dispose();assert.equal(budget.snapshot().cpu.liveBytes,0);assert.equal(budget.snapshot().gpu.liveBytes,0);
}finally{globalThis.fetch=original;device.destroy();budget.restore();}
console.log('staged SF3D loading preserves mixed-dtype bytes and lazy consumers without a whole-file fetch');

for(const kind of ['ignored-range','wrong-range','changed-source','truncated','overlong','queue-failure','cpu-refusal','gpu-refusal']){
  const dev=fakeWeightDevice(),b=createLoaderMemoryBudget({cpuBytes:kind==='cpu-refusal'?31:2*h+2048,gpuBytes:kind==='gpu-refusal'?31:1_000_000});
  b.bindOwnedDevice(dev);let payloads=0;
  globalThis.fetch=async(url,init)=>{
    const match=init.headers.Range.match(/^bytes=(\d+)-(\d+)$/),start=Number(match[1]),end=Number(match[2]);
    const payload=start>=h;payloads+=Number(payload);
    let bytes=fixture.bytes.slice(start,end+1);
    if(payload&&kind==='truncated')bytes=bytes.slice(0,-1);
    if(payload&&kind==='overlong')bytes=new Uint8Array(bytes.length+1);
    return new Response(bytes,{status:payload&&kind==='ignored-range'?200:206,headers:{
      ETag:payload&&kind==='changed-source'?'"different"':etag,
      'Content-Range':`bytes ${payload&&kind==='wrong-range'?start+1:start}-${end}/${fixture.bytes.length}`,
      'Content-Length':String(end-start+1)}});
  };
  if(kind==='queue-failure')dev.queue.onSubmittedWorkDone=async()=>{throw Error('queue failed');};
  try{
    await assert.rejects(loadWeightTensorUnit(dev,'fixture.bin',[name],{memoryBudget:b,expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag}),
      kind==='queue-failure'?/queue failed/:kind.endsWith('refusal')?/before allocation/:/range.*(mismatch|truncated|exceed)/);
    assert.equal(payloads,kind==='cpu-refusal'?0:1);
    assert.equal(dev.buffers.length,kind==='queue-failure'?1:0);
    assert.ok(dev.buffers.every(buffer=>buffer.destroyed===1));
    assert.equal(b.snapshot().cpu.liveBytes,0);assert.equal(b.snapshot().gpu.liveBytes,0);
  }finally{globalThis.fetch=original;dev.destroy();b.restore();}
}
{
  const header=fixture.bytes.slice(0,h),view=new DataView(header.buffer);
  // Canonical writer behavior: rank 5 with only four stored axes. Do not read
  // offset as axis 5. Payload semantics remain bound by dtype/size, not a fake shape.
  view.setUint32(16+132,5,true);[1,1,8,1].forEach((d,i)=>view.setUint32(16+136+4*i,d,true));
  const info=parseFlatWeightHeader(header.buffer,fixture.bytes.length).tensors.get(name);
  assert.equal(info.shapeComplete,false);assert.equal(info.declaredRank,5);assert.deepEqual(info.shape,[1,1,8,1]);
  view.setUint32(16+128,2,true);
  assert.throws(()=>parseFlatWeightHeader(header.buffer,fixture.bytes.length),/invalid.*tensor/);
}
console.log('range fallback/source drift, partial bytes, queue failure and allocation refusal all fail closed with owned cleanup');
