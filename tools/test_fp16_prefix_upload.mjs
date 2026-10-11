import assert from 'node:assert/strict';
import {createWeightPhaseSource} from '../src/lib/weights.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {parseFlatWeightHeader} from '../src/lib/flat_tensor_ranges.js';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
const count=2*262144+2,etag='"prefix-upload-fixture"';
const fixture=weightFixture({fp16Names:new Set(['tokenizer.embeddings']),
  tensorShapes:new Map([['tokenizer.embeddings',[count]]])});
const headerBytes=new DataView(fixture.bytes.buffer).getUint32(12,true);
const info=parseFlatWeightHeader(fixture.bytes.buffer.slice(0,headerBytes),fixture.bytes.length).tensors.get('tokenizer.embeddings');
const half=new Uint16Array(fixture.bytes.buffer,info.offset,count);
for(let i=0;i<count;i++)half[i]=i%65536;
const originalFetch=globalThis.fetch;let fetches=0;
globalThis.fetch=async(url,init)=>{
  fetches++;assert.equal(init.headers['If-Match'],etag);
  const [start,end]=init.headers.Range.slice(6).split('-').map(Number);
  return new Response(fixture.bytes.slice(start,end+1),{status:206,headers:{
    ETag:etag,'Content-Range':'bytes '+start+'-'+end+'/'+fixture.bytes.length,'Content-Length':String(end-start+1)}});
};
const make=()=>{
  const device=fakeWeightDevice(),create=device.createBuffer.bind(device),descriptors=[],writes=[],backings=new Set();
  let pending=null,failWrite=false,failFence=false,failDestroy=false;
  device.createBuffer=desc=>{descriptors.push(desc);const b=create(desc),destroy=b.destroy.bind(b);
    b.destroy=()=>{assert.equal(pending,null,'no destruction before completed prefix');if(failDestroy)throw Error('upload destroy unresolved');destroy();};return b;};
  device.queue.writeBuffer=(buffer,offset,data,dataOffset=0,size=data.length-dataOffset)=>{
    if(failWrite)throw Error('injected queue write failure');
    assert.equal(pending,null,'one upload prefix completes before scratch reuse');
    assert.equal(data instanceof Float32Array,true);assert.equal(dataOffset,0);
    backings.add(data.buffer);writes.push({offset,bytes:size*4});pending={buffer,offset,data,size};
  };
  device.queue.onSubmittedWorkDone=async()=>{
    if(failFence)throw Error('injected unresolved upload prefix');
    if(pending){const p=pending;new Uint8Array(p.buffer.data,p.offset,p.size*4).set(new Uint8Array(p.data.buffer,p.data.byteOffset,p.size*4));pending=null;}
  };
  const budget=createLoaderMemoryBudget({cpuBytes:16*1024*1024,gpuBytes:16*1024*1024});budget.bindOwnedDevice(device);
  return {device,budget,descriptors,writes,backings,setFaults(w,f,d){failWrite=w;failFence=f;failDestroy=d;}};
};
let current;
try{
  current=make();let source=await createWeightPhaseSource(current.device,'fixture',{
    memoryBudget:current.budget,expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag,streamFp16Uploads:true});
  await source.withWeights(source.template.tokenizer,selected=>{
    assert.equal(current.descriptors[0].mappedAtCreation,false,'streamed phase must not allocate complete mapped-at-creation FP32 staging');
    assert.equal(current.descriptors[0].usage&GPUBufferUsage.COPY_DST,GPUBufferUsage.COPY_DST);
    assert.equal(selected.embeddings.size,count*4,'full destination is not truncated');
    assert.deepEqual(current.writes,[{offset:0,bytes:1048576},{offset:1048576,bytes:1048576},{offset:2097152,bytes:8}]);
    assert.equal(current.backings.size,1,'one reused conversion working backing');
    const values=new Float32Array(selected.embeddings.data);
    for(let i=0;i<count;i++){
      const bits=i%65536,sign=bits&32768?-1:1,e=(bits>>>10)&31,m=bits&1023;
      const expected=e===31?(m?NaN:sign*Infinity):e===0?sign*m*2**-24:sign*(1024+m)*2**(e-25);
      if(Number.isNaN(expected))assert(Number.isNaN(values[i]));else assert(Object.is(values[i],expected),'complete converted value differs at '+i);
    }
  });
  assert.equal(source.loadingReport.gpuFp16UploadMode,'queue-write-complete-prefix-v1');
  assert.equal(current.budget.snapshot().cpu.peakLiveBytes>=count*4,true,'full conversion allowance remains charged');
  assert.equal(current.budget.snapshot().gpu.liveBytes,0);assert.equal(current.budget.snapshot().cpu.liveBytes,0);
  await source.dispose();current.device.destroy();current.budget.restore();current=null;
  console.log('PASS full FP16 values/all65536 encodings/repeated chunks/tail, one unmapped complete GPU destination, one working backing, completed prefix before reuse, unchanged full allowance');

  for(const kind of ['write','fence','destroy']){
    current=make();source=await createWeightPhaseSource(current.device,'fixture',{
      memoryBudget:current.budget,expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag,streamFp16Uploads:true});
    current.setFaults(kind==='write',kind==='fence',kind==='destroy');
    let consumed=false;
    await assert.rejects(source.withWeights(source.template.tokenizer,()=>{consumed=true;}),error=>
      kind==='write'?error.message==='injected queue write failure':
      error instanceof AggregateError&&error.errors.some(e=>e.message===
        (kind==='fence'?'injected unresolved upload prefix':'upload destroy unresolved')));
    assert.equal(consumed,kind==='destroy','only fully uploaded values reach consumer');
    if(kind==='write'){
      assert.equal(current.budget.snapshot().gpu.liveBytes,0);assert.equal(source.phases.at(-1).status,'failed-retired');
    }else{
      assert.equal(source.phases.at(-1).status,'cleanup-unresolved');
      assert.equal(current.budget.snapshot().gpu.liveBytes,count*4);
      if(kind==='fence')assert.equal(current.budget.snapshot().cpu.liveBytes,count*4,'unresolved working storage allowance remains owned');
      await assert.rejects(source.withWeights(source.template.tokenizer,()=>{}),/quarantined/);
      assert.equal(current.descriptors.length,1,'no replacement destination while quarantined');
    }
    current.setFaults(false,false,false);await source.dispose();
    assert.equal(current.budget.snapshot().gpu.liveBytes,0);assert.equal(current.budget.snapshot().cpu.liveBytes,0);
    current.device.destroy();current.budget.restore();current=null;
  }
  const before=fetches,currentInvalid=make();
  for(const streamFp16Uploads of ['true',null])await assert.rejects(createWeightPhaseSource(currentInvalid.device,'fixture',{
    memoryBudget:currentInvalid.budget,expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag,streamFp16Uploads}),/boolean/);
  assert.equal(fetches,before,'invalid upload mode refuses before source IO');
  currentInvalid.device.destroy();currentInvalid.budget.restore();
  console.log('PASS write/prefix/destroy failure preserves actual phase custody and live charges, quarantine/no replacement/recovery; invalid mode refuses before IO');
}finally{if(current){current.setFaults(false,false,false);current.device.destroy();current.budget.restore();}globalThis.fetch=originalFetch;}
