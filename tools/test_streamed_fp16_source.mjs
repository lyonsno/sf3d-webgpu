import assert from 'node:assert/strict';
import {createWeightPhaseSource} from '../src/lib/weights.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {parseFlatWeightHeader} from '../src/lib/flat_tensor_ranges.js';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';

globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
const count=524290,etag='"complete-stream-source"',name='tokenizer.embeddings';
const fixture=weightFixture({fp16Names:new Set([name]),tensorShapes:new Map([[name,[count]]])});
const header=new DataView(fixture.bytes.buffer).getUint32(12,true);
const info=parseFlatWeightHeader(fixture.bytes.buffer.slice(0,header),fixture.bytes.length).tensors.get(name);
const half=new Uint16Array(fixture.bytes.buffer,info.offset,count);
for(let i=0;i<count;i++)half[i]=i%65536;
const originalFetch=globalThis.fetch,OriginalBytes=globalThis.Uint8Array;
let fault=null,current=null,source=null,fullDestinations=0,payloadRequests=0;
const make=()=>{
  const device=fakeWeightDevice(),create=device.createBuffer.bind(device),working=new Set(),writes=[];
  let pending=null;
  device.createBuffer=descriptor=>{
    const buffer=create(descriptor),destroy=buffer.destroy.bind(buffer);
    buffer.destroy=()=>{if(fault==='destroy')throw Error('stream destroy unresolved');destroy();};return buffer;
  };
  device.queue.writeBuffer=(buffer,offset,data,dataOffset=0,size=data.length-dataOffset)=>{
    if(fault==='write')throw Error('stream write failure');
    assert.equal(pending,null,'no reuse before a completed upload prefix');
    assert(data instanceof Float32Array);assert.equal(dataOffset,0);
    working.add(data.buffer);writes.push({offset,size});pending={buffer,offset,data,size};
  };
  device.queue.onSubmittedWorkDone=async()=>{
    if(fault==='fence'&&pending)throw Error('stream queue unresolved');
    await Promise.resolve();
    if(pending){const p=pending;new OriginalBytes(p.buffer.data,p.offset,p.size*4).set(new OriginalBytes(p.data.buffer,p.data.byteOffset,p.size*4));pending=null;}
  };
  const budget=createLoaderMemoryBudget({cpuBytes:16*1024*1024,gpuBytes:16*1024*1024});budget.bindOwnedDevice(device);
  return {device,budget,working,writes,get pending(){return pending;}};
};
globalThis.fetch=async(url,init)=>{
  assert.equal(init.headers['If-Match'],etag);
  const [start,end]=init.headers.Range.slice(6).split('-').map(Number),size=end-start+1,payload=start===info.offset;
  if(payload)payloadRequests++;
  let sent=0,index=0;
  const chunks=[1,65539,7,524289,3,65537];
  const reader={
    async read(){
      assert.equal(current.pending,null,'source reader backpressure follows queue prefix completion');
      if(payload&&fault==='read'&&sent>=524288)throw Error('stream body failure');
      const effectiveSize=size-(payload&&fault==='truncated'?1:0);
      if(sent===effectiveSize){
        if(payload&&fault==='overlong'&&index++<1000)return {done:false,value:new OriginalBytes(1)};
        return {done:true};
      }
      const take=Math.min(effectiveSize-sent,payload?chunks[index++%chunks.length]:effectiveSize);
      const value=fixture.bytes.subarray(start+sent,start+sent+take);sent+=take;return {done:false,value};
    },
    async cancel(){},releaseLock(){},
  };
  return {status:payload&&fault==='status'?200:206,headers:new Headers({
    ETag:payload&&fault==='etag'?'"other-source"':etag,
    'Content-Range':payload&&fault==='range'?'bytes 0-1/2':'bytes '+start+'-'+end+'/'+fixture.bytes.length,
    'Content-Length':String(payload&&fault==='length'?size+1:size)}),body:{getReader(){return reader;}}};
};
const open=async()=>createWeightPhaseSource(current.device,'fixture',{memoryBudget:current.budget,
  expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag,streamFp16Uploads:true});
const close=async()=>{fault=null;await source.dispose();current.device.destroy();current.budget.restore();current=null;source=null;};
try{
  current=make();
  globalThis.Uint8Array=new Proxy(OriginalBytes,{construct(target,args){
    if(args.length===1&&args[0]===info.size)fullDestinations++;
    return Reflect.construct(target,args);
  }});
  source=await open();
  await source.withWeights(source.template.tokenizer,selected=>{
    assert.equal(fullDestinations,0,'streamed upload must not materialize a complete raw FP16 destination');
    assert.equal(selected.embeddings.size,count*4);
    assert.deepEqual(current.writes,[{offset:0,size:262144},{offset:1048576,size:262144},{offset:2097152,size:2}]);
    assert.equal(current.working.size,1,'one complete reusable conversion backing');
    const values=new Float32Array(selected.embeddings.data);
    for(let i=0;i<count;i++){
      const bits=i%65536,sign=bits&32768?-1:1,e=(bits>>>10)&31,m=bits&1023;
      const expected=e===31?(m?NaN:sign*Infinity):e===0?sign*m*2**-24:sign*(1024+m)*2**(e-25);
      assert(Number.isNaN(expected)?Number.isNaN(values[i]):Object.is(values[i],expected),'converted value differs at '+i);
    }
  });
  globalThis.Uint8Array=OriginalBytes;
  assert.equal(source.loadingReport.gpuFp16UploadMode,'queue-write-complete-source-prefix-v1');
  assert.equal(source.loadingReport.maximumSourceUnitBytes,info.size,'source unit remains the entire tensor');
  assert(source.loadingReport.ranges.some(r=>r.offset===info.offset&&r.size===info.size));
  assert(current.budget.snapshot().cpu.peakLiveBytes>=2*info.size+count*4,'full response/destination/conversion reservations remain');
  assert.equal(current.budget.snapshot().cpu.liveBytes,0);assert.equal(current.budget.snapshot().gpu.liveBytes,0);
  await close();
  for(const kind of ['etag','range','status','length','truncated','overlong','read','write','fence','destroy']){
    current=make();source=await open();fault=kind;let consumed=false;
    const expected=['etag','range','status','length'].includes(kind)?'weight range response/source identity mismatch':
      kind==='truncated'?'weight range bytes truncated':kind==='overlong'?'weight range bytes exceed requested unit':
      kind==='read'?'stream body failure':kind==='write'?'stream write failure':
      kind==='fence'?'stream queue unresolved':'stream destroy unresolved';
    await assert.rejects(source.withWeights(source.template.tokenizer,()=>{consumed=true;}),error=>
      error instanceof AggregateError?error.errors.some(e=>e.message.includes(expected)):error.message.includes(expected));
    assert.equal(consumed,kind==='destroy','incomplete/wrong source never reaches model work');
    if(kind==='fence'||kind==='destroy'){
      assert.equal(source.phases.at(-1).status,'cleanup-unresolved');
      assert.equal(current.budget.snapshot().gpu.liveBytes,count*4);
      if(kind==='fence')assert.equal(current.budget.snapshot().cpu.liveBytes,count*4,'unresolved conversion backing stays owned');
      const before=payloadRequests;
      await assert.rejects(source.withWeights(source.template.tokenizer,()=>{}),/quarantined/);
      assert.equal(payloadRequests,before);
    }else{assert.equal(current.budget.snapshot().gpu.liveBytes,0);assert.equal(current.budget.snapshot().cpu.liveBytes,0);}
    await close();
  }
  console.log('PASS complete source stream/odd byte boundaries/all65536 half encodings/tail/no full raw destination/one reusable backing/unchanged full allowance');
  console.log('PASS wrong HTTP identity/route/length, partial/overlong/read/write/fence/destroy failures cannot reach model work; exact unresolved owner/charge/quarantine/recovery');
}finally{
  globalThis.Uint8Array=OriginalBytes;globalThis.fetch=originalFetch;
  if(current){fault=null;await source?.dispose();current.device.destroy();current.budget.restore();}
}
