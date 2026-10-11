import assert from 'node:assert/strict';
import {TwoStreamBackbone} from '../src/lib/two_stream.js';
import {createEmptyBuffer} from '../src/lib/gpu.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
import {twoStreamPhaseDemand} from './resident_two_stream_acceptance.mjs';
import {runCooperativeTwoStream,retireTwoStreamWork} from '../src/lib/cooperative_two_stream.js';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};

// Real dependency/copy/coverage methods, tiny deterministic row arithmetic.
// This proves local reuse semantics, not native shader numerical parity.
function fixture(reuse){
  const device=fakeWeightDevice(),backbone=new TwoStreamBackbone(device),N=7,D=2;
  backbone._diagnosticBuffers={};
  const old=createEmptyBuffer(device,N*D*4),residual=createEmptyBuffer(device,N*D*4),norm=createEmptyBuffer(device,N*D*4);
  new Float32Array(old.data).fill(-100);
  const a=new Float32Array(residual.data),b=new Float32Array(norm.data);
  for(let i=0;i<N*D;i++){a[i]=Math.fround((i-4)/9);b[i]=Math.fround((i+2)/13);}
  const expected=Array.from(a,(v,i)=>Math.fround(v+Math.fround(b[i]*1.25-0.25)));
  const operation={kind:'fuse',ownerId:'block-0-fuse-out',N_z:N,D,zBuf:old,z1Buf:residual,z2NormBuf:norm,weights:{ff:{}},
    attention:null,attentionProjection:null,reusableTriplaneStorage:{buffer:old,afterDutyIndex:10}};
  const state={residentFFN:true,reuseDeadTriplaneStorage:reuse,ffnRowsPerTile:2,currentTriplane:old,currentLatent:{},activeOperation:operation,nextFineDutyIndex:11};
  backbone._residentWorkOwner={buffers:new Set([old,residual,norm]),allocations:[]};
  const calls=[],copies=[];
  const encoder={copyBufferToBuffer(src,offset,dst,dstOffset,size){
    copies.push({src,dst,size});new Uint8Array(dst.data,dstOffset,size).set(new Uint8Array(src.data,offset,size));
  }};
  backbone._dispatchFFNRow=(enc,input,output,scratch,ff,start,count,dim)=>{
    calls.push({start,count,output});const src=new Float32Array(input.data),dst=new Float32Array(output.data);
    for(let i=start*dim;i<(start+count)*dim;i++)dst[i]=Math.fround(src[i]*1.25-0.25);
  };
  backbone._dispatchAdd=(enc,dst,src,count)=>{
    const a=new Float32Array(dst.data),b=new Float32Array(src.data);
    for(let i=0;i<count;i++)a[i]=Math.fround(a[i]+b[i]);
  };
  const duties=Array.from({length:4},(_,rangeIndex)=>{
    const rowStart=rangeIndex*2,rowCount=Math.min(2,N-rowStart);
    return {dutyIndex:11+rangeIndex,block:0,ownerId:'block-0-fuse-out-ffn-projection',rangeIndex,
      rangeCount:4,totalRows:N,rowStart,rowCount,rowEnd:rowStart+rowCount};
  });
  return {device,backbone,N,D,old,residual,norm,operation,state,encoder,calls,copies,duties,expected};
}
const f=fixture(true);
for(const duty of f.duties)f.backbone._dispatchFineFuseResidentFFNRange(f.encoder,f.state,duty);
assert.equal(f.operation.ffnProjection.output,f.old,'reuse must overwrite the already-owned dead triplane, not allocate a complete FFN output');
assert.deepEqual(f.calls.map(c=>[c.start,c.count]),[[0,2],[2,2],[4,2],[6,1]],'all rows and the partial tail execute');
assert.equal(f.device.buffers.length,7,'three existing values plus four scratch stores, no new complete output');
f.backbone._dispatchFineFuseFinalResidual(f.encoder,f.state,{block:0});
assert.equal(f.state.currentTriplane,f.residual,'same-order final add reuses its initialized residual destination');
assert.equal(f.device.buffers.length,7,'final residual must not allocate another complete triplane');
assert.deepEqual(Array.from(new Float32Array(f.state.currentTriplane.data)),f.expected);
assert.equal(f.copies.length,0,'no self-copy or destructive overwrite of the completed FFN output');

const legacy=fixture(false);
for(const duty of legacy.duties)legacy.backbone._dispatchFineFuseResidentFFNRange(legacy.encoder,legacy.state,duty);
legacy.backbone._dispatchFineFuseFinalResidual(legacy.encoder,legacy.state,{block:0});
assert.notEqual(legacy.state.currentTriplane,legacy.residual,'default route remains allocation-compatible');
assert.deepEqual(new Uint8Array(legacy.state.currentTriplane.data),new Uint8Array(f.state.currentTriplane.data));

for(const mutate of [x=>delete x.operation.reusableTriplaneStorage,
  x=>x.operation.reusableTriplaneStorage.afterDutyIndex=9,
  x=>x.backbone._residentWorkOwner.buffers.delete(x.old),
  x=>x.operation.reusableTriplaneStorage.buffer=x.norm]){
  const x=fixture(true);mutate(x);const before=x.device.buffers.length;
  assert.throws(()=>x.backbone._dispatchFineFuseResidentFFNRange(x.encoder,x.state,x.duties[0]),/owned|prefix|reuse/i);
  assert.equal(x.device.buffers.length,before,'invalid reuse authority refuses before scratch allocation');
  assert.deepEqual(Array.from(new Float32Array(x.old.data)),Array(14).fill(-100),'borrowed/stale storage remains untouched');
}
const prefix=fixture(true);delete prefix.operation.reusableTriplaneStorage;
prefix.state.nextFineDutyIndex=11;
prefix.backbone._markFineFuseStorageReusable(prefix.state,{kind:'fuse-residual-norm',dutyIndex:10,block:0});
assert.equal(prefix.operation.reusableTriplaneStorage.buffer,prefix.old);
assert.equal(prefix.operation.reusableTriplaneStorage.afterDutyIndex,10);
for(const mutate of [x=>x.state.nextFineDutyIndex=10,x=>x.backbone._residentWorkOwner.buffers.delete(x.old),
  x=>x.operation.attention={pending:true},x=>x.operation.z1Buf=x.old]){
  const x=fixture(true);delete x.operation.reusableTriplaneStorage;mutate(x);
  assert.throws(()=>x.backbone._markFineFuseStorageReusable(x.state,{kind:'fuse-residual-norm',dutyIndex:10,block:0}),/owned|prefix|reuse/i);
  assert.equal(x.operation.reusableTriplaneStorage,undefined);
}
const demand=twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-resident-ffn-range',rangeIndex:0},reuseDeadTriplaneStorage:true});
assert.equal(demand.workGpuBytes,7344128,'fresh admission counts the four 128-row scratch stores and uniforms, not a new113246208-byte output');
assert.equal(twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-final-residual'},reuseDeadTriplaneStorage:true}).workGpuBytes,4096);
assert.equal(twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-resident-ffn-range',rangeIndex:0}}).workGpuBytes,120590336,'ordinary demand remains unchanged');
for(const value of [true,'yes',1]){
  const device=fakeWeightDevice(),backbone=new TwoStreamBackbone(device);
  await assert.rejects(runCooperativeTwoStream({device,backbone,N_img:1,imageTokensBuf:{},weights:{},
    dutyGranularity:'attention-tile',residentFFN:true,reuseDeadTriplaneStorage:value}),/owned work retirement/);
  assert.equal(device.buffers.length,0,'reuse without owned retirement fails before model allocation');
}
assert.throws(()=>new TwoStreamBackbone(fakeWeightDevice()).createAttentionForwardState({},1,{},
  {reuseDeadTriplaneStorage:true}),/resident FFN/);

// Exercise the real kit prefix and wrapper's marking placement; native math is
// deliberately replaced by tiny allocations, never a backend/parity witness.
globalThis.requestAnimationFrame=callback=>queueMicrotask(()=>callback(performance.now()));
for(const failPrefix of [false,true]){
  const x=fixture(true),input=createEmptyBuffer(x.device,16),marked=[];
  await retireTwoStreamWork(x.backbone);
  let lastEncoded=null,lastFenced=null,blocked=false;
  x.device.queue.onSubmittedWorkDone=async()=>{
    if(failPrefix&&lastEncoded?.kind==='fuse-residual-norm'){blocked=true;throw Error('failed actual prefix');}
    if(blocked)throw Error('failed actual prefix');lastFenced=lastEncoded?.dutyIndex;
  };
  x.device.createCommandEncoder=()=>({finish(){return {};}});
  const mark=x.backbone._markFineFuseStorageReusable.bind(x.backbone);
  x.backbone._markFineFuseStorageReusable=(state,duty)=>{
    assert.equal(lastFenced,duty.dutyIndex,'storage transfer must follow the actual GPU prefix');
    const result=mark(state,duty);marked.push(result);return result;
  };
  x.backbone.dispatchAttentionForwardDuty=(encoder,state,index)=>{
    const duty=state.finePlan[index];lastEncoded=duty;
    if(index===0)state.currentLatent=createEmptyBuffer(x.device,16);
    if(duty.kind==='fuse-residual-norm'){
      const a=createEmptyBuffer(x.device,56),b=createEmptyBuffer(x.device,56),c=createEmptyBuffer(x.device,56);
      state.currentTriplane=a;
      state.activeOperation={kind:'fuse',ownerId:`block-${duty.block}-fuse-out`,N_z:7,D:2,zBuf:a,z1Buf:b,z2NormBuf:c,
        attention:null,attentionProjection:null};
    }
    if(duty.kind==='fuse-final-residual')state.activeOperation=null;
    if(index===state.finePlan.length-1){state.activeOperation=null;state.result={buffer:createEmptyBuffer(x.device,16)};}
    state.nextFineDutyIndex=index+1;
  };
  x.backbone.getForwardResult=state=>state.result;
  const options={device:x.device,backbone:x.backbone,N_img:1,imageTokensBuf:input,weights:{},
    dutyGranularity:'attention-tile',residentFFN:true,retireIntermediateBuffers:true,reuseDeadTriplaneStorage:true};
  if(failPrefix){
    await assert.rejects(runCooperativeTwoStream(options),/prefix|unresolved/);
    assert.equal(marked.length,0,'failed prefix cannot mint reuse authority');
    assert.ok(x.backbone._residentWorkOwner,'failed prefix keeps exact allocations recoverable');
    blocked=false;lastEncoded=null;await retireTwoStreamWork(x.backbone);
  }else{
    const result=await runCooperativeTwoStream(options);
    assert.deepEqual(marked.map(r=>r.block),[0,1,2,3]);
    assert.deepEqual(result.report.adapterTelemetry.storageReuse,marked);
    assert.equal(result.report.adapterTelemetry.reuseDeadTriplaneStorage,true);
    result.result.buffer.destroy();
  }
  assert.equal(input.destroyed,0,'borrowed input is not reused or destroyed');
  assert.equal(x.device.destroyed,0);
}
console.log('owned dead-triplane reuse preserves all rows and same-order residual, with prefix/borrowed-storage refusal');
