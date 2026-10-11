import assert from 'node:assert/strict';
import {TwoStreamBackbone} from '../src/lib/two_stream.js';
import {createEmptyBuffer} from '../src/lib/gpu.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
import {twoStreamPhaseDemand} from './resident_two_stream_acceptance.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
assert.equal(typeof TwoStreamBackbone.prototype._markFineAttentionResidualReusable,'function',
  'completed owned attention projection must acquire a prefix-bound residual reuse token');
function fixture(reuse=true){
  const device=fakeWeightDevice(),backbone=new TwoStreamBackbone(device),N=7,D=2;
  const old=createEmptyBuffer(device,N*D*4),projected=createEmptyBuffer(device,N*D*4);
  const a=new Float32Array(old.data),b=new Float32Array(projected.data);
  for(let i=0;i<a.length;i++){a[i]=Math.fround((i-4)/9);b[i]=Math.fround((i+2)/13);}
  // Include cancellation and both signs of zero in the actual float32 adds.
  a[0]=0;b[0]=-0;a[1]=1;b[1]=-1;a[2]=-1;b[2]=1;
  const phase={nextRangeIndex:4,rangeCount:4,nextRowStart:N,output:projected};
  const operation={kind:'fuse',ownerId:'block-0-fuse-out',N_z:N,D,zBuf:old,
    attentionProjection:phase,attention:{attnOutBuf:{}},weights:{normZ2:{}}};
  const state={activeOperation:operation,residentFFN:true,reuseDeadTriplaneStorage:true,
    reuseAttentionResidualStorage:reuse,currentTriplane:old,nextFineDutyIndex:10};
  backbone._residentWorkOwner={buffers:new Set([old,projected]),allocations:[]};
  backbone._diagnosticBuffers={};
  const copies=[],encoder={copyBufferToBuffer(src,o,dst,p,size){
    copies.push({src,dst,size});new Uint8Array(dst.data,p,size).set(new Uint8Array(src.data,o,size));
  }};
  backbone._dispatchAdd=(enc,dst,src,count)=>{
    const x=new Float32Array(dst.data),y=new Float32Array(src.data);
    for(let i=0;i<count;i++)x[i]=Math.fround(x[i]+y[i]);
  };
  backbone._dispatchLayerNorm=(enc,input,output)=>{
    new Uint8Array(output.data).set(new Uint8Array(input.data));
  };
  return {device,backbone,N,D,old,projected,phase,operation,state,encoder,copies};
}
const duty={kind:'fuse-attention-linear-range',dutyIndex:9,block:0,rangeIndex:3,rangeCount:4};
const f=fixture(),legacy=fixture(false),before=f.device.buffers.length;
const token=f.backbone._markFineAttentionResidualReusable(f.state,duty);
assert.equal(token.bytes,56);assert.equal(token.afterDutyIndex,9);
assert.equal(token.source,'owned-projected-attention-after-gpu-prefix');
const residualDuty={kind:'fuse-residual-norm',dutyIndex:10,block:0};
f.backbone._dispatchFineFuseResidualNorm(f.encoder,f.state,residualDuty);
legacy.backbone._dispatchFineFuseResidualNorm(legacy.encoder,legacy.state,residualDuty);
assert.equal(f.operation.z1Buf,f.projected,'reuse exact already-complete projected output');
assert.notEqual(f.operation.z1Buf,f.old,'old triplane remains intact for its later FFN reuse');
assert.equal(f.device.buffers.length-before,1,'only complete normalization output is new');
assert.equal(legacy.device.buffers.length,4,'default still allocates both complete residual and norm');
assert.equal(f.copies.length,0,'never overwrite projected attention before adding the residual');
assert.deepEqual(new Uint8Array(f.operation.z1Buf.data),new Uint8Array(legacy.operation.z1Buf.data));
assert.deepEqual(new Uint8Array(f.operation.z2NormBuf.data),new Uint8Array(legacy.operation.z2NormBuf.data));
for(const mutate of [x=>x.backbone._residentWorkOwner.buffers.delete(x.projected),
  x=>x.phase.nextRowStart--,x=>x.phase.nextRangeIndex--,x=>x.state.nextFineDutyIndex--,
  x=>x.phase.output=x.old,x=>x.state.reuseAttentionResidualStorage=false]){
  const x=fixture();mutate(x);const count=x.device.buffers.length;
  assert.throws(()=>x.backbone._markFineAttentionResidualReusable(x.state,duty),/owned|prefix|projection|reuse/i);
  assert.equal(x.device.buffers.length,count);
}
for(const mutate of [x=>delete x.operation.reusableAttentionResidualStorage,
  x=>x.operation.reusableAttentionResidualStorage.afterDutyIndex--,
  x=>x.backbone._residentWorkOwner.buffers.delete(x.projected),
  x=>x.phase.output=x.old]){
  const x=fixture();x.backbone._markFineAttentionResidualReusable(x.state,duty);mutate(x);
  const count=x.device.buffers.length,bytes=new Uint8Array(x.projected.data).slice();
  assert.throws(()=>x.backbone._dispatchFineFuseResidualNorm(x.encoder,x.state,residualDuty),/owned|prefix|projection|reuse/i);
  assert.equal(x.device.buffers.length,count,'invalid token refuses before new allocation');
  assert.deepEqual(new Uint8Array(x.projected.data),bytes,'invalid token cannot overwrite owned/borrowed data');
}
assert.equal(twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-residual-norm'},
  reuseDeadTriplaneStorage:true,reuseAttentionResidualStorage:true}).workGpuBytes,113246208+4096);
assert.equal(twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-residual-norm'}}).workGpuBytes,2*113246208+4096);
for(const value of [true,'yes',1])assert.throws(()=>new TwoStreamBackbone(fakeWeightDevice())
  .createAttentionForwardState({},1,{}, {reuseAttentionResidualStorage:value}),/resident|reuse/i);
console.log('PASS owned projected attention residual reuse: complete finite float32 sums, exact prefix/store custody, real reduced new demand; local semantics only');

globalThis.requestAnimationFrame=callback=>queueMicrotask(()=>callback(performance.now()));
const {runCooperativeTwoStream,retireTwoStreamWork}=await import('../src/lib/cooperative_two_stream.js');
for(const failPrefix of [false,true]){
  const x=fixture(),input=createEmptyBuffer(x.device,16),marked=[];
  await retireTwoStreamWork(x.backbone);
  let lastEncoded=null,lastFenced=null,blocked=false;
  x.device.queue.onSubmittedWorkDone=async()=>{
    if(failPrefix&&lastEncoded?.kind==='fuse-attention-linear-range'&&lastEncoded.rangeIndex===lastEncoded.rangeCount-1){
      blocked=true;throw Error('failed actual projection prefix');
    }
    if(blocked)throw Error('failed actual projection prefix');lastFenced=lastEncoded?.dutyIndex;
  };
  x.device.createCommandEncoder=()=>({...x.encoder,finish(){return {};}});
  const mark=x.backbone._markFineAttentionResidualReusable.bind(x.backbone);
  x.backbone._markFineAttentionResidualReusable=(state,duty)=>{
    assert.equal(lastFenced,duty.dutyIndex,'reuse must follow the actual projection queue prefix');
    const receipt=mark(state,duty);marked.push(receipt);return receipt;
  };
  x.backbone.dispatchAttentionForwardDuty=(encoder,state,index)=>{
    const duty=state.finePlan[index];lastEncoded=duty;
    if(index===0){state.currentLatent=createEmptyBuffer(x.device,16);state.currentTriplane=createEmptyBuffer(x.device,56);}
    if(duty.kind==='fuse-attention-linear-range'&&duty.rangeIndex===duty.rangeCount-1){
      state.activeOperation={kind:'fuse',ownerId:`block-${duty.block}-fuse-out`,N_z:7,D:2,zBuf:state.currentTriplane,
        attention:{attnOutBuf:{}},weights:{normZ2:{}},
        attentionProjection:{nextRangeIndex:duty.rangeCount,rangeCount:duty.rangeCount,nextRowStart:7,output:createEmptyBuffer(x.device,56)}};
    }
    if(duty.kind==='fuse-residual-norm')x.backbone._dispatchFineFuseResidualNorm(encoder,state,duty);
    if(duty.kind==='fuse-final-residual'){state.currentTriplane=state.activeOperation.z1Buf;state.activeOperation=null;}
    if(index===state.finePlan.length-1){state.activeOperation=null;state.result={buffer:createEmptyBuffer(x.device,16)};}
    state.nextFineDutyIndex=index+1;
  };
  x.backbone.getForwardResult=state=>state.result;
  const options={device:x.device,backbone:x.backbone,N_img:1,imageTokensBuf:input,weights:{},
    dutyGranularity:'attention-tile',residentFFN:true,retireIntermediateBuffers:true,
    reuseDeadTriplaneStorage:true,reuseAttentionResidualStorage:true};
  if(failPrefix){
    await assert.rejects(runCooperativeTwoStream(options),/prefix|unresolved/);
    assert.equal(marked.length,0,'failed queue prefix cannot grant any reuse token');
    assert.ok(x.backbone._residentWorkOwner,'failed prefix retains exact inventory for recovery');
    blocked=false;lastEncoded=null;await retireTwoStreamWork(x.backbone);
  }else{
    const result=await runCooperativeTwoStream(options);
    assert.deepEqual(marked.map(r=>r.block),[0,1,2,3],'all four complete projections transfer after their own prefix');
    assert.deepEqual(result.report.adapterTelemetry.attentionResidualReuse,marked);
    assert.equal(result.report.adapterTelemetry.reuseAttentionResidualStorage,true);
    result.result.buffer.destroy();
  }
  assert.equal(input.destroyed,0,'borrowed input remains untouched');assert.equal(x.device.destroyed,0);
}
console.log('PASS actual kit prefix marking/recovery and full four-block reuse telemetry; synthetic allocation route, not native math');
