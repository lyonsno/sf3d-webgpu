import assert from 'node:assert/strict';
import {TwoStreamBackbone} from '../src/lib/two_stream.js';
import {createEmptyBuffer,captureGpuBufferAllocations} from '../src/lib/gpu.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
import {twoStreamPhaseDemand} from './resident_two_stream_acceptance.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
function fixture(reuse){
 const device=fakeWeightDevice(),backbone=new TwoStreamBackbone(device),N=7,D=5,capacity=3;
 const buffer=n=>createEmptyBuffer(device,n*4),original=buffer(N*D),old=buffer(N*D),weight=buffer(D*D),bias=buffer(D);
 new Float32Array(original.data).set(Array.from({length:N*D},(_,i)=>Math.fround((i-13)/19)));
 new Float32Array(weight.data).set(Array.from({length:D*D},(_,i)=>Math.fround((i%7-3)/11)));
 new Float32Array(bias.data).set(Array.from({length:D},(_,i)=>Math.fround((i-2)/13)));
 const operation={kind:'fuse',ownerId:'block-0-fuse-out',N_z:N,D,zBuf:old,
  attention:{nextTileIndex:2,tileCount:2,qBuf:original,attnOutBuf:original,attnWeights:{proj:{weight,bias}}}};
 const state={activeOperation:operation,residentFFN:true,reuseDeadTriplaneStorage:true,reuseAttentionResidualStorage:true,
  reuseAttentionQueryStorage:true,reuseAttentionProjectionStorage:reuse,ffnRowsPerTile:capacity,nextFineDutyIndex:0};
 backbone._residentWorkOwner={buffers:new Set([original,old]),allocations:[]};
 const calls=[],encoder={copyBufferToBuffer(src,o,dst,p,n){
  assert.notEqual(src,dst,'copy cache never aliases the source');calls.push({kind:'copy',src,dst,bytes:n});
  new Uint8Array(dst.data,p,n).set(new Uint8Array(src.data,o,n));}};
 backbone._dispatchLinearRange=(enc,input,output,w,b,totalRows,inDim,outDim,start,count)=>{
  assert.notEqual(input,output,'writable/read projection binding must not alias');calls.push({kind:'linear',input,output,totalRows,start,count});
  const a=new Float32Array(input.data),matrix=new Float32Array(w.data),bb=new Float32Array(b.data),target=new Float32Array(output.data);
  for(let row=start;row<start+count;row++)for(let col=0;col<outDim;col++){
   const sums=[0,0,0,0];let k=0;
   for(;k<Math.floor(inDim/4)*4;k+=4)for(let j=0;j<4;j++)sums[j]=Math.fround(sums[j]+Math.fround(a[row*inDim+k+j]*matrix[(k+j)*outDim+col]));
   for(;k<inDim;k++)sums[0]=Math.fround(sums[0]+Math.fround(a[row*inDim+k]*matrix[k*outDim+col]));
   target[row*outDim+col]=Math.fround(Math.fround(Math.fround(sums[0]+sums[1])+Math.fround(sums[2]+sums[3]))+bb[col]);
  }
 };
 const duties=Array.from({length:3},(_,i)=>({kind:'fuse-attention-linear-range',direction:'out',block:0,dutyIndex:i,
  dutyId:'range-'+i,ownerId:'block-0-fuse-out-attention-projection',rangeIndex:i,rangeCount:3,totalRows:N,
  rowStart:i*capacity,rowCount:Math.min(capacity,N-i*capacity),rowEnd:Math.min(N,(i+1)*capacity)}));
 const dispatch=duty=>{
  captureGpuBufferAllocations(()=>backbone._dispatchFineFuseAttentionLinearRange(encoder,state,duty),
   {ownedAllocations:backbone._residentWorkOwner.allocations});
  for(const a of backbone._residentWorkOwner.allocations)backbone._residentWorkOwner.buffers.add(a.buffer);
  backbone._residentWorkOwner.allocations.length=0;state.nextFineDutyIndex=duty.dutyIndex+1;
 };
 return{device,backbone,N,D,capacity,original,old,state,operation,encoder,calls,duties,dispatch};
}
const current=fixture(true),legacy=fixture(false),initial=current.device.buffers.length;
const unread=new Uint8Array(current.original.data).slice();
for(const duty of current.duties){current.dispatch(duty);legacy.dispatch(duty);
 assert.deepEqual(new Uint8Array(current.original.data,duty.rowEnd*current.D*4),unread.subarray(duty.rowEnd*current.D*4),'future rows survive until own last read');}
assert.equal(current.operation.attentionProjection.output===current.original,true,'reuse exact owned complete attention-result backing, not a new full projection');
assert.deepEqual(new Uint8Array(current.original.data),new Uint8Array(legacy.operation.attentionProjection.output.data),'complete projected float32 results equal legacy including last partial row');
assert.deepEqual(current.device.buffers.slice(initial).map(b=>b.size),[current.capacity*current.D*4,current.capacity*current.D*4],'only input/output row caches are new');
assert.equal(current.calls.filter(c=>c.kind==='linear').length,3);
assert(current.calls.filter(c=>c.kind==='linear').every(c=>c.start===0&&c.totalRows===current.capacity));
const receipt=current.backbone._markFineAttentionProjectionReusable(current.state,current.duties.at(-1));
assert.equal(receipt.bytes,current.N*current.D*4);assert.equal(receipt.scratchBytes,2*current.capacity*current.D*4);
current.backbone._markFineAttentionResidualReusable(current.state,current.duties.at(-1));
for(const mutate of [x=>x.backbone._residentWorkOwner.buffers.delete(x.original),x=>x.operation.attention.attnOutBuf=x.old,
 x=>x.operation.attention.nextTileIndex--,x=>x.state.reuseAttentionQueryStorage=false,x=>x.duties[0].rowStart=1,
 x=>x.duties[0].rowEnd++,x=>x.duties[0].rangeIndex=1,x=>x.duties[0].ownerId='other']){
 const x=fixture(true);mutate(x);const before=x.device.buffers.length;
 assert.throws(()=>x.dispatch(x.duties[0]),/owned|projection|range|row|reuse|tile|owner/i);
 assert.equal(x.device.buffers.length,before,'invalid ownership/range refuses before cache allocation');assert.equal(x.calls.length,0);
}
for(const mutate of [x=>x.state.nextFineDutyIndex--,x=>x.operation.attentionProjection.nextRowStart--,
 x=>x.backbone._residentWorkOwner.buffers.delete(x.original),x=>x.operation.attentionProjection.completedPrefix=undefined]){
 const x=fixture(true);for(const d of x.duties)x.dispatch(d);
 if(mutate.toString().includes('completedPrefix')){assert.throws(()=>x.backbone._markFineAttentionResidualReusable(x.state,x.duties.at(-1)),/prefix|projection|reuse/i);continue;}
 mutate(x);assert.throws(()=>x.backbone._markFineAttentionProjectionReusable(x.state,x.duties.at(-1)),/owned|projection|prefix/i);
}
const demand=twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-attention-linear-range',rangeIndex:0},
 reuseDeadTriplaneStorage:true,reuseAttentionResidualStorage:true,reuseAttentionQueryStorage:true,reuseAttentionProjectionStorage:true,linearRowsPerDuty:128});
assert.equal(demand.workGpuBytes,2*128*1024*4+4096,'NEW matches actual complete row caches, not nonexistent full destination');
assert.equal(twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-attention-linear-range',rangeIndex:0}}).workGpuBytes,113246208+4096);
console.log('PASS complete owned attention projection row-cache reuse, exact local float32 oracle, unchanged future rows/default, range/custody refusal and actual NEW; synthetic only');
globalThis.requestAnimationFrame=callback=>queueMicrotask(()=>callback(performance.now()));
const {runCooperativeTwoStream,retireTwoStreamWork}=await import('../src/lib/cooperative_two_stream.js');
for(const failPrefix of [false,true]){
 const x=fixture(true),borrowed=createEmptyBuffer(x.device,16),marked=[];
 await retireTwoStreamWork(x.backbone);
 let last=null,fenced=null,blocked=false;
 x.device.queue.onSubmittedWorkDone=async()=>{
  if(blocked||failPrefix&&last?.kind==='fuse-attention-linear-range'&&last.rangeIndex===last.rangeCount-1){
   blocked=true;throw Error('failed actual projection row prefix');}
  fenced=last?.dutyIndex;
 };
 x.device.createCommandEncoder=()=>({...x.encoder,finish(){return {};}});
 x.backbone._dispatchAdd=(enc,dst,src,count)=>{const a=new Float32Array(dst.data),b=new Float32Array(src.data);
  for(let i=0;i<count;i++)a[i]=Math.fround(a[i]+b[i]);};
 x.backbone._dispatchLayerNorm=(enc,input,output)=>new Uint8Array(output.data).set(new Uint8Array(input.data));
 const marker=x.backbone._markFineAttentionProjectionReusable.bind(x.backbone);
 x.backbone._markFineAttentionProjectionReusable=(state,duty)=>{assert.equal(fenced,duty.dutyIndex);const r=marker(state,duty);marked.push(r);return r;};
 x.backbone.dispatchAttentionForwardDuty=(encoder,state,index)=>{
  const d=state.finePlan[index];last=d;
  if(index===0){state.currentLatent=createEmptyBuffer(x.device,16);state.currentTriplane=createEmptyBuffer(x.device,140);}
  if(d.kind==='fuse-attention-linear-range'&&d.rangeIndex===d.rangeCount-1){
   const q=createEmptyBuffer(x.device,140),input=createEmptyBuffer(x.device,128*5*4),output=createEmptyBuffer(x.device,128*5*4);
   state.activeOperation={kind:'fuse',ownerId:`block-${d.block}-fuse-out`,N_z:7,D:5,zBuf:state.currentTriplane,
    attention:{qBuf:q,attnOutBuf:q},weights:{normZ2:{}},attentionProjection:{output:q,reuseAttentionProjectionStorage:true,
     nextRangeIndex:d.rangeCount,rangeCount:d.rangeCount,nextRowStart:7,rowCache:{capacity:128,input,output}}};
  }
  if(d.kind==='fuse-residual-norm')x.backbone._dispatchFineFuseResidualNorm(encoder,state,d);
  if(d.kind==='fuse-final-residual'){state.currentTriplane=state.activeOperation.z1Buf;state.activeOperation=null;}
  if(index===state.finePlan.length-1){state.activeOperation=null;state.result={buffer:createEmptyBuffer(x.device,16)};}
  state.nextFineDutyIndex=index+1;
 };
 x.backbone.getForwardResult=state=>state.result;
 const options={device:x.device,backbone:x.backbone,N_img:1297,imageTokensBuf:borrowed,weights:{},
  dutyGranularity:'attention-tile',attentionRowsPerDuty:32,residentFFN:true,retireIntermediateBuffers:true,
  reuseDeadTriplaneStorage:true,reuseAttentionResidualStorage:true,reuseAttentionQueryStorage:true,reuseAttentionProjectionStorage:true};
 if(failPrefix){await assert.rejects(runCooperativeTwoStream(options),/prefix|unresolved/);assert.equal(marked.length,0);
  assert(x.backbone._residentWorkOwner);blocked=false;last=null;await retireTwoStreamWork(x.backbone);
 }else{const result=await runCooperativeTwoStream(options);
  assert.deepEqual(marked.map(r=>r.block),[0,1,2,3]);assert.deepEqual(result.report.adapterTelemetry.attentionProjectionReuse,marked);
  assert.equal(result.report.adapterTelemetry.declaredDutyCount,7958);result.result.buffer.destroy();
 }
 assert.equal(borrowed.destroyed,0,'borrowed device/input are never retired');assert.equal(x.device.destroyed,0);
}
console.log('PASS actual locked-Kit full7958-duty prefix marking and failure quarantine with synthetic operands; no native math/fit claim');
