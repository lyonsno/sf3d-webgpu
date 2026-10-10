import assert from 'node:assert/strict';
import {TwoStreamBackbone} from '../src/lib/two_stream.js';
import {runCooperativeTwoStream,defineTwoStreamManifest,groupTwoStreamDuties,retireTwoStreamWork} from '../src/lib/cooperative_two_stream.js';
import {createTwoStreamAttentionDutyPlan,TWO_STREAM_STAGE_IDS} from '../src/lib/two_stream.js';
import {createEmptyBuffer} from '../src/lib/gpu.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
// Deterministic test host turn; no native rendering/liveness claim.
globalThis.requestAnimationFrame=callback=>queueMicrotask(()=>callback(performance.now()));
assert.equal(defineTwoStreamManifest({dutyGranularity:'attention-tile',N_img:1297,residentFFN:true}).phases[0].boundaries[0].totalItems,3350,
  'effective kit manifest must declare resident work rather than requested-only identity');
assert.deepEqual(groupTwoStreamDuties(createTwoStreamAttentionDutyPlan(1297,{residentFFN:true})).map(g=>g.stageId),TWO_STREAM_STAGE_IDS);
// Native kernels are stubbed; these tests establish allocation custody, not
// native numerical/backend conformance. Borrowed device and inputs survive.
function fixture(fail=null){
  const device=fakeWeightDevice();let drains=0,blocked=false;
  device.queue.onSubmittedWorkDone=async()=>{drains++;if(blocked)throw Error('unresolved prefix');};
  const backbone=new TwoStreamBackbone(device),input=createEmptyBuffer(device,16);
  const trace=[],allocated=[];
  device.createCommandEncoder=()=>({finish(){return {};}});
  backbone.dispatchAttentionForwardDuty=(encoder,state,index)=>{
    trace.push(index);
    const temp=createEmptyBuffer(device,16),output=createEmptyBuffer(device,16);allocated.push(temp,output);
    if(index===1)assert.ok(allocated[0].destroyed,'setup temporary retired before next duty');
    if(index===state.finePlan.length-1)state.result={buffer:output};
    state.currentTriplane=output;state.nextFineDutyIndex=index+1;
    if(fail!==null&&index===1){blocked=fail==='drain';throw fail==='falsy'?null:Error('encode failed');}
  };
  backbone.getForwardResult=state=>state.result;
  return {device,backbone,input,allocated,trace,unblock(){blocked=false;},get drains(){return drains;}};
}
// Actual kit execution and full declared plan, tiny deterministic buffers.
const f=fixture();
const options={device:f.device,backbone:f.backbone,imageTokensBuf:f.input,N_img:1,weights:{},
  dutyGranularity:'attention-tile',residentFFN:true,retireIntermediateBuffers:true};
const actual=await runCooperativeTwoStream(options);
assert.equal(actual.report.adapterTelemetry.residentFFN,true);
assert.equal(actual.report.adapterTelemetry.declaredDutyCount,createTwoStreamAttentionDutyPlan(1,{residentFFN:true}).length);
assert.equal(f.trace.length,actual.report.adapterTelemetry.declaredDutyCount);
assert.ok(f.allocated.every(b=>b===actual.result.buffer?!b.destroyed:b.destroyed));
assert.equal(f.backbone._residentWorkOwner,null);
assert.ok(!f.input.destroyed,'borrowed input remains owned by its caller');
assert.equal(f.device.destroyed,0);
for(const fail of ['falsy','drain']){
  const x=fixture(fail);
  let caught=false,error;
  try{await runCooperativeTwoStream({...options,device:x.device,backbone:x.backbone,imageTokensBuf:x.input});}
  catch(e){caught=true;error=e;}
  assert.ok(caught);
  if(fail==='falsy'){
    assert.equal(error.message,'null','installed kit decorates the exact falsy model failure');
    assert.ok(x.allocated.every(b=>b.destroyed));
  }
  else{
    assert.ok(x.backbone._residentWorkOwner,'unresolved allocations remain recoverable');
    assert.ok(x.allocated.slice(-2).every(b=>!b.destroyed),'failed fence does not authorize destruction');
    await assert.rejects(runCooperativeTwoStream({...options,device:x.device,backbone:x.backbone}),/quarantined/);
    x.unblock();await retireTwoStreamWork(x.backbone);
    assert.equal(x.backbone._residentWorkOwner,null);
    assert.ok(x.allocated.every(b=>b.destroyed));
  }
  assert.ok(!x.input.destroyed);assert.equal(x.device.destroyed,0);
}
const drift=fixture();drift.backbone.createAttentionForwardState=()=>({finePlan:[]});
await assert.rejects(runCooperativeTwoStream({...options,device:drift.device,backbone:drift.backbone}),/plan/);
console.log('resident two-stream effective plan and private allocation custody contract');
