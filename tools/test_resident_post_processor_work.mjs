import assert from 'node:assert/strict';
import {runCooperativePostProcessor,retirePostProcessorWork,describePostProcessorDutyDemand} from '../src/lib/cooperative_post_processor.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={UNIFORM:64,STORAGE:128,COPY_SRC:4,COPY_DST:8};
globalThis.requestAnimationFrame=cb=>queueMicrotask(()=>cb(performance.now()));
function fixture(){
  const device=fakeWeightDevice();let blocked=false,failAt=-1,encoded=0;
  device.createBuffer=({size,label})=>{
    const buffer={size,label,destroyed:0,getMappedRange(){return new ArrayBuffer(Math.min(size,64))},
      unmap(){},destroy(){this.destroyed++}};
    device.buffers.push(buffer);return buffer;
  };
  device.createBindGroup=desc=>desc;
  device.queue.onSubmittedWorkDone=async()=>{if(blocked)throw Error('prefix unresolved')};
  const trace=[];
  device.createCommandEncoder=()=>({
    beginComputePass(){return{setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}}},
    copyBufferToBuffer(src,offset,dst,target,size){trace.push(['copy',target,size])},
    finish(){if(encoded++===failAt){blocked=true;throw Error('encode failed')}return {}},
  });
  const input=device.createBuffer({size:113246208}),weights={convLayers:Array.from({length:4},
    ()=>({weight:device.createBuffer({size:4}),bias:device.createBuffer({size:4})}))};
  const handle={device};
  return{device,input,weights,handle,trace,fail(n){failAt=n},unblock(){blocked=false}};
}
const f=fixture();let before=0,after=0,groups=0;
const result=await runCooperativePostProcessor({device:f.device,triplanesBuf:f.input,weights:f.weights,
  dutyGranularity:'channel-range',residentWork:f.handle,
  onBeforeDuty(){before++},onAfterDuty(){after++},
  async withGroupWeights(group,work){groups++;return work(f.weights)},
});
assert.equal(before,703,'702 complete duties plus separate complete-output allocation must be observed');
assert.equal(after,702);
assert.equal(groups,18);
assert.deepEqual(f.trace,[['copy',0,23592960],['copy',23592960,23592960],['copy',47185920,23592960]]);
assert.equal(result.result.buffer.size,70778880);
const borrowed=new Set([f.input,...f.weights.convLayers.flatMap(x=>[x.weight,x.bias])]);
assert.ok(f.device.buffers.filter(b=>!borrowed.has(b)&&b!==result.result.buffer).every(b=>b.destroyed===1),'all owned dead stores retire once');
assert.equal(result.result.buffer.destroyed,0);
assert.ok([...borrowed].every(b=>!b.destroyed));
assert.equal(f.handle._postProcessorWorkOwner,null);
assert.equal(f.device.destroyed,0);
const bad=fixture();bad.fail(2);
await assert.rejects(runCooperativePostProcessor({device:bad.device,triplanesBuf:bad.input,weights:bad.weights,
  dutyGranularity:'channel-range',residentWork:bad.handle,onBeforeDuty(){}}),/cleanup is unresolved/);
assert.ok(bad.handle._postProcessorWorkOwner,'failed prefix keeps precise allocated inventory');
assert.ok(bad.handle._postProcessorWorkOwner.buffers.size>0);
const outstanding=[...bad.handle._postProcessorWorkOwner.buffers];
assert.ok(outstanding.every(b=>!b.destroyed),'failed prefix grants no destruction authority');
const count=bad.device.buffers.length;
await assert.rejects(runCooperativePostProcessor({device:bad.device,triplanesBuf:bad.input,weights:bad.weights,
  dutyGranularity:'channel-range',residentWork:bad.handle,onBeforeDuty(){}}),/quarantined/);
assert.equal(bad.device.buffers.length,count,'quarantine permits no new allocation');
bad.unblock();await retirePostProcessorWork(bad.handle);
assert.ok(outstanding.every(b=>b.destroyed===1));
assert.equal(bad.input.destroyed,0);
const refusal=fixture();
await assert.rejects(runCooperativePostProcessor({device:refusal.device,triplanesBuf:refusal.input,weights:refusal.weights,
  dutyGranularity:'channel-range',residentWork:refusal.handle,
  onBeforeDuty(){throw Error('fresh host refusal')}}),/fresh host refusal/);
assert.equal(refusal.device.buffers.length,9,'fresh complete-output refusal allocates no model store');
assert.equal(refusal.handle._postProcessorWorkOwner,null);
assert.equal(describePostProcessorDutyDemand({kind:'gather'}).workGpuBytes,37748756);
assert.equal(describePostProcessorDutyDemand({kind:'conv-range',rangeIndex:0,totalChannels:1024}).workGpuBytes,37748800);
assert.equal(describePostProcessorDutyDemand({kind:'conv-range',rangeIndex:1,totalChannels:1024}).workGpuBytes,64);
assert.equal(describePostProcessorDutyDemand({kind:'pixel-shuffle-copy'}).workGpuBytes,23592984);
console.log('PASS real complete postprocessor driver with private tiny metadata-only allocation fixture');

