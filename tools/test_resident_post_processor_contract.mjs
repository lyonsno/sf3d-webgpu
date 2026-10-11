import assert from 'node:assert/strict';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {runResidentPostProcessor,disposeResidentPostProcessor,selectPostProcessorPhase} from '../src/lib/resident_post_processor.js';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={UNIFORM:64,STORAGE:128,COPY_SRC:4,COPY_DST:8};
globalThis.requestAnimationFrame=cb=>queueMicrotask(()=>cb(performance.now()));
// Tiny synthetic weights and metadata-only output stores establish local custody,
// selection and full coverage only. They are not model numerics or backend fit.
const fixture=weightFixture(),etag='"post-phase-fixture"',prior=globalThis.fetch;
globalThis.fetch=async(url,init)=>{
  assert.equal(init.headers['If-Match'],etag);
  const [,a,b]=init.headers.Range.match(/^bytes=(\d+)-(\d+)$/),start=Number(a),end=Number(b);
  return new Response(fixture.bytes.slice(start,end+1),{status:206,headers:{
    ETag:etag,'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`,
    'Content-Length':String(end-start+1)}});
};
function deviceFixture(){
  const device=fakeWeightDevice();
  device.createBuffer=({size,label})=>{
    const data=new ArrayBuffer(Math.min(size,64));
    const buffer={size,label,data,destroyed:0,getMappedRange(){return data},
      unmap(){},destroy(){this.destroyed++}};
    device.buffers.push(buffer);return buffer;
  };
  device.createBindGroup=desc=>desc;
  device.createCommandEncoder=()=>({
    beginComputePass(){return{setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}}},
    copyBufferToBuffer(){},finish(){return {}},
  });
  return device;
}
const device=deviceFixture(),budget=createLoaderMemoryBudget({cpuBytes:268435456,gpuBytes:1073741824,totalBytes:1342177280});
await budget.requestOwnedDevice({requestDevice:async()=>device});
const input=device.createBuffer({size:113246208}),postProcessor={device};
const params={device,postProcessor,memoryBudget:budget,weightsUrl:'post-fixture.bin',
  expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag,triplanesBuf:input};
try{
  const names=[],duties=[];let output;
  const result=await runResidentPostProcessor({...params,
    onBeforePhase(phase){names.push(phase.tensors.map(t=>t.name))},
    onBeforeDuty(duty){duties.push(duty)},
    async withResult(actual){
      output=actual.buffer;
      assert.equal(output.size,70778880);assert.equal(output.destroyed,0);
      assert.ok(budget.snapshot().gpu.liveBytes>=input.size+output.size);
      assert.equal(actual.numPlanes,3);await Promise.resolve();assert.equal(output.destroyed,0);
      return {consumed:true};
    },
  });
  assert.deepEqual(result.value,{consumed:true});
  assert.deepEqual(result.shape,[3,40,384,384]);
  assert.equal(result.cooperative.adapterTelemetry.residentWork,true);
  assert.equal(result.cooperative.adapterTelemetry.stageDuties.length,702);
  assert.equal(result.weightPhases.length,12);
  assert.ok(result.weightPhases.every(p=>p.status==='completed-retired'));
  assert.equal(names.length,12);
  for(let index=0;index<12;index++)assert.deepEqual(names[index],
    ['weight','bias'].map(suffix=>'post_processor.upsample.'+(2*(index%4))+'.'+suffix));
  assert.equal(duties.length,703);
  assert.equal(output.destroyed,1,'output retires only after awaited consumer');
  assert.equal(input.destroyed,0);
  assert.equal(budget.snapshot().gpu.liveBytes,input.size);
  assert.equal(budget.snapshot().cpu.liveBytes,0);
  assert.equal(postProcessor._residentAdapterOwner,null);
  assert.equal(postProcessor._postProcessorWorkOwner,null);
  assert.equal(device.destroyed,0);
  const before=device.buffers.length;
  await assert.rejects(runResidentPostProcessor({...params,onBeforePhase(){},onBeforeDuty(){
    throw Error('host refuses before output')},withResult(){throw Error('must not consume')}}),/host refuses/);
  assert.equal(device.buffers.length,before);
  for(const bad of [-1,4,1.5])assert.throws(()=>selectPostProcessorPhase({postProcessor:{convLayers:[{},{},{},{}]}},bad),/exact four-layer/);
  await assert.rejects(runResidentPostProcessor({...params,triplanesBuf:{size:1},onBeforePhase(){},onBeforeDuty(){},withResult(){}}),/complete/);
  await disposeResidentPostProcessor(postProcessor);
}finally{
  globalThis.fetch=prior;input.destroy();budget.restore();
}
console.log('PASS complete postprocessor adapter: 12 real-schema phases, all702 duties, awaited output and exact recovery custody');
