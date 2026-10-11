import assert from 'node:assert/strict';
import {TriplaneDecoder} from '../src/lib/triplane_decoder.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {weightFixture} from './fixtures/weight_resource_fixture.mjs';
const subject=await import('../src/lib/resident_decoder.js').catch(error=>{
  if(error.code==='ERR_MODULE_NOT_FOUND'&&error.url===new URL('../src/lib/resident_decoder.js',import.meta.url).href)return {};
  throw error;
});
assert.equal(typeof subject.runResidentDecoderQueries,'function',
  'every complete geometry point needs an observed owned decoder range, not full-grid transient allocation');
assert.equal(typeof subject.selectDecoderHeads,'function','complete geometry and texture select authentic phase leaves');
assert.equal(typeof subject.runResidentDecoder,'function','canonical decoder source is loaded only for its complete query phase');
globalThis.GPUBufferUsage={UNIFORM:1,COPY_DST:2,STORAGE:4,COPY_SRC:8,MAP_READ:16};
globalThis.GPUMapMode={READ:1};
const buffers=[],dispatches=[];let fences=0,refuseFence=false,refuseRetirement=false,badReadback=false;
const device={createBuffer({size,label=''}){const data=new ArrayBuffer(size),buffer={size,label,data,retired:0,
  getMappedRange(){return data;},unmap(){},async mapAsync(){if(badReadback&&label==='resident-decoder-readback')new Float32Array(data)[0]=NaN;},
  destroy(){if(refuseRetirement&&label==='resident-decoder-input')throw Error('owned retirement refused');this.retired++;}};
  buffers.push(buffer);return buffer;},destroy(){throw Error('borrowed device destroyed');},
  createComputePipeline(){return {getBindGroupLayout(){return {};}};},createShaderModule(){return {};},
  createBindGroup(d){return d;},createCommandEncoder(){const copies=[];return {
    beginComputePass(){return {setPipeline(){},setBindGroup(g,bg){const p=bg.entries[0].resource.buffer;
      dispatches.push(new Uint32Array(p.data).slice());},dispatchWorkgroups(){},end(){}};},
    copyBufferToBuffer(...args){copies.push(args);},finish(){return {copies};}};},
  queue:{writeBuffer(buffer,offset,data,dataOffset=0,size=data.byteLength-dataOffset){
    new Uint8Array(buffer.data,offset,size).set(new Uint8Array(data.buffer??data,(data.byteOffset??0)+dataOffset,size));},
    submit(commands){for(const command of commands)for(const [s,so,d,doff,n] of command.copies)
      new Uint8Array(d.data,doff,n).set(new Uint8Array(s.data,so,n));},
    async onSubmittedWorkDone(){fences++;if(refuseFence)throw Error('prefix unavailable');}}};
const budget=createLoaderMemoryBudget({cpuBytes:1000000,gpuBytes:1000000});
await budget.requestOwnedDevice({async requestDevice(){return device;}});
const decoder=new TriplaneDecoder(device);decoder.init();
const borrowed={size:70778880,destroy(){throw Error('borrowed triplanes destroyed');}};
const layer=()=>({weight:{size:4},bias:{size:4}});
const weights={heads:{density:Array.from({length:3},layer),vertex_offset:Array.from({length:3},layer)}};
const positions=new Float32Array(17*3),observed=[],phases=[];
let consuming=false;
const result=await subject.runResidentDecoderQueries({device,decoder,memoryBudget:budget,positions,
  triplanesBuf:borrowed,weights,heads:['density','vertex_offset'],batchPoints:8,
  onBeforePhase:async p=>phases.push(p),onBeforeDuty:async d=>observed.push({...d,allocations:buffers.length}),
  async withResult(output){consuming=true;assert.equal(output.density.length,17);assert.equal(output.vertex_offset.length,51);
    assert.ok(buffers.some(b=>!b.retired),'consumer executes before owned scratch disposal');return 'whole-grid';}});
assert.equal(result.value,'whole-grid');assert.equal(consuming,true);
assert.deepEqual(observed.map(d=>[d.start,d.end]),[[0,8],[8,16],[16,17]],'all points including tail, no output cap');
assert.equal(result.ranges.length,3);assert.ok(fences>=3);
assert.ok(buffers.reduce((n,b)=>n+b.size,0)<=result.demand.workGpuBytes,
  'actual complete dispatcher allocation inventory fits the stated prospective new GPU demand');
assert.equal(result.slotInventory.reduce((n,s)=>n+s.bytes,0),
  result.demand.components.decoderPerPointBytes*8+result.demand.components.decoderBaseBytes,
  'actual geometry slots match source-derived graph, not the bake-only arena');
assert.equal(phases[0].rangeCpuBytes,17*16,'complete result reservation, not batch-sized output');
assert.equal(observed[0].allocations,0,'first range guard precedes every backing allocator');
assert.ok(buffers.every(b=>b.retired===1),'all exact owned buffers retire once');
assert.equal(decoder._slotProvider,null);assert.equal(decoder._uniformCache.size,0);
assert.equal(budget.snapshot().cpu.liveBytes,0);assert.equal(budget.snapshot().gpu.liveBytes,0);
const before=buffers.length;
await assert.rejects(subject.runResidentDecoderQueries({device,decoder,memoryBudget:budget,positions,
  triplanesBuf:borrowed,weights,heads:['density','vertex_offset'],batchPoints:8,
  onBeforePhase:async()=>{throw Error('fresh guard refused');},onBeforeDuty:async()=>{},withResult:async()=>{}}),/guard refused/);
assert.equal(buffers.length,before,'refusal precedes allocator');
refuseFence=true;
await assert.rejects(subject.runResidentDecoderQueries({device,decoder,memoryBudget:budget,positions,
  triplanesBuf:borrowed,weights,heads:['density','vertex_offset'],batchPoints:8,
  onBeforePhase:async()=>{},onBeforeDuty:async()=>{},withResult:async()=>{assert.fail('cannot consume before prefix');}}));
assert.ok(decoder._residentQueryWorkOwner,'failed prefix preserves reachable exact owner');
assert.ok(buffers.slice(before).some(b=>!b.retired),'no retirement credit on failed prefix');
refuseFence=false;await subject.disposeResidentDecoderQueries(decoder);
assert.ok(buffers.every(b=>b.retired===1));assert.equal(decoder._residentQueryWorkOwner,null);
const bakeWeights={heads:{features:Array.from({length:4},layer),perturb_normal:Array.from({length:4},layer)}};
const bakeBefore=buffers.length;
const bake=await subject.runResidentDecoderQueries({device,decoder,memoryBudget:budget,positions,
  triplanesBuf:borrowed,weights:bakeWeights,heads:['features','perturb_normal'],batchPoints:8,
  onBeforePhase:async()=>{},onBeforeDuty:async()=>{},withResult:async o=>{
    assert.equal(o.features.length,51);assert.equal(o.perturb_normal.length,51);return 'all-texels';}});
assert.ok(buffers.slice(bakeBefore).reduce((n,b)=>n+b.size,0)<=bake.demand.workGpuBytes);
assert.equal(bake.slotInventory.reduce((n,s)=>n+s.bytes,0),
  bake.demand.components.decoderPerPointBytes*8+bake.demand.components.decoderBaseBytes);
assert.equal(bake.value,'all-texels');assert.ok(buffers.every(b=>b.retired===1));
const fixture=weightFixture({fullDecoder:true}),header=new DataView(fixture.bytes.buffer).getUint32(12,true);
const etag='"synthetic-complete-decoder-phase"',originalFetch=globalThis.fetch,reads=[];
globalThis.fetch=async(url,init)=>{
  assert.equal(init.headers['If-Match'],etag);
  const [,a,b]=init.headers.Range.match(/^bytes=(\d+)-(\d+)$/),start=Number(a),end=Number(b);
  reads.push({start,end});return new Response(fixture.bytes.slice(start,end+1),{status:206,headers:{ETag:etag,
    'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`,'Content-Length':String(end-start+1)}});
};
try{
  const weightObservations=[];
  const canonical=await subject.runResidentDecoder({device,decoder,memoryBudget:budget,positions,
    triplanesBuf:borrowed,weightsUrl:'synthetic.bin',expectedWeightBytes:fixture.bytes.length,
    expectedSourceETag:etag,heads:['density','vertex_offset'],batchPoints:8,
    onBeforePhase:async p=>weightObservations.push(p),onBeforeDuty:async()=>{},withResult:async o=>{
      assert.equal(o.density.length,17);assert.equal(o.vertex_offset.length,51);return 'canonical-source-phase';}});
  assert.equal(canonical.value,'canonical-source-phase');
  assert.equal(weightObservations[0].tensors.length,12,'all three geometry layers/pairs, no bake-only schema');
  assert.equal(reads.filter(r=>r.start>=header).length,12,'only selected authentic full heads loaded');
  assert.equal(canonical.weightPhases.length,1);assert.equal(canonical.weightPhases[0].status,'completed-retired');
  assert.equal(decoder._residentDecoderAdapterOwner,null);assert.equal(budget.snapshot().gpu.liveBytes,0);
  assert.equal(budget.snapshot().cpu.liveBytes,0);
  assert.throws(()=>subject.selectDecoderHeads({decoder:weights},['density','density']),/selected decoder/);
  assert.throws(()=>subject.selectDecoderHeads({decoder:{heads:{density:[layer()]}}},['density']),/selected decoder/);
}finally{globalThis.fetch=originalFetch;}
const again=extra=>subject.runResidentDecoderQueries({device,decoder,memoryBudget:budget,positions,
  triplanesBuf:borrowed,weights,heads:['density','vertex_offset'],batchPoints:8,
  onBeforePhase:async()=>{},onBeforeDuty:async()=>{},withResult:async()=>{},...extra});
await assert.rejects(again({withResult:async()=>{throw 0;}}),e=>e===0,
  'falsy consumer rejection does not become success');
badReadback=true;
await assert.rejects(again({withResult:async()=>assert.fail('nonfinite output cannot reach consumer')}),/nonfinite/);
badReadback=false;assert.equal(decoder._residentQueryWorkOwner,null);
refuseRetirement=true;
await assert.rejects(again({}),/retirement unresolved/);
assert.ok(decoder._residentQueryWorkOwner,'destruction refusal preserves unresolved owner');
assert.equal(decoder._residentQueryWorkOwner.allocations.length,1,'only failed owned retirement remains');
assert.ok(budget.snapshot().gpu.liveBytes>0,'failed destroy keeps exact charge');
await assert.rejects(again({}),/nonquarantined/);
refuseRetirement=false;await subject.disposeResidentDecoderQueries(decoder);
assert.equal(budget.snapshot().gpu.liveBytes,0);assert.equal(budget.snapshot().cpu.liveBytes,0);
assert.ok(buffers.every(b=>b.retired===1),'successful recoveries never retire an exact buffer twice');
console.log('PASS actual decoder complete ranges, uniform/slot custody, refusal and prefix recovery; synthetic policy only');
