import assert from 'node:assert/strict';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
const subject=await import('../src/lib/resident_clip.js').catch(error=>{
  if(error.code==='ERR_MODULE_NOT_FOUND'&&error.url===new URL('../src/lib/resident_clip.js',import.meta.url).href)return {};
  throw error;
});
assert.equal(typeof subject.runResidentMaterials,'function',
  'actual complete GPU visual transformer and both CPU heads need owned computation-bound phases');
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:16,MAP_READ:32};
globalThis.GPUMapMode={READ:1};
// A virtual complete table/range server: real shapes without allocating a
// synthetic whole CLIP file. All payloads are zeros, no numerical claim.
const small=weightFixture({fullMaterialNames:true}),headerBytes=new DataView(small.bytes.buffer).getUint32(12,true);
const header=small.bytes.slice(0,headerBytes),view=new DataView(header.buffer),tensors=[];
let offset=0;
for(let i=0;i<view.getUint32(8,true);i++){
  const e=16+i*160,name=new TextDecoder().decode(header.subarray(e,e+128)).split('\0')[0];let shape=[1];
  if(name==='image_estimator.model.visual.conv1.weight')shape=[768,3072];
  else if(name==='image_estimator.model.visual.class_embedding')shape=[768];
  else if(name==='image_estimator.model.visual.positional_embedding')shape=[50,768];
  else if(name==='image_estimator.model.visual.proj')shape=[512,768];
  else if(name.startsWith('image_estimator.heads.')){
    const out=/\.[12]\.2\./.test(name)?1:512;shape=name.endsWith('.bias')?[out]:[512,out];
  }else if(name.startsWith('image_estimator.model.visual.')){
    if(name.includes('ln_'))shape=[768];
    else if(name.includes('in_proj_weight'))shape=[2304,768];
    else if(name.includes('in_proj_bias'))shape=[2304];
    else if(name.includes('out_proj'))shape=name.endsWith('.weight')?[768,768]:[768];
    else if(name.includes('c_fc'))shape=name.endsWith('.weight')?[768,3072]:[3072];
    else if(name.includes('c_proj'))shape=name.endsWith('.weight')?[3072,768]:[768];
  }
  const size=shape.reduce((n,d)=>n*d,4);view.setUint32(e+132,shape.length,true);
  for(let a=0;a<4;a++)view.setUint32(e+136+a*4,shape[a]??0,true);
  view.setUint32(e+152,offset,true);view.setUint32(e+156,size,true);
  tensors.push({name,start:headerBytes+offset,size});offset+=size;
}
const total=headerBytes+offset,etag='"synthetic-complete-CLIP"',originalFetch=globalThis.fetch,reads=[];
let wrongPayload=false,poisonName=null,refusePrefix=false,refuseDestroy=false,refuseUniform=false,badReadback=false;
globalThis.fetch=async(url,init)=>{
  if(String(url).endsWith('.wgsl'))return new Response('@compute @workgroup_size(1) fn main() {}',{headers:{'content-type':'text/wgsl'}});
  assert.equal(init.headers['If-Match'],etag);
  const [,a,b]=init.headers.Range.match(/^bytes=(\d+)-(\d+)$/),start=Number(a),end=Number(b);
  reads.push({start,end});const payload=start<headerBytes?header.slice(start,end+1):new Uint8Array(end-start+1);
  if(poisonName&&tensors.some(t=>t.name===poisonName&&t.start===start))new DataView(payload.buffer).setFloat32(0,NaN,true);
  return new Response(payload,{status:206,headers:{ETag:wrongPayload&&start>=headerBytes?'"foreign-source"':etag,
    'Content-Range':`bytes ${start}-${end}/${total}`,'Content-Length':String(end-start+1)}});
};
const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:64*1024*1024,gpuBytes:64*1024*1024});
device.createBindGroup=d=>d;
const create=device.createBuffer;
device.createBuffer=function(desc){const b=create.call(this,desc),destroy=b.destroy;b.mapAsync=async()=>{
  if(badReadback&&(desc.usage&GPUBufferUsage.MAP_READ))new Float32Array(b.data)[0]=NaN;
};
  b.destroy=function(){if(refuseDestroy&&b.label==='clip-resident-input')throw Error('owned destroy refused');
    destroy.call(this);this.data=null;};return b;};
device.queue.writeBuffer=(b,o,d)=>{
  if(refuseUniform&&b.size===16)throw Error('owned uniform write refused');
  new Uint8Array(b.data,o,d.byteLength).set(new Uint8Array(d.buffer,d.byteOffset,d.byteLength));
};
device.queue.onSubmittedWorkDone=async()=>{if(refusePrefix)throw Error('actual queue prefix unavailable');};
device.createCommandEncoder=()=>{const copies=[];return {beginComputePass(){return {setPipeline(){},setBindGroup(){},dispatchWorkgroups(){},end(){}};},
  copyBufferToBuffer(...a){copies.push(a);},finish(){return {copies};}};};
device.queue.submit=commands=>{for(const command of commands)for(const [s,so,d,to,n]of command.copies)
  new Uint8Array(d.data,to,n).set(new Uint8Array(s.data,so,n));};
await budget.requestOwnedDevice({async requestDevice(){return device;}});
const handle={device},phases=[],duties=[];
try{
  const result=await subject.runResidentMaterials({device,handle,memoryBudget:budget,
    weightsUrl:'synthetic.bin',expectedWeightBytes:total,expectedSourceETag:etag,rgba:new Uint8Array(512*512*4),
    onBeforePhase:async p=>phases.push(p),onBeforeDuty:async p=>duties.push(p)});
  assert.equal(result.visualBackend,'webgpu-complete-12-blocks');assert.equal(result.tokens,50);
  assert.equal(result.duties.length,14);assert.deepEqual(result.duties.map(d=>d.name),
    ['clip-pre',...Array.from({length:12},(_,i)=>'clip-block-'+i),'clip-post']);
  assert.equal(result.weightPhases.length,18,'prep,pre,12blocks,post,projection,two actual heads');
  assert.deepEqual(phases.slice(0,2).map(p=>[p.name,p.rangeCpuBytes]),
    [['weight-header-prefix',32],['weight-header',2*headerBytes]],'actual CLIP source callback is forwarded before both source ranges');
  assert.ok(result.weightPhases.every(p=>p.status==='completed-retired'));
  assert.equal(result.roughness,0.5);assert.equal(result.metallic,0.5,'real beta-mode heads on zero fixture, not defaults');
  assert.equal(phases.filter(p=>p.storageKind==='cpu-fp32').length,4);
  assert.equal(duties.at(-1).name,'clip-complete-token-readback');
  assert.equal(handle._residentClipOwner,null);assert.equal(budget.snapshot().cpu.liveBytes,0);assert.equal(budget.snapshot().gpu.liveBytes,0);
  assert.ok(device.buffers.every(b=>b.destroyed===1),'exact work and weights retire once');
  const count=device.buffers.length,readCount=reads.length;
  await assert.rejects(()=>subject.runResidentMaterials({device,handle,memoryBudget:budget,
    weightsUrl:'synthetic.bin',expectedWeightBytes:total,expectedSourceETag:etag,rgba:new Uint8Array(512*512*4),
    onBeforePhase:async()=>{throw Error('fresh admission refused');},onBeforeDuty:async()=>{}}),/admission refused/);
  assert.equal(device.buffers.length,count,'fresh refusal before any CLIP payload/backing');
  assert.equal(reads.length,readCount,'first fresh refusal precedes every source fetch including prefix/header');
  const run=extra=>subject.runResidentMaterials({device,handle,memoryBudget:budget,
    weightsUrl:'synthetic.bin',expectedWeightBytes:total,expectedSourceETag:etag,rgba:new Uint8Array(512*512*4),
    onBeforePhase:async()=>{},onBeforeDuty:async()=>{},...extra});
  await assert.rejects(run({onBeforePhase:async()=>{throw 0;}}),e=>e===0,'falsy guard refusal cannot become success');
  wrongPayload=true;await assert.rejects(run({}),/source identity mismatch/);wrongPayload=false;
  assert.equal(handle._residentClipOwner,null);assert.equal(budget.snapshot().cpu.liveBytes,0);
  refuseUniform=true;const uniformStart=device.buffers.length;
  await assert.rejects(run({}),/uniform write refused/);refuseUniform=false;
  assert.ok(device.buffers.slice(uniformStart).some(b=>b.size===16&&b.destroyed===1),
    'actual direct uniform is owned before write failure');
  assert.equal(handle._residentClipOwner,null);assert.equal(budget.snapshot().gpu.liveBytes,0);
  await assert.rejects(run({onBeforeDuty:async d=>{if(d.name==='clip-pre')refusePrefix=true;}}));
  assert.ok(handle._residentClipOwner,'failed prefix preserves reachable work/source/CPU ownership');
  assert.ok(budget.snapshot().gpu.liveBytes>0);assert.ok(budget.snapshot().cpu.liveBytes>0);
  await assert.rejects(run({}),/nonquarantined/);
  refusePrefix=false;await subject.disposeResidentMaterials(handle);
  assert.equal(budget.snapshot().gpu.liveBytes,0);assert.equal(budget.snapshot().cpu.liveBytes,0);
  refuseDestroy=true;await assert.rejects(run({}),e=>e instanceof AggregateError&&
    e.message.includes('cleanup unresolved')&&e.errors.some(x=>x.message?.includes('retirement unresolved')));
  assert.equal(handle._residentClipOwner.allocations.length,1,'only exact failed destroy remains');
  assert.equal(handle._residentClipOwner.allocations[0].buffer.label,'clip-resident-input');
  assert.ok(budget.snapshot().gpu.liveBytes>0);await assert.rejects(run({}),/nonquarantined/);
  refuseDestroy=false;await subject.disposeResidentMaterials(handle);
  badReadback=true;await assert.rejects(run({}),/finite|NaN/);badReadback=false;
  poisonName='image_estimator.heads.metallic.2.2.bias';
  await assert.rejects(run({}),/invalid actual metallic/);poisonName=null;
  assert.equal(handle._residentClipOwner,null);assert.equal(budget.snapshot().gpu.liveBytes,0);
  assert.equal(budget.snapshot().cpu.liveBytes,0);
  assert.ok(device.buffers.every(b=>b.destroyed===1),'recoveries never double-destroy owned resources');
  const names=[];
  assert.throws(()=>subject.selectResidentClipPhase({reference:n=>({name:n}),describe:s=>{
    names.push(...Object.keys(s));return [{name:'image_estimator.heads.roughness.1.2.weight',shape:[1,512]}];
  }},'roughness'),/canonical material head shape mismatch/);
  assert.equal(names.length,14,'all seven actual head pairs selected before shape validation');
}finally{refuseDestroy=false;refusePrefix=false;globalThis.fetch=originalFetch;
  await subject.disposeResidentMaterials(handle);device.destroy();budget.restore();}
console.log('PASS complete canonical CLIP phase order, all12 GPU blocks/both real heads, refusal and exact retirement; synthetic policy only');
