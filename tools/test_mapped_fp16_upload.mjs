import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import * as gpu from '../src/lib/gpu.js';
import {createWeightPhaseSource} from '../src/lib/weights.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
const source=fs.readFileSync(new URL('../src/lib/weights.js',import.meta.url),'utf8');
const allocations=[],leases=[],buffers=[];
const Float32= new Proxy(Float32Array,{construct(target,args){if(typeof args[0]==='number')allocations.push(args[0]*4);return Reflect.construct(target,args);}});
const extract=vm.compileFunction(source.slice(source.indexOf('function fp16ToFp32('),source.indexOf('function extractTensorCPU('))+'\nreturn extractTensor;',
  ['createStorageBuffer','createMappedStorageBuffer','extractBytes'],{parsingContext:vm.createContext({Float32Array:Float32})})
  (gpu.createStorageBuffer,gpu.createMappedStorageBuffer,(buffer,offset,size)=>new Uint8Array(buffer,offset,size));
const device={createBuffer(desc){const b={...desc,data:new ArrayBuffer(desc.size),destroyed:0,unmapped:0,
  getMappedRange(){return b.data;},unmap(){b.unmapped++;},destroy(){b.destroyed++;}};buffers.push(b);return b;}};
const half=new Uint16Array(65536);for(let i=0;i<half.length;i++)half[i]=i;
const ownedBuffers=new Set(),context={ownedBuffers,ownedCpu:new Set(),budget:{reserveCpu(size,label){
  const lease={size,label,released:false,release(){lease.released=true;}};leases.push(lease);return lease;}}};
const result=extract(device,half.buffer,{dtype:1,offset:0,size:half.byteLength},context);
assert.deepEqual(allocations,[],'FP16 GPU upload must not allocate a separate full FP32 CPU backing');
assert.equal(result.size,262144);assert.equal(result.unmapped,1);
assert.equal(leases.length,1,'unchanged conversion CPU allowance still protects mapped upload');
assert.equal(leases[0].size,262144);assert.equal(leases[0].released,true);
const values=new Float32Array(result.data);
for(let bits=0;bits<65536;bits++){
  const sign=bits&32768?-1:1,e=(bits>>>10)&31,m=bits&1023;
  const expected=e===31?(m?NaN:sign*Infinity):e===0?sign*m*2**-24:sign*(1024+m)*2**(e-25);
  if(Number.isNaN(expected))assert(Number.isNaN(values[bits]));
  else assert(Object.is(values[bits],expected),'complete IEEE half conversion differs at '+bits);
}
assert.equal(ownedBuffers.has(result),true,'range owner acquires output before initialization');
assert.equal(buffers.length,1,'one complete mapped GPU destination only');
console.log('PASS all65536 half encodings including signed zeros/subnormals/NaN/inf, direct mapped destination, same CPU reservation, no separate full FP32 backing');

for(const phase of ['map','initialize','unmap'])for(const failDestroy of [false,true]){
  const owned=new Set(),error=Error('injected '+phase),retirement=Error('injected destroy');
  const b={getMappedRange(){if(phase==='map')throw error;return new ArrayBuffer(16);},
    unmap(){if(phase==='unmap')throw error;},destroy(){if(failDestroy)throw retirement;}};
  const d={createBuffer(){return b;}};
  let caught;try{gpu.createMappedStorageBuffer(d,16,mapped=>{
    assert.equal(owned.has(b),true,'mapping/initialization already has exact recovery custody');
    if(phase==='initialize')throw error;new Uint8Array(mapped).fill(1);
  },{ownedBuffers:owned});}catch(e){caught=e;}
  assert(caught);assert.equal(owned.has(b),failDestroy,'failed destroy retains exact buffer, successful destroy removes it');
  if(failDestroy){assert.deepEqual(caught.errors,[error,retirement]);assert.equal(caught.unresolvedGpuBuffer,b);}
  else assert.equal(caught,error);
}
for(const size of [-1,0,1,NaN])assert.throws(()=>gpu.createMappedStorageBuffer({
  createBuffer(){throw Error('must refuse before allocation');}},size,()=>{}),/aligned/);
console.log('PASS map/initializer/unmap faults retire exact owned destination or preserve unresolved inventory and both causes; no GPU/native evidence');

const fixture=weightFixture({fp16Names:new Set(['tokenizer.embeddings']),
  tensorShapes:new Map([['tokenizer.embeddings',[2]]])}),etag='"mapped-fixture"';
const originalFetch=globalThis.fetch;
globalThis.fetch=async(url,init)=>{
  assert.equal(init.headers['If-Match'],etag);
  const [start,end]=init.headers.Range.slice(6).split('-').map(Number);
  return new Response(fixture.bytes.slice(start,end+1),{status:206,headers:{
    ETag:etag,'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`,'Content-Length':String(end-start+1)}});
};
const phaseDevice=fakeWeightDevice(),create=phaseDevice.createBuffer.bind(phaseDevice);
let failCleanup=true;
phaseDevice.createBuffer=desc=>{
  const b=create(desc),destroy=b.destroy.bind(b);
  b.unmap=()=>{throw Error('mapped upload unmap failed');};
  b.destroy=()=>{if(failCleanup)throw Error('mapped destination destroy unresolved');destroy();};
  return b;
};
const budget=createLoaderMemoryBudget({cpuBytes:1048576,gpuBytes:1048576});budget.bindOwnedDevice(phaseDevice);
let phaseSource;
try{
  phaseSource=await createWeightPhaseSource(phaseDevice,'fixture',{memoryBudget:budget,
    expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag});
  let computed=false;
  await assert.rejects(phaseSource.withWeights(phaseSource.template.tokenizer,()=>{computed=true;}),/cleanup.*unresolved/);
  assert.equal(computed,false);assert.equal(phaseSource.phases.at(-1).status,'cleanup-unresolved');
  assert.equal(budget.snapshot().gpu.liveBytes,8,'unresolved exact destination stays charged');
  await assert.rejects(phaseSource.withWeights(phaseSource.template.tokenizer,()=>{}),/quarantined/);
  assert.equal(phaseDevice.buffers.length,1,'quarantine creates no replacement destination');
  failCleanup=false;await phaseSource.dispose();
  assert.equal(budget.snapshot().gpu.liveBytes,0);assert.equal(budget.snapshot().cpu.liveBytes,0);
  assert.equal(phaseDevice.buffers[0].destroyed,1);
}finally{failCleanup=false;await phaseSource?.dispose();phaseDevice.destroy();budget.restore();globalThis.fetch=originalFetch;}
console.log('PASS actual authenticated range owner quarantines failed mapped destination, preserves exact budget and recovers before another phase; synthetic source only');
