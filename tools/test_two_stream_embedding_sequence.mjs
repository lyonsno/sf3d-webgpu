import assert from 'node:assert/strict';
import {TwoStreamBackbone} from '../src/lib/two_stream.js';
import {runResidentTwoStream} from '../src/lib/resident_two_stream.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
import {twoStreamPhaseDemand} from './resident_two_stream_acceptance.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64};
const fixture=weightFixture(),header=new DataView(fixture.bytes.buffer).getUint32(12,true),etag='"embedding-sequence-fixture"',prior=globalThis.fetch;
const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:2*header+4096,gpuBytes:4096});
await budget.requestOwnedDevice({requestDevice:async()=>device});
globalThis.fetch=async(url,init)=>{const [,start,end]=init.headers.Range.match(/^bytes=(\d+)-(\d+)$/);
  return new Response(fixture.bytes.slice(Number(start),Number(end)+1),{status:206,headers:{ETag:etag,'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`}});};
const phases=[];let failure;
try{
  try{await runResidentTwoStream({device,backbone:new TwoStreamBackbone(device),memoryBudget:budget,weightsUrl:'fixture.bin',
    expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag,imageTokensBuf:{size:1297*1024*4},N_img:1297,
    onBeforeDuty:async()=>{},withResult:async()=>{},onBeforePhase:async phase=>{
      phases.push(phase);if(phases.length===2){assert.equal(device.buffers.length,1,'weight exists before fresh rearrange observation');
        assert.equal(budget.snapshot().cpu.liveBytes,0,'logical range CPU custody released before rearrangement; not GC evidence');throw Error('rearrange refused');}
    }});}catch(e){failure=e;}
  assert.deepEqual(phases.map(p=>p.name),['two-stream-embedding-weights','two-stream-embedding-rearrange'],
    'weight conversion and rearrangement must be two separately observed sequential allocation cliffs');
  assert.match(failure.message,/rearrange refused/);assert.equal(phases[0].workGpuBytes,0);
  assert.deepEqual(phases[1].tensors,[]);assert.equal(phases[1].workGpuBytes,113246228);
  assert.ok(device.buffers.every(b=>b.destroyed));assert.equal(budget.snapshot().gpu.liveBytes,0);
  assert.equal(twoStreamPhaseDemand({name:phases[0].name,tensors:[{size:56623104,dtype:1}]}).requiredBytes,339738624);
  assert.equal(twoStreamPhaseDemand(phases[1]).requiredBytes,113246228);
}finally{globalThis.fetch=prior;device.destroy();budget.restore();}
console.log('sequential embedding gates retain fresh post-upload baseline and refusal cleanup');

