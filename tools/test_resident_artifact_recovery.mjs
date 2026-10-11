import assert from 'node:assert/strict';
import fs from 'node:fs';
import {weightFixture,fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import * as artifact from '../src/lib/resident_artifact.js';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:16,MAP_READ:32};
const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:256*1024*1024,gpuBytes:1024*1024*1024,totalBytes:1280*1024*1024});
await budget.requestOwnedDevice({async requestDevice(){return device;}});
const fixture=weightFixture({fullDecoder:true}),etag='"synthetic-decoder"',originalFetch=globalThis.fetch;
let refusePrefix=false;device.queue.onSubmittedWorkDone=async()=>{if(refusePrefix)throw Error('owned prefix unavailable');};
globalThis.fetch=async(url,init)=>{
  if(String(url).startsWith('/tets/')){const bytes=fs.readFileSync(new URL('../public'+url,import.meta.url));return new Response(bytes,{headers:{'Content-Length':String(bytes.length)}});}
  const [,a,b]=init.headers.Range.match(/^bytes=(\d+)-(\d+)$/),start=Number(a),end=Number(b);
  return new Response(fixture.bytes.slice(start,end+1),{status:206,headers:{ETag:etag,
    'Content-Range':`bytes ${start}-${end}/${fixture.bytes.length}`,'Content-Length':String(end-start+1)}});
};
const handle={device},borrowed={size:70778880,destroy(){assert.fail('borrowed planes destroyed');}};
try{
  await assert.rejects(()=>artifact.runResidentArtifact({device,handle,memoryBudget:budget,triplanesBuf:borrowed,
    conditionRgba:new Uint8Array(512*512*4),weightsUrl:'synthetic.bin',expectedWeightBytes:fixture.bytes.length,expectedSourceETag:etag,
    onBeforePhase:async()=>{},onBeforeDuty:async()=>{refusePrefix=true;throw Error('range refused');},withResult:async()=>assert.fail('unsettled output consumed')}));
  assert.ok(handle._residentArtifactOwner,'failed lower decoder/source cleanup must remain reachable by the complete consumer');
  assert.ok(handle._residentArtifactOwner.decoder._residentQueryWorkOwner);
  assert.ok(budget.snapshot().cpu.liveBytes>0);
  assert.equal(typeof artifact.disposeResidentArtifact,'function');
  await assert.rejects(()=>artifact.runResidentArtifact({device,handle,memoryBudget:budget,triplanesBuf:borrowed,
    conditionRgba:new Uint8Array(512*512*4),onBeforePhase:async()=>{},onBeforeDuty:async()=>{},withResult:async()=>{}}),/nonquarantined/);
  refusePrefix=false;await artifact.disposeResidentArtifact(handle);
  assert.equal(handle._residentArtifactOwner,null);assert.equal(budget.snapshot().cpu.liveBytes,0);assert.equal(budget.snapshot().gpu.liveBytes,0);
  assert.equal(device.destroyed,0);assert.ok(device.buffers.every(b=>b.destroyed===1));
}finally{refusePrefix=false;globalThis.fetch=originalFetch;
  if(budget.snapshot().cpu.liveBytes===0&&budget.snapshot().gpu.liveBytes===0)budget.restore();}
console.log('PASS complete artifact failure preserves exact decoder/source recovery and retry disposal, no borrowed device/planes destruction; canonical tet assets, synthetic weights/no shader numerics');
