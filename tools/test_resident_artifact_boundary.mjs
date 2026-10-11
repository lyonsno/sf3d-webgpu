import assert from 'node:assert/strict';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
const subject=await import('../src/lib/resident_artifact.js').catch(e=>{if(e.code==='ERR_MODULE_NOT_FOUND'&&e.url===new URL('../src/lib/resident_artifact.js',import.meta.url).href)return {};throw e;});
assert.equal(typeof subject.runResidentArtifact,'function','actual complete geometry/material/texture/GLB needs an awaited owned consumer');
const device=fakeWeightDevice(),budget=createLoaderMemoryBudget({cpuBytes:256*1024*1024,gpuBytes:1024*1024*1024,totalBytes:1280*1024*1024});
await budget.requestOwnedDevice({async requestDevice(){return device;}});
const originalFetch=globalThis.fetch;let reads=0,consumed=false;
globalThis.fetch=async()=>{reads++;throw Error('should not fetch after refusal');};
try {
  const historical=budget.reserveCpu(1,'event-schema-witness');historical.release();
  assert.equal(budget.events.filter(e=>['reserved','retirement-requested'].includes(e.kind)).length,2,
    'reservation-history witness must select the actual authority-bearing event schema');
  const beforeEvents=budget.events.length;
  const activeHandle={device};
  await assert.rejects(()=>subject.runResidentArtifact({device,handle:activeHandle,memoryBudget:budget,triplanesBuf:{size:70778880},
    conditionRgba:new Uint8Array(512*512*4),onBeforePhase:async()=>{
      await assert.rejects(()=>subject.disposeResidentArtifact(activeHandle),/cannot dispose an active/);throw 0;
    },onBeforeDuty:async()=>{},withResult:async()=>{consumed=true;}}),e=>e===0);
  assert.equal(activeHandle._residentArtifactOwner,null);
  assert.equal(reads,0);assert.equal(device.buffers.length,0);assert.equal(consumed,false);
  assert.equal(budget.snapshot().cpu.liveBytes,0);
  assert.deepEqual(budget.events.slice(beforeEvents),[]);
  globalThis.fetch=async()=>{reads++;return new Response(new Uint8Array(16),{headers:{'Content-Length':'16'}});};
  const run=()=>subject.runResidentArtifact({device,handle:{device},memoryBudget:budget,triplanesBuf:{size:70778880},
    conditionRgba:new Uint8Array(512*512*4),onBeforePhase:async()=>{},onBeforeDuty:async()=>{},withResult:async()=>{consumed=true;}});
  await assert.rejects(run(),/complete canonical asset response required/);
  assert.equal(reads,1);assert.equal(budget.snapshot().cpu.liveBytes,0);
  globalThis.fetch=async()=>new Response(new Uint8Array(subject.RESIDENT_ARTIFACT_CONFIG.gridBytes),
    {headers:{'Content-Length':String(subject.RESIDENT_ARTIFACT_CONFIG.gridBytes)}});
  await assert.rejects(run(),/canonical asset content identity mismatch/);
  assert.equal(budget.snapshot().cpu.liveBytes,0);assert.equal(consumed,false);
}finally{globalThis.fetch=originalFetch;budget.restore();}
console.log('PASS actual artifact first refusal precedes canonical asset fetch, model/device allocation and consumer; local policy only');
