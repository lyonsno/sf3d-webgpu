import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import * as admission from './memory_admission.mjs';

const fakeExec=command=>command==='memory_pressure'?'System-wide memory free percentage: 52%':'total = 5120.00M used = 4000.00M free = 1120.00M';
const observed=admission.observeMacMemory({exec:fakeExec,statfs:()=>({bavail:4,bsize:4096}),availableMemory:()=>4096});
assert.equal(observed.availableMemory?.bytes,4096,'actual available-memory estimate must be retained alongside raw free pages');
assert.equal(observed.availableMemory.source,'process.availableMemory/uv_get_available_memory');
assert.equal(observed.availableMemory.nodeVersion,process.version);
assert.equal(observed.availableMemory.libuvVersion,process.versions.uv);
assert.ok(observed.availableMemory.observedAtUnixMs>=observed.startedAtUnixMs);
assert.ok(observed.availableMemory.observedAtUnixMs<=Date.parse(observed.observedAt));
assert.equal(admission.observeMacMemory({exec:fakeExec,statfs:()=>({bavail:4,bsize:4096}),availableMemory:()=>{throw Error('unsupported');}}).availableMemory.bytes,null);
assert.equal(typeof admission.evaluatePhaseHostHeadroom,'function','explicit diagnostic policy evaluator required');
const policy='darwin-available-memory-estimate-v1';
const host={source:'live-macos',platform:'darwin',hostname:'fixture',hostTotalBytes:8192,hostFreeBytes:100,hostMemoryPressureFreePercent:52,
  startedAtUnixMs:1000,observedAt:new Date(1002).toISOString(),observerErrors:[],availableMemory:{bytes:4096,source:'process.availableMemory/uv_get_available_memory',
    nodeVersion:'v25.9.0',libuvVersion:'1.52.1',executable:'/actual/node',observedAtUnixMs:1001}};
const decide=(h=host,p=policy)=>admission.evaluatePhaseHostHeadroom({host:h,requiredBytes:1024,policy:p,requestedAtUnixMs:1000});
assert.equal(decide().verdict,'admitted');
assert.equal(decide(host,'raw-free-pages-v0').verdict,'refused','ordinary default raw-free policy remains unchanged');
assert.equal(admission.evaluatePhaseHostHeadroom({host,requiredBytes:1024}).verdict,'refused');
for(const mutate of [h=>delete h.availableMemory,h=>h.availableMemory.bytes=0,h=>h.availableMemory.bytes='4096',h=>h.availableMemory.bytes=NaN,
  h=>h.availableMemory.bytes=Infinity,h=>h.availableMemory.bytes=8193,h=>h.availableMemory.source='os.freemem',h=>h.availableMemory.nodeVersion='v26.0.0',
  h=>h.availableMemory.libuvVersion='other',h=>delete h.availableMemory.executable,h=>h.availableMemory.observedAtUnixMs=999,
  h=>h.startedAtUnixMs=999,h=>h.availableMemory.observedAtUnixMs=1003,h=>h.platform='linux',h=>h.source='replay',
  h=>h.hostMemoryPressureFreePercent=24,h=>h.hostMemoryPressureFreePercent=23,h=>h.hostMemoryPressureFreePercent=null,
  h=>h.observerErrors.push({error:'missing field'})]){
  const drift=structuredClone(host);mutate(drift);assert.equal(decide(drift).verdict,'refused','invalid/stale/fallback evidence cannot admit');
}
assert.equal(decide(host,'unknown').verdict,'refused');
assert.equal(admission.evaluatePhaseHostHeadroom({host,requiredBytes:-1,policy,requestedAtUnixMs:1000}).verdict,'refused');
const phase={host,demand:{requiredBytes:1024},phaseRequestedAtUnixMs:1000,hostHeadroom:decide()};
const report={requested:{hostHeadroomPolicy:policy},hostPressureGuard:{status:'observed',policy,pressureStopFreePercent:24}};
assert.equal(admission.phaseHostHeadroomIsAdmitted(report,phase),true);
for(const mutate of [p=>delete p.hostHeadroom,p=>p.hostHeadroom.policy='raw-free-pages-v0',p=>p.hostHeadroom.availableBytes=5000,
  p=>p.hostHeadroom.verdict='refused',p=>p.hostHeadroom.pressureStopFreePercent=1]){
  const drift=structuredClone(phase);mutate(drift);assert.equal(admission.phaseHostHeadroomIsAdmitted(report,drift),false);
}
assert.equal(admission.phaseHostHeadroomIsAdmitted({...report,hostPressureGuard:{status:'failed'}},phase),false);
assert.equal(admission.phaseHostHeadroomIsAdmitted({requested:{}},{host:{hostFreeBytes:20},demand:{requiredBytes:10}}),true,'historical/default reports stay replayable');

const leaf=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-available-contract-'));
try{
  let stops=0,low=false;
  const observe=()=>{const at=Date.now();return {...structuredClone(host),startedAtUnixMs:at,observedAt:new Date(at).toISOString(),
    availableMemory:{...host.availableMemory,observedAtUnixMs:at},hostMemoryPressureFreePercent:low?24:52};};
  const guard=await admission.startHostPressureGuard({rawPath:path.join(leaf,'host.jsonl'),observe,onUnsafe:async()=>{stops++;}});
  low=true;await guard.sample();const summary=await guard.stop();
  assert.equal(stops,1);assert.equal(summary.status,'refused');assert.equal(summary.safety.reason,'host-pressure-warning');
  assert.equal(fs.readFileSync(summary.rawPath,'utf8').trim().split('\n').length,2,'complete host signal retained');
  assert.equal(summary.policy,policy);assert.equal(summary.pressureStopFreePercent,24);
  await assert.rejects(admission.startHostPressureGuard({rawPath:path.join(leaf,'missing','host.jsonl'),observe,onUnsafe:async()=>{stops++;}}));
  assert.equal(stops,2,'report failure cannot prevent the safety stop');
}finally{fs.rmSync(leaf,{recursive:true});}
console.log('available-memory source/version/freshness, explicit policy, unchanged default and pressure stop contracts pass');
