import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {startProcessMemory} from './process_memory_guard.mjs';
const leaf=await fs.mkdtemp(path.join(os.tmpdir(),'sf3d-fresh-process-contract-'));
let uploaded=false,calls=0,resolveOld;
const row=bytes=>({runId:'fresh',rootPid:42,status:'observed',atUnixMs:Date.now(),processes:[{pid:42,processStartAbstime:123,
  physicalFootprintBytes:bytes,kernelLifetimePeakPhysicalFootprintBytes:bytes}],sampledAggregatePhysicalFootprintBytes:bytes});
const monitor=await startProcessMemory({rootPid:42,runId:'fresh',rawPath:path.join(leaf,'process.jsonl'),probe:()=>{
  const captured=row(uploaded?100:10);if(++calls===2)return new Promise(resolve=>resolveOld=()=>resolve(captured));return captured;
}});
try{
  const old=monitor.sample();assert.equal(calls,2);uploaded=true;
  const afterUpload=monitor.sample({fresh:true});assert.equal(calls,2);
  resolveOld();await old;const current=await afterUpload;
  assert.equal(current.lastObservation.sampledAggregatePhysicalFootprintBytes,100,'allocation-boundary request must not reuse in-flight pre-upload measurement');
  assert.equal(calls,3,'a distinct post-request probe is required');
  assert.equal(current.freshness.route,'new-probe-after-request');
  assert.ok(current.lastObservation.atUnixMs>=current.freshness.requestedAtUnixMs);
  assert.equal(current.freshness.observationIndex,current.sampleCount);
  const baseline=await monitor.sample();assert.equal(baseline.lastObservation.sampledAggregatePhysicalFootprintBytes,100);
  // Fresh mode also refuses a source-shaped but old captured row.
  let count=0;
  const stale=await startProcessMemory({rootPid:42,runId:'fresh',rawPath:path.join(leaf,'stale.jsonl'),probe:()=>{
    const r=row(10);if(count++)r.atUnixMs=1;return r;
  }});
  await assert.rejects(stale.sample({fresh:true}),/fresh|post.request/);await stale.stop();
}finally{await monitor.stop();await fs.rm(leaf,{recursive:true});}
console.log('fresh boundary sample waits for coalesced older probe then measures a distinct post-request baseline');

