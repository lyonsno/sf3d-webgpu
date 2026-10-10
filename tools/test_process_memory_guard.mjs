// Adapted from Kaminos models/trellis2/tests/process-memory-collector-contracts.mjs @ 5c9a2fc2939e0a28ac3560a1e9108e81ba245fa7.
// Source provenance retained; SF3D schema/name adaptation, not a new physical-capacity claim.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {startProcessMemory} from './process_memory_guard.mjs';
const out=await fs.mkdtemp(path.join(os.tmpdir(),'trellis-memory-collector-'));
const row=(runId='one')=>({runId,rootPid:42,status:'observed',processes:[{pid:42,processStartAbstime:123,
  physicalFootprintBytes:100,residentBytes:200,kernelLifetimePeakPhysicalFootprintBytes:120}],sampledAggregatePhysicalFootprintBytes:100});
try{
  const monitor=await startProcessMemory({rootPid:42,runId:'one',rawPath:path.join(out,'one.jsonl'),periodMs:1000,probe:async()=>row()});
  const baseline=await monitor.sample();
  assert.equal(baseline?.lastObservation?.runId,'one','sample must return its current in-memory baseline, not require a terminal summary file');
  assert.equal(baseline.rootPid,42);assert.equal(baseline.status,'running');assert.equal(baseline.coverage,'sampled-owned-process-tree');
  assert.equal(baseline.lastObservation.sampledAggregatePhysicalFootprintBytes,100);
  await assert.rejects(fs.access(path.join(out,'one.jsonl.summary.json')),/ENOENT/);
  baseline.lastObservation.runId='caller-mutated';baseline.processes['42:123'].physicalFootprintBytes=0;
  const next=await monitor.sample();assert.equal(next.lastObservation.runId,'one');assert.equal(next.processes['42:123'].physicalFootprintBytes,100);
  const s=await monitor.stop();assert.equal(s.status,'observed');assert.ok(s.sampleCount>=3);
  assert.equal(s.sampledPeakAggregatePhysicalFootprintBytes,100);assert.equal(s.processes['42:123'].kernelLifetimePeakPhysicalFootprintBytes,120);
  assert.equal((await fs.readFile(s.rawPath,'utf8')).trim().split('\n').length,s.sampleCount);
  // Native libproc sample: memory-current-native-r3/report.json,
  // PID28064/start2545885766296. Current and lifetime counters can disagree;
  // retain both instead of assuming an atomic ordered snapshot.
  const nativeCounters={...row(),processes:[{...row().processes[0],
    physicalFootprintBytes:1309445288,kernelLifetimePeakPhysicalFootprintBytes:1309412520}],
    sampledAggregatePhysicalFootprintBytes:1309445288};
  const nativeMonitor=await startProcessMemory({rootPid:42,runId:'one',rawPath:path.join(out,'native-counters.jsonl'),
    probe:async()=>nativeCounters,maxFootprintBytes:8589934592,onUnsafe:()=>assert.fail('valid native counters must not stop generation')});
  const nativeSummary=await nativeMonitor.stop();assert.equal(nativeSummary.status,'observed');
  assert.equal(nativeSummary.sampledPeakAggregatePhysicalFootprintBytes,1309445288);
  assert.equal(nativeSummary.processes['42:123'].kernelLifetimePeakPhysicalFootprintBytes,1309412520);
  await assert.rejects(startProcessMemory({rootPid:42,runId:'one',rawPath:path.join(out,'negative-peak.jsonl'),
    probe:async()=>({...row(),processes:[{...row().processes[0],kernelLifetimePeakPhysicalFootprintBytes:-1}]})}),/footprint\/start/);
  await assert.rejects(startProcessMemory({rootPid:42,runId:'one',rawPath:path.join(out,'stale.jsonl'),probe:async()=>row('old')}),/current-owner/);
  await assert.rejects(startProcessMemory({rootPid:42,runId:'one',rawPath:path.join(out,'missing.jsonl'),probe:async()=>({...row(),status:'unavailable'})}),/current-owner/);
  let count=0;const failed=await startProcessMemory({rootPid:42,runId:'one',rawPath:path.join(out,'failed.jsonl'),probe:async()=>{
    if(count++)throw Error('observed sampler failure');return row();}});
  const failure=await failed.stop();assert.equal(failure.status,'failed');assert.match(failure.error,/sampler failure/);
  let staleCount=0;const stale=await startProcessMemory({rootPid:42,runId:'one',rawPath:path.join(out,'stale-after-start.jsonl'),
    probe:async()=>row(staleCount++?'old':'one')});
  const staleSnapshot=await stale.sample();assert.equal(staleSnapshot.status,'failed');assert.match(staleSnapshot.error,/current-owner/);
  assert.equal((await stale.stop()).status,'failed');
}finally{await fs.rm(out,{recursive:true,force:true});}
console.log('Uncapped raw samples and distinct process/lifetime peaks preserve identity; stale, missing and failed sampling cannot masquerade as measured zero.');
