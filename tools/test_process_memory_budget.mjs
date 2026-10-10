// Adapted from Kaminos models/trellis2/tests/process-memory-budget-contracts.mjs @ 5c9a2fc2939e0a28ac3560a1e9108e81ba245fa7.
// Source provenance retained; SF3D schema/name adaptation, not a new physical-capacity claim.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {startProcessMemory} from './process_memory_guard.mjs';
const out=await fs.mkdtemp(path.join(os.tmpdir(),'trellis-memory-budget-'));
const row=bytes=>({runId:'budget',rootPid:42,status:'observed',processes:[{pid:42,processStartAbstime:123,
  physicalFootprintBytes:bytes,kernelLifetimePeakPhysicalFootprintBytes:bytes}],sampledAggregatePhysicalFootprintBytes:bytes});
try{
  let count=0,actions=[];
  const monitor=await startProcessMemory({rootPid:42,runId:'budget',rawPath:path.join(out,'later.jsonl'),
    maxFootprintBytes:150,onUnsafe:async r=>actions.push(r),probe:async()=>row(count++?200:100)});
  await monitor.sample();const summary=await monitor.stop();
  assert.equal(actions.length,1,'a caller-selected footprint ceiling needs an actual one-shot stop action');
  assert.equal(summary.status,'budget-refused');assert.equal(summary.safety.reason,'process-footprint-budget');
  assert.equal(summary.safety.observedBytes,200);assert.equal(summary.safety.maxFootprintBytes,150);
  assert.equal(JSON.parse(await fs.readFile(summary.summaryPath)).safety.observedBytes,200);
  actions=[];
  await assert.rejects(startProcessMemory({rootPid:42,runId:'budget',rawPath:path.join(out,'initial.jsonl'),
    maxFootprintBytes:50,onUnsafe:async r=>actions.push(r),probe:async()=>row(100)}),/memory budget/);
  assert.equal(actions.length,1,'startup refusal occurs before browser launch');
  actions=[];let probes=0;
  const unavailable=await startProcessMemory({rootPid:42,runId:'budget',rawPath:path.join(out,'unavailable.jsonl'),
    maxFootprintBytes:150,onUnsafe:async r=>actions.push(r),probe:async()=>probes++?row(-1):row(100)});
  await unavailable.sample();await unavailable.stop();assert.equal(actions.length,1);
  assert.equal(actions[0].reason,'memory-observation-unavailable','bounded run cannot continue when its guard loses observation');
  const exited={pid:43,parentPid:42,errno:3,exitedBeforeMeasurement:true,
    exitEvidence:{route:'ps-pid-status',returnCode:0,status:'Z'}};
  const retired=await startProcessMemory({rootPid:42,runId:'budget',rawPath:path.join(out,'exited.jsonl'),
    maxFootprintBytes:150,onUnsafe:async()=>{throw Error('confirmed exit must not stop a run');},
    probe:async()=>({...row(100),unavailableProcesses:[exited]})});
  assert.equal((await retired.stop()).status,'observed','OS-confirmed exited descendants do not make live coverage partial');
  for(const missing of [{...exited,exitEvidence:{...exited.exitEvidence,status:'S'}},
    {...exited,errno:13},{...exited,exitEvidence:undefined}]){
    const stopped=[];
    await assert.rejects(startProcessMemory({rootPid:42,runId:'budget',rawPath:path.join(out,'live-missing.jsonl'),
      maxFootprintBytes:150,onUnsafe:async reason=>stopped.push(reason),
      probe:async()=>({...row(100),unavailableProcesses:[missing]})}),/partial process coverage/);
    assert.equal(stopped.length,1,'an exit label without observed OS exit evidence cannot waive missing coverage');
  }
  await assert.rejects(startProcessMemory({rootPid:42,runId:'budget',rawPath:path.join(out,'no-action.jsonl'),
    maxFootprintBytes:150,probe:async()=>row(100)}),/onUnsafe/);
  actions=[];let ioProbes=0;
  const broken=await startProcessMemory({rootPid:42,runId:'budget',rawPath:path.join(out,'broken-report.jsonl'),summaryPath:out,
    maxFootprintBytes:150,onUnsafe:async r=>actions.push(r),probe:async()=>row(ioProbes++?200:100)});
  await broken.sample().catch(()=>{});await broken.stop().catch(()=>{});
  assert.equal(actions.length,1,'report persistence failure must not suppress the safety stop action');
}finally{await fs.rm(out,{recursive:true,force:true});}
console.log('Process guard: explicit threshold, startup refusal, partial coverage and report-failure safety action pass.');
