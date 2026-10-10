import assert from 'node:assert/strict';
import {acceptLoaderMemoryWitness} from './loader_memory_witness_acceptance.mjs';
const runId='synthetic-acceptance-policy', rootPid=23, ownedPid=41, revision='a'.repeat(40);
const row=()=>({requested:{cpuBytes:100,gpuBytes:8},error:{name:'SF3DMemoryBudgetError'},
  budget:{cpu:{maxBytes:100,liveBytes:0,physicalMemoryMeasured:false},gpu:{maxBytes:8,liveBytes:0,peakLiveBytes:8,physicalMemoryMeasured:false}}});
const valid=()=>{
  const cases={sourceRefusal:row(),conversionRefusal:row(),gpuRefusal:row()};
  cases.sourceRefusal.fetchCount=0;cases.conversionRefusal.budget.refusal={label:'fp16-conversion'};cases.conversionRefusal.budget.gpu.peakLiveBytes=0;
  return {status:'passed',runId,route:'sf3d-loader-native-refusal-synthetic-weights-no-inference.v0',
    source:{clean:true,revision},requested:{revision},backend:{vendor:'apple',isFallbackAdapter:false},cases,
    deviceControl:[7,11,13,17],validationError:null,ownedBrowserPid:ownedPid,
    memorySafety:{stop:{exitObserved:true,ownedPid}},processObservation:{runId,rootPid},
    processRefusal:{status:'budget-refused',runId:runId+'-refusal',rootPid,coverage:'sampled-owned-process-tree',
      safety:{reason:'process-footprint-budget',observedBytes:100,maxFootprintBytes:25,actionStatus:'returned'},
      lastObservation:{status:'observed',runId:runId+'-refusal',rootPid,effectiveRoute:'darwin-libproc-proc_pid_rusage/RUSAGE_INFO_V4',
        sampledAggregatePhysicalFootprintBytes:100,processes:[{pid:rootPid},{pid:ownedPid}]}},
    cleanup:{browser:{exitObserved:true,ownedPid},server:'closed'}};
};
assert.equal(acceptLoaderMemoryWitness(valid()).ok,true,'synthetic fixture tests acceptance policy, not an external runtime contract');
for(const [name,mutate]of [
  ['failed terminal report',r=>{r.status='failed';r.error={message:'earlier operation failed'};}],
  ['wrong route',r=>r.route='other'],['fallback',r=>r.backend.isFallbackAdapter=true],
  ['stale source',r=>r.source.revision='b'.repeat(40)],['blank evidence',r=>r.cases={}],
  ['shadowed caller allowance',r=>r.cases.gpuRefusal.budget.gpu.maxBytes=4],
  ['signal without exit',r=>r.memorySafety.stop.exitObserved=false],
  ['wrong owned child',r=>r.memorySafety.stop.ownedPid=99],
  ['stale process run',r=>r.processRefusal.lastObservation.runId='old'],
  ['partial process coverage',r=>r.processRefusal.coverage='partial-process-coverage'],
  ['missing browser observation',r=>r.processRefusal.lastObservation.processes=[{pid:rootPid}]],
  ['missing action result',r=>delete r.processRefusal.safety.actionStatus],
]){const r=valid();mutate(r);assert.equal(acceptLoaderMemoryWitness(r).ok,false,name+' must not close the native witness');}
console.log('Native memory witness rejects wrong route/source, fallback, blank/partial/stale evidence, shadowed allowance, and signal-only cleanup.');
