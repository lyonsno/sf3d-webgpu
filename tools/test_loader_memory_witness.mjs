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
    memorySafety:{stop:{exitObserved:true,ownedPid}},processObservation:{runId,rootPid,status:'observed'},
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
const shared=()=>{
  const r=valid();r.requested.sharedAllowance=true;
  const ledger=()=>({cpu:{maxBytes:64,liveBytes:0,peakLiveBytes:32,physicalMemoryMeasured:false},gpu:{maxBytes:64,liveBytes:0,peakLiveBytes:48,physicalMemoryMeasured:false}});
  r.sharedAllowance={requested:{cpuBytes:64,gpuBytes:64},backend:{vendor:'apple',isFallbackAdapter:false},distinctOwnedDevices:true,parent:ledger(),children:[ledger(),ledger()],validationErrors:[null,null],outputs:[[14,22,26,34],[14,22,26,34]],
    held:{cpu:{liveBytes:32},gpu:{liveBytes:48}},refusals:{cpu:{name:'SF3DMemoryBudgetError',memoryBudget:{requestedBytes:40,liveBytes:32,maxBytes:64}},gpu:{name:'SF3DMemoryBudgetError',memoryBudget:{requestedBytes:32,liveBytes:48,maxBytes:64}}}};
  r.sharedAllowance.backends=[{vendor:'apple',isFallbackAdapter:false},{vendor:'apple',isFallbackAdapter:false}];
  return r;
};
for(const [name,mutate]of [
  ['requested shared mode silently omitted',r=>delete r.sharedAllowance],
  ['wrong shared quota',r=>r.sharedAllowance.parent.gpu.maxBytes=128],
  ['refusal without root contention',r=>r.sharedAllowance.refusals.gpu.memoryBudget.liveBytes=0],
  ['same device substituted',r=>r.sharedAllowance.distinctOwnedDevices=false],
  ['shared fallback route',r=>r.sharedAllowance.backend.isFallbackAdapter=true],
  ['second adapter fallback',r=>r.sharedAllowance.backends[1].isFallbackAdapter=true],
  ['second compute missing',r=>r.sharedAllowance.outputs.pop()],
  ['wrong native numerics',r=>r.sharedAllowance.outputs[1][0]=0],
  ['stranded root charge',r=>r.sharedAllowance.parent.gpu.liveBytes=48],
  ['validation error',r=>r.sharedAllowance.validationErrors[1]='device error'],
]){const r=shared();mutate(r);assert.equal(acceptLoaderMemoryWitness(r).ok,false,name);}
assert.equal(acceptLoaderMemoryWitness(shared()).ok,true,'synthetic shared fixture only tests acceptance policy');
const hostProducer=()=>{
  const r=valid();r.requested.hostProducerAllowance=true;
  r.producerArtifact={sourceRevision:revision,buildSucceeded:true,entryPath:'/dist-lib/sf3d-producer.js',sha256:'c'.repeat(64),servedSha256:'c'.repeat(64),lockSha256:'d'.repeat(64)};
  const budget=()=>({deviceAcquisition:'requested-through-budget',cpu:{liveBytes:0,maxBytes:1,physicalMemoryMeasured:false},gpu:{liveBytes:80,maxBytes:128,physicalMemoryMeasured:false}});
  r.hostProducer={moduleUrl:'/dist-lib/sf3d-producer.js',backend:{vendor:'apple',isFallbackAdapter:false},sameDevice:true,
    refusal:{name:'SF3DMemoryBudgetError',memoryBudget:{label:'weight-source'}},fetchCount:0,
    before:budget(),after:budget(),postRefusal:{name:'SF3DMemoryBudgetError',memoryBudget:{label:'post-producer-guard-control'}},
    control:[7,11,13,17],validationError:null,terminal:{cpu:{liveBytes:0},gpu:{liveBytes:0}}};return r;
};
{
  const r=hostProducer();delete r.hostProducer;
  assert.equal(acceptLoaderMemoryWitness(r).ok,false,'requested producer host contract cannot silently omit its primary evidence');
}
for(const [name,mutate]of [
  ['stale library source',r=>r.producerArtifact.sourceRevision='b'.repeat(40)],
  ['library not built',r=>r.producerArtifact.buildSucceeded=false],
  ['wrong served artifact',r=>r.producerArtifact.servedSha256='e'.repeat(64)],
  ['producer module substituted',r=>r.hostProducer.moduleUrl='/other.js'],
  ['producer fallback',r=>r.hostProducer.backend.isFallbackAdapter=true],
  ['wrong device',r=>r.hostProducer.sameDevice=false],
  ['unobserved acquisition',r=>r.hostProducer.after.deviceAcquisition='bound-after-acquisition'],
  ['auth failure pretending allocation refusal',r=>r.hostProducer.refusal.name='TypeError'],
  ['source fetched before refusal',r=>r.hostProducer.fetchCount=1],
  ['host guard removed',r=>delete r.hostProducer.postRefusal],
  ['host baseline retired',r=>r.hostProducer.after.gpu.liveBytes=0],
  ['shadowed host allowance',r=>r.hostProducer.after.gpu.maxBytes=256],
  ['no usable host control',r=>r.hostProducer.control=[]],
  ['native producer validation failure',r=>r.hostProducer.validationError='bad pipeline'],
  ['host allocations stranded',r=>r.hostProducer.terminal.gpu.liveBytes=80],
]){const r=hostProducer();mutate(r);assert.equal(acceptLoaderMemoryWitness(r).ok,false,name);}
assert.equal(acceptLoaderMemoryWitness(hostProducer()).ok,true,'synthetic producer fixture tests reporting policy, not platform conformance');
