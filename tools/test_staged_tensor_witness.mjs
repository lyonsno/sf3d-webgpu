import assert from 'node:assert/strict';
import {acceptStagedTensorWitness} from './loader_memory_witness_acceptance.mjs';
const revision='a'.repeat(40),sha='b'.repeat(64),runId='policy-fixture',rootPid=23,ownedPid=41;
const valid=()=>({status:'passed',route:'sf3d-canonical-tensor-ranges-no-inference.v0',runId,
  requested:{revision,weightsSha256:sha,tensorNames:['mean'],cpuBytes:1024,gpuBytes:128},source:{clean:true,revision},
  backend:{vendor:'apple',isFallbackAdapter:false},validationError:null,ownedBrowserPid:ownedPid,
  canonicalSource:{sha256:sha,etag:'"canonical"',byteLength:512,units:[{name:'mean',expandedBytes:4,expectedF32Words:[1065353216]}]},
  tensorUnit:{loadingReport:{mode:'tensor-ranges',sourceETag:'"canonical"',expectedWeightBytes:512,ranges:[{},{},{}]},
    readbacks:[{name:'mean',bytes:4,f32Words:[1065353216]}],budget:{cpu:{maxBytes:1024,liveBytes:0,physicalMemoryMeasured:false},gpu:{maxBytes:128,liveBytes:0,physicalMemoryMeasured:false}}},
  memorySafety:{stop:{exitObserved:true,ownedPid}},processObservation:{runId,rootPid},
  processRefusal:{status:'budget-refused',runId:runId+'-refusal',rootPid,coverage:'sampled-owned-process-tree',
    safety:{reason:'process-footprint-budget',observedBytes:100,maxFootprintBytes:25,actionStatus:'returned'},
    lastObservation:{status:'observed',runId:runId+'-refusal',rootPid,effectiveRoute:'darwin-libproc-proc_pid_rusage/RUSAGE_INFO_V4',sampledAggregatePhysicalFootprintBytes:100,processes:[{pid:rootPid},{pid:ownedPid}]}},
  cleanup:{browser:{exitObserved:true,ownedPid},server:'closed'}});
assert.equal(acceptStagedTensorWitness(valid()).ok,true,'synthetic receipt tests local acceptance, not native conformance');
for(const [name,mutate]of [
  ['failed primary',r=>{r.status='failed';r.error={message:'upload failed'};}],
  ['fallback',r=>r.backend.isFallbackAdapter=true],['wrong route',r=>r.route='synthetic-weights'],
  ['wrong source head',r=>r.source.revision='c'.repeat(40)],['wrong artifact',r=>r.canonicalSource.sha256='c'.repeat(64)],
  ['wrong tensor selection',r=>r.requested.tensorNames=['other']],['range fallback',r=>r.tensorUnit.loadingReport.mode='whole-file'],
  ['changed ETag',r=>r.tensorUnit.loadingReport.sourceETag='"other"'],['partial acquisition',r=>r.tensorUnit.loadingReport.ranges.pop()],
  ['blank output',r=>r.tensorUnit.readbacks=[]],['wrong numerics',r=>r.tensorUnit.readbacks[0].f32Words=[0]],
  ['shadowed allowance',r=>r.tensorUnit.budget.cpu.maxBytes=512],['unretired buffers',r=>r.tensorUnit.budget.gpu.liveBytes=4],
  ['physical-fit inflation',r=>r.tensorUnit.budget.cpu.physicalMemoryMeasured=true],['signal-only',r=>r.memorySafety.stop.exitObserved=false],
  ['stale observer',r=>r.processRefusal.lastObservation.runId='old'],['partial observation',r=>r.processRefusal.coverage='partial'],
]){const r=valid();mutate(r);assert.equal(acceptStagedTensorWitness(r).ok,false,name);}
console.log('canonical-unit acceptance rejects wrong source/config/route, partial outputs, altered numerics and signal-only cleanup');
