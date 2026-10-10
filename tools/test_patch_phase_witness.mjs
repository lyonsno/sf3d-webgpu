import assert from 'node:assert/strict';
import {acceptStagedTensorWitness} from './loader_memory_witness_acceptance.mjs';
const revision='a'.repeat(40),sha='b'.repeat(64),runId='phase-policy-fixture',rootPid=23,ownedPid=41;
const valid=()=>({status:'passed',route:'sf3d-canonical-patch-embedding.v0',runId,
  requested:{revision,weightsSha256:sha,inputSha256:sha,tensorNames:['weight'],cpuBytes:1024,gpuBytes:128},source:{clean:true,revision,hostname:'nlm2pr.local'},
  backend:{vendor:'apple',isFallbackAdapter:false},validationError:null,ownedBrowserPid:ownedPid,
  canonicalSource:{sha256:sha,etag:'"canonical"',byteLength:512,units:[{name:'weight',expandedBytes:4}]},
  inputArtifact:{sha256:sha},
  phaseAdmission:{authority:'reversible-selected-phase-only',host:{source:'live-macos',hostname:'nlm2pr.local',model:'Mac14,9',processor:'Apple M2 Pro',hostFreeBytes:4096},requiredBytes:1024,verdict:'admitted'},
  patchPhase:{shape:[1297,1024],inputBytes:3145728,outputBytes:5312512,finiteCount:1328128,nonzeroCount:1328000,
    outputSha256:sha,inputSha256:sha,reference:{tested:66,mismatches:0},
    loadingReport:{mode:'tensor-ranges',sourceETag:'"canonical"',expectedWeightBytes:512,ranges:[{},{},{}]},
    budget:{cpu:{maxBytes:1024,liveBytes:0,physicalMemoryMeasured:false},gpu:{maxBytes:128,liveBytes:0,physicalMemoryMeasured:false}}},
  memorySafety:{stop:{exitObserved:true,ownedPid}},processObservation:{runId,rootPid},
  processRefusal:{status:'budget-refused',runId:runId+'-refusal',rootPid,coverage:'sampled-owned-process-tree',
    safety:{reason:'process-footprint-budget',observedBytes:100,maxFootprintBytes:25,actionStatus:'returned'},
    lastObservation:{status:'observed',runId:runId+'-refusal',rootPid,effectiveRoute:'darwin-libproc-proc_pid_rusage/RUSAGE_INFO_V4',sampledAggregatePhysicalFootprintBytes:100,processes:[{pid:rootPid},{pid:ownedPid}]}},
  cleanup:{browser:{exitObserved:true,ownedPid},server:'closed'}});
assert.equal(acceptStagedTensorWitness(valid()).ok,true,'actual patch-phase receipt must have its own learned-operation acceptance, not weight-upload acceptance');
for(const [name,mutate]of [
  ['fallback',r=>r.backend.isFallbackAdapter=true],['wrong route',r=>r.route='full-model-fit'],
  ['wrong input',r=>r.inputArtifact.sha256='c'.repeat(64)],['no live admission',r=>r.phaseAdmission.host.source='replay'],
  ['wrong M2 identity',r=>r.phaseAdmission.host.model='other'],['overcommitted phase',r=>r.phaseAdmission.requiredBytes=8192],
  ['no phase admission',r=>r.phaseAdmission.verdict='refused'],['blank output',r=>r.patchPhase.nonzeroCount=0],
  ['partial output',r=>r.patchPhase.outputBytes=4],['NaN output',r=>r.patchPhase.finiteCount=1],
  ['wrong numerics',r=>r.patchPhase.reference.mismatches=1],['missing reference',r=>r.patchPhase.reference.tested=0],
  ['no raw output',r=>delete r.patchPhase.outputSha256],['unretired buffers',r=>r.patchPhase.budget.gpu.liveBytes=4],
  ['unobserved owned exit',r=>r.memorySafety.stop.exitObserved=false],['stale observer',r=>r.processRefusal.lastObservation.runId='old'],
]){const r=valid();mutate(r);assert.equal(acceptStagedTensorWitness(r).ok,false,name);}
console.log('learned patch phase accepts source/input/host-bound complete numerics and rejects false closure');
