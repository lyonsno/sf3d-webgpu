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
  memorySafety:{stop:{exitObserved:true,ownedPid}},processObservation:{runId,rootPid,status:'observed'},
  processRefusal:{status:'budget-refused',runId:runId+'-refusal',rootPid,coverage:'sampled-owned-process-tree',
    safety:{reason:'process-footprint-budget',observedBytes:100,maxFootprintBytes:25,actionStatus:'returned'},
    lastObservation:{status:'observed',runId:runId+'-refusal',rootPid,effectiveRoute:'darwin-libproc-proc_pid_rusage/RUSAGE_INFO_V4',sampledAggregatePhysicalFootprintBytes:100,processes:[{pid:rootPid},{pid:ownedPid}]}},
  cleanup:{browser:{exitObserved:true,ownedPid},server:'closed'}});
assert.equal(acceptStagedTensorWitness(valid()).ok,true,'actual patch-phase receipt must have its own learned-operation acceptance, not weight-upload acceptance');
{
  const omitted=valid();omitted.requested.foreground={revision:'c'.repeat(40),grid:32};
  assert.equal(acceptStagedTensorWitness(omitted).ok,false,'requested actual foreground cannot pass on a learned-only report');
}
{
  const r=valid();r.requested.foreground={revision:'c'.repeat(40),grid:32};r.browserArguments=['--window-size=1280,960'];
  const state={active:true,error:null,renderer:'ordinary-volume',grid:32,frameCount:2,simStepCount:2,submissions:2};
  r.requested.totalBytes=1152;
  const allowance={cpuBytes:1024,gpuBytes:128,totalBytes:1152};
  const budget={cpu:{maxBytes:1024,liveBytes:0},gpu:{maxBytes:128,liveBytes:0},total:{maxBytes:1152,liveBytes:0}};
  const child={...structuredClone(budget),parentAllowance:allowance};
  Object.assign(r.patchPhase.budget,{total:{maxBytes:1152,liveBytes:0},parentAllowance:allowance});
  r.patchPhase.hostBaseline=structuredClone(child);
  r.patchPhase.hostBudgetAfterPhase=structuredClone(child);
  r.patchPhase.deviceOwnership='pre-bound-caller-host';
  r.evidencePaths={foregroundBefore:'/explicit/before.png',foregroundAfter:'/explicit/after.png'};
  r.foreground={source:{revision:r.requested.foreground.revision,trackedClean:true},sameDevice:true,visibility:'headed-independent-browser',
    console:[],
    progressInterval:'before-selected-operation-to-immediate-operation-return',
    before:{...state,budget:{root:structuredClone(budget),children:[{budget:structuredClone(child)}]}},
    after:{...state,frameCount:4,simStepCount:4,submissions:4,budget:{root:structuredClone(budget),children:[{budget:structuredClone(child)}]}},
    terminalBudget:{root:budget,children:[{budget:child}]},
    textureEvents:[{bytes:16,descriptor:{format:'r32float'},effective:{format:'r32float'}}]};
  r.foreground.source.packageLockSha256=sha;
  const dependency='node_modules/three-mesh-bvh/build/index.module.js';
  r.foreground.servedSources={[dependency]:sha};
  r.foreground.dependencies={[dependency]:{package:'three-mesh-bvh',version:'0.8.3',lockSha256:sha,sha256:sha,
    canonicalIntegrityVerified:true,installedBytesMatched:true,byteLength:192653}};
  r.requested.foreground.repoRoot='/pinned/kaminos';
  r.foreground.service={route:'owned-actual-kaminos-handler.v0',ownedPid:57,serveSha256:sha,
    effective:{source:{repoRoot:'/pinned/kaminos',commit:r.requested.foreground.revision}},loopback:true};
  r.cleanup.foregroundService={ownedPid:57,exitObserved:true};
  assert.equal(acceptStagedTensorWitness(r).ok,true,'actual foreground must have its own positive acceptance path');
  for(const [label,mutate]of [
    ['observed black-route failure',x=>x.foreground.console.push({type:'error',text:"Scene route load failed: TypeError: Failed to execute 'createView' on 'GPUTexture'"})],
    ['offscreen',x=>x.browserArguments.push('--headless=new')],['wrong source',x=>x.foreground.source.revision='d'.repeat(40)],
    ['second device',x=>x.foreground.sameDevice=false],['rAF-only',x=>x.foreground.after.simStepCount=2],
    ['not presented',x=>x.foreground.after.frameCount=2],['no submits',x=>x.foreground.after.submissions=2],
    ['fallback renderer',x=>x.foreground.after.renderer='alternate-volume'],['silent smaller grid',x=>x.foreground.after.grid=16],
    ['no frame',x=>delete x.evidencePaths.foregroundAfter],['no textures',x=>x.foreground.textureEvents=[]],
    ['live host backing',x=>x.foreground.terminalBudget.root.gpu.liveBytes=1],
    ['missing total',x=>delete x.patchPhase.budget.total],
    ['wrong child total',x=>x.patchPhase.budget.total.maxBytes=999999],
    ['wrong parent total',x=>x.patchPhase.budget.parentAllowance.totalBytes=999999],
    ['missing root total',x=>delete x.foreground.terminalBudget.root.total],
    ['wrong root total',x=>x.foreground.terminalBudget.root.total.maxBytes=999999],
    ['wrong baseline total',x=>x.patchPhase.hostBaseline.total.maxBytes=999999],
    ['wrong active total',x=>x.foreground.after.budget.root.total.maxBytes=999999],
    ['later progress interval',x=>x.foreground.progressInterval='after-reference'],
    ['missing host parent',x=>delete x.patchPhase.hostBaseline.parentAllowance],
    ['missing dependency',x=>delete x.foreground.dependencies],
    ['unverified dependency',x=>x.foreground.dependencies[dependency].canonicalIntegrityVerified=false],
    ['changed served dependency',x=>x.foreground.servedSources[dependency]='c'.repeat(64)],
    ['missing actual service',x=>delete x.foreground.service],
    ['wrong actual server root',x=>x.foreground.service.effective.source.repoRoot='/other/kaminos'],
    ['stale actual server revision',x=>x.foreground.service.effective.source.commit='e'.repeat(40)],
    ['unowned actual service',x=>x.foreground.service.ownedPid=null],
    ['unobserved server exit',x=>x.cleanup.foregroundService.exitObserved=false],
  ]){const changed=structuredClone(r);mutate(changed);assert.equal(acceptStagedTensorWitness(changed).ok,false,label);}
}
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
{
  const crossed=valid();crossed.processObservation.status='budget-refused';
  crossed.processObservation.safety={reason:'process-footprint-budget',observedBytes:1192699536,maxFootprintBytes:1073741824};
  assert.equal(acceptStagedTensorWitness(crossed).ok,false,'a real consumer memory stop must not be overwritten by a later passing refusal control');
}
console.log('learned patch phase accepts source/input/host-bound complete numerics and rejects false closure');
