import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {createTwoStreamAttentionDutyPlan} from '../src/lib/two_stream.js';
import {groupTwoStreamDuties} from '../src/lib/cooperative_two_stream.js';
import {acceptResidentTwoStream,twoStreamPhaseDemand,residentTwoStreamExpectedPhases} from './resident_two_stream_acceptance.mjs';
const hash=b=>createHash('sha256').update(b).digest('hex');
const leaf=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-two-stream-review-contract-'));
try{
  const input=path.join(leaf,'input'),output=path.join(leaf,'dino'),backboneOutput=path.join(leaf,'backbone');
  const ib=Buffer.alloc(3*512*512*4),ob=Buffer.alloc(1297*1024*4);ob.writeFloatLE(1,0);
  fs.writeFileSync(input,ib);fs.writeFileSync(output,ob);
  // Complete finite nonblank stream, not a model result or native authority.
  const fd=fs.openSync(backboneOutput,'wx');const chunk=Buffer.alloc(65536);chunk.writeFloatLE(1,0);
  fs.writeSync(fd,chunk);fs.ftruncateSync(fd,27648*1024*4);fs.closeSync(fd);
  const names=['preprocess','camera',...Array.from({length:24},(_,i)=>'dino-block-'+i),'dino-output',
    'two-stream-embedding-weights','two-stream-embedding-rearrange'];
  const graph=groupTwoStreamDuties(createTwoStreamAttentionDutyPlan(1297,{residentFFN:true,linearRowsPerDuty:128}));
  for(const group of graph){names.push('two-stream-'+group.stageId);
    for(const d of group.duties)if(twoStreamPhaseDemand({name:'two-stream-duty',duty:d}).requiredBytes>0)
      names.push('two-stream-duty-'+d.dutyIndex);}
  names.push('two-stream-output');assert.equal(names.length,118);
  const observation=phase=>({phase,phaseRequestedAtUnixMs:1000,verdict:'admitted',host:{source:'live-macos',hostname:'fixture',model:'Mac14,9',processor:'Apple M2 Pro',observerErrors:[],hostFreeBytes:100},
    demand:{requiredBytes:10},process:{coverage:'sampled-owned-process-tree',sampleCount:1,freshness:{route:'new-probe-after-request',requestedAtUnixMs:1000,probeAtUnixMs:1001,observationIndex:1},
      lastObservation:{runId:'fixture',rootPid:42,status:'observed',atUnixMs:1001,sampledAggregatePhysicalFootprintBytes:10}}});
  const valid={status:'passed',phase:'complete',source:{clean:true,revision:'fixture-head',hostname:'fixture'},
    requested:{revision:'fixture-head',weightsSha256:'fixture-weights',processBudgetBytes:100,attentionRowsPerDuty:128},
    backend:{vendor:'apple',isFallbackAdapter:false},artifact:{sha256:'fixture-bundle',servedSha256:'fixture-bundle',sourceRevision:'fixture-head',kitVersion:'0.1.53'},
    canonicalSource:{sha256:'fixture-weights',etag:'"fixture-weights"',byteLength:200},
    dino:{weightPhases:Array.from({length:25},()=>({status:'completed-retired'})),loadingReport:{sourceETag:'"fixture-weights"',expectedWeightBytes:200},
      cooperative:{status:'succeeded',schedulingMode:'cooperative',queueCompletionAuthority:'per-gpu-duty-prefix-fence',boundaries:[{completedItems:24,totalItems:24,actualRangeCount:24}]}},
    twoStream:{residentFFN:true,linearRowsPerDuty:128,attentionRowsPerDuty:128,blocks:4,basicBlocks:12,shape:[3,1024,96,96],weightPhases:Array.from({length:23},()=>({status:'completed-retired'})),
      loadingReport:{sourceETag:'"fixture-weights"',expectedWeightBytes:200},cooperative:{status:'succeeded',schedulingMode:'cooperative',
        routeId:'sf3d.image-to-mesh.webgpu-local.v0',manifestId:'sf3d.two-stream-attention-cooperative-boundaries.v0',queueCompletionAuthority:'per-gpu-duty-prefix-fence',
        boundaries:[{boundaryId:'two-stream-attention-duties',completedItems:3350,totalItems:3350,actualRangeCount:3350}],
        adapterTelemetry:{residentFFN:true,linearRowsPerDuty:128,attentionRowsPerDuty:128,dutyGranularity:'attention-tile',declaredDutyCount:3350}}},
    phaseObservations:names.map(observation),expectedPhaseOrder:names,runId:'fixture',rootPid:42,validationError:null,budget:{cpu:{liveBytes:0},gpu:{liveBytes:0}},
    processObservation:{status:'observed',coverage:'sampled-owned-process-tree',sampledPeakAggregatePhysicalFootprintBytes:20},cleanup:{browser:{exitObserved:true},server:'closed'},
    evidencePaths:{input,output,twoStreamOutput:backboneOutput},inputTensor:{sha256:hash(ib)},inputConsumed:{sha256:hash(ib)},output:{sha256:hash(ob)},
    twoStreamReadbackBytes:27648*1024*4,twoStreamOutput:{sha256:hash(fs.readFileSync(backboneOutput))}};
  const absent=structuredClone(valid);absent.phaseObservations=absent.phaseObservations.slice(0,27);absent.expectedPhaseOrder=names.slice(0,27);
  assert.equal(acceptResidentTwoStream(absent).ok,false,'zero backbone observations cannot pass via a self-reported shortened expected graph');
  assert.equal(acceptResidentTwoStream(valid).ok,true,'synthetic acceptance-policy fixture only');
  const reuse=structuredClone(valid);reuse.requested.reuseDeadTriplaneStorage=true;
  assert.equal(acceptResidentTwoStream(reuse).ok,false,'requested reuse cannot conceal an allocating effective route');
  reuse.twoStream.reuseDeadTriplaneStorage=true;reuse.twoStream.cooperative.adapterTelemetry.reuseDeadTriplaneStorage=true;
  const reusePlan=createTwoStreamAttentionDutyPlan(1297,{residentFFN:true});
  reuse.twoStream.cooperative.adapterTelemetry.storageReuse=reusePlan.filter(d=>d.kind==='fuse-residual-norm').map(d=>
    ({block:d.block,afterDutyIndex:d.dutyIndex,bytes:27648*1024*4,source:'owned-work-inventory-after-gpu-prefix'}));
  for(const o of reuse.phaseObservations)if(o.phase.startsWith('two-stream-duty-'))o.descriptor={reuseDeadTriplaneStorage:true};
  assert.equal(acceptResidentTwoStream(reuse).ok,true,'effective per-block prefix transfers plus complete source work are required');
  for(const mutate of [r=>delete r.twoStream.reuseDeadTriplaneStorage,
    r=>r.twoStream.cooperative.adapterTelemetry.reuseDeadTriplaneStorage=false,
    r=>r.twoStream.cooperative.adapterTelemetry.storageReuse.pop(),
    r=>r.twoStream.cooperative.adapterTelemetry.storageReuse[0].afterDutyIndex--,
    r=>r.twoStream.cooperative.adapterTelemetry.storageReuse[0].source='requested-only',
    r=>r.phaseObservations.find(o=>o.phase.startsWith('two-stream-duty-')).descriptor.reuseDeadTriplaneStorage=false]){
    const drift=structuredClone(reuse);mutate(drift);assert.equal(acceptResidentTwoStream(drift).ok,false);
  }
  const diagnostic=structuredClone(valid);diagnostic.requested.hostHeadroomPolicy='darwin-available-memory-estimate-v1';
  diagnostic.hostPressureGuard={status:'observed',policy:'darwin-available-memory-estimate-v1',pressureStopFreePercent:24};
  for(const o of diagnostic.phaseObservations){Object.assign(o.host,{platform:'darwin',hostTotalBytes:1000,hostFreeBytes:0,
    hostMemoryPressureFreePercent:52,startedAtUnixMs:1000,observedAt:new Date(1002).toISOString(),
    availableMemory:{bytes:100,source:'process.availableMemory/uv_get_available_memory',nodeVersion:'v25.9.0',libuvVersion:'1.52.1',executable:'/fixture/node',observedAtUnixMs:1001}});
    o.hostHeadroom={policy:'darwin-available-memory-estimate-v1',verdict:'admitted',availableBytes:100,requiredBytes:10,pressureStopFreePercent:24};}
  assert.equal(acceptResidentTwoStream(diagnostic).ok,true,'explicit estimate policy is accepted only with complete effective observations and pressure guard');
  for(const mutate of [r=>delete r.hostPressureGuard,r=>r.hostPressureGuard.status='refused',r=>r.phaseObservations[28].host.availableMemory.bytes=0,
    r=>delete r.phaseObservations[0].hostHeadroom,r=>r.phaseObservations[28].hostHeadroom.policy='raw-free-pages-v0',
    r=>r.phaseObservations[28].host.hostMemoryPressureFreePercent=24,r=>r.requested.hostHeadroomPolicy='unknown']){
    const drift=structuredClone(diagnostic);mutate(drift);assert.equal(acceptResidentTwoStream(drift).ok,false);}
  const smaller=structuredClone(valid),rows=32;
  const smallerNames=['preprocess','camera',...Array.from({length:24},(_,i)=>'dino-block-'+i),'dino-output',
    'two-stream-embedding-weights','two-stream-embedding-rearrange'];
  const smallerPlan=createTwoStreamAttentionDutyPlan(1297,{residentFFN:true,attentionRowsPerDuty:rows});
  for(const g of groupTwoStreamDuties(smallerPlan)){smallerNames.push('two-stream-'+g.stageId);
    for(const duty of g.duties)if(twoStreamPhaseDemand({name:'two-stream-duty',duty,attentionRowsPerDuty:rows}).requiredBytes>0)
      smallerNames.push('two-stream-duty-'+duty.dutyIndex);}
  smallerNames.push('two-stream-output');
  assert.deepEqual(smallerNames,residentTwoStreamExpectedPhases(rows),'canonical independently-derived graph matches complete source plan at measured smaller allocation granule');
  smaller.requested.attentionRowsPerDuty=rows;smaller.twoStream.attentionRowsPerDuty=rows;
  const c=smaller.twoStream.cooperative;c.adapterTelemetry.attentionRowsPerDuty=rows;c.adapterTelemetry.declaredDutyCount=smallerPlan.length;
  Object.assign(c.boundaries[0],{completedItems:smallerPlan.length,totalItems:smallerPlan.length,actualRangeCount:smallerPlan.length});
  smaller.expectedPhaseOrder=smallerNames;smaller.phaseObservations=smallerNames.map(observation);
  assert.equal(acceptResidentTwoStream(smaller).ok,true);
  c.adapterTelemetry.attentionRowsPerDuty=128;assert.equal(acceptResidentTwoStream(smaller).ok,false,'requested smaller scratch cannot conceal effective unchanged128-row route');
  for(const mutate of [
    r=>r.expectedPhaseOrder.pop(),r=>r.phaseObservations.pop(),r=>r.phaseObservations.splice(28,1),
    r=>r.twoStream.cooperative.routeId='other.model',r=>r.twoStream.cooperative.manifestId='other.manifest',
    r=>r.twoStream.cooperative.adapterTelemetry.linearRowsPerDuty=1,
    r=>r.twoStream.cooperative.adapterTelemetry.dutyGranularity='stage',
    r=>r.twoStream.cooperative.boundaries[0].boundaryId='other.boundary',
    r=>r.phaseObservations[28].phaseRequestedAtUnixMs=1002,
    r=>delete r.phaseObservations[28].process.freshness,
    r=>r.phaseObservations[28].process.freshness.probeAtUnixMs=999,
    r=>r.phaseObservations[28].process.freshness.observationIndex=0,
    r=>r.phaseObservations[28].process.lastObservation.atUnixMs=999,
  ]){const drift=structuredClone(valid);mutate(drift);assert.equal(acceptResidentTwoStream(drift).ok,false);}
}finally{fs.rmSync(leaf,{recursive:true});}
console.log('canonical complete backbone observations, effective route/config and pre-output import failure contracts');

