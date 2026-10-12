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

  const attentionReuse=structuredClone(reuse);attentionReuse.requested.reuseAttentionResidualStorage=true;
  assert.equal(acceptResidentTwoStream(attentionReuse).ok,false,'requested residual reuse cannot certify an allocating route');
  attentionReuse.twoStream.reuseAttentionResidualStorage=true;
  attentionReuse.twoStream.cooperative.adapterTelemetry.reuseAttentionResidualStorage=true;
  attentionReuse.twoStream.cooperative.adapterTelemetry.attentionResidualReuse=reusePlan
    .filter(d=>d.kind==='fuse-attention-linear-range'&&d.rangeIndex===d.rangeCount-1).map(d=>
      ({block:d.block,afterDutyIndex:d.dutyIndex,bytes:27648*1024*4,source:'owned-projected-attention-after-gpu-prefix'}));
  for(const o of attentionReuse.phaseObservations)if(o.phase.startsWith('two-stream-duty-'))o.descriptor.reuseAttentionResidualStorage=true;
  attentionReuse.requested.processBudgetBytes=2147483648;
  const residualCliffs=attentionReuse.twoStream.cooperative.adapterTelemetry.attentionResidualReuse.map(r=>'two-stream-duty-'+(r.afterDutyIndex+1));
  for(const o of attentionReuse.phaseObservations.filter(o=>residualCliffs.includes(o.phase))){
    o.descriptor.duty={kind:'fuse-residual-norm'};
    o.demand=twoStreamPhaseDemand({name:'two-stream-duty',...o.descriptor});o.host.hostFreeBytes=2147483648;
  }
  assert.equal(acceptResidentTwoStream(attentionReuse).ok,true,'complete effective attention residual transfers required');
  for(const mutate of [r=>delete r.twoStream.reuseAttentionResidualStorage,
    r=>r.twoStream.cooperative.adapterTelemetry.reuseAttentionResidualStorage=false,
    r=>r.twoStream.cooperative.adapterTelemetry.attentionResidualReuse.pop(),
    r=>r.twoStream.cooperative.adapterTelemetry.attentionResidualReuse[0].afterDutyIndex--,
    r=>r.twoStream.cooperative.adapterTelemetry.attentionResidualReuse[0].bytes--,
    r=>r.twoStream.cooperative.adapterTelemetry.attentionResidualReuse[0].source='requested-only',
    r=>r.phaseObservations.find(o=>residualCliffs.includes(o.phase)).demand.requiredBytes=0,
    r=>r.phaseObservations.find(o=>residualCliffs.includes(o.phase)).demand.workGpuBytes=0,
    r=>r.phaseObservations.find(o=>o.phase.startsWith('two-stream-duty-')).descriptor.reuseAttentionResidualStorage=false]){
    const drift=structuredClone(attentionReuse);mutate(drift);assert.equal(acceptResidentTwoStream(drift).ok,false);
  }

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
  const queryReuse=structuredClone(smaller);
  queryReuse.requested.reuseAttentionQueryStorage=true;
  queryReuse.requested.processBudgetBytes=2147483648;
  queryReuse.twoStream.reuseAttentionQueryStorage=true;
  queryReuse.twoStream.cooperative.adapterTelemetry.reuseAttentionQueryStorage=true;
  for(const o of queryReuse.phaseObservations){
    o.host.hostFreeBytes=2147483648;
    if(o.phase.startsWith('two-stream-duty-')){
      const duty=smallerPlan[Number(o.phase.slice('two-stream-duty-'.length))];
      o.descriptor={name:o.phase,duty,attentionRowsPerDuty:rows,normX:false,reuseAttentionQueryStorage:true};
      o.demand=twoStreamPhaseDemand({...o.descriptor,name:'two-stream-duty'});
    }
  }
  assert.equal(acceptResidentTwoStream(queryReuse).ok,true,'full positive effective query reuse fixture');
  const unprovedProjection=structuredClone(queryReuse);unprovedProjection.requested.reuseAttentionProjectionStorage=true;
  assert.equal(acceptResidentTwoStream(unprovedProjection).ok,false,'requested projection reuse without effective owned row prefixes cannot certify lower NEW');
  const projected=structuredClone(queryReuse);
  Object.assign(projected.requested,{reuseDeadTriplaneStorage:true,reuseAttentionResidualStorage:true,reuseAttentionProjectionStorage:true});
  Object.assign(projected.twoStream,{reuseDeadTriplaneStorage:true,reuseAttentionResidualStorage:true,reuseAttentionProjectionStorage:true});
  const pt=projected.twoStream.cooperative.adapterTelemetry;
  Object.assign(pt,{reuseDeadTriplaneStorage:true,reuseAttentionResidualStorage:true,reuseAttentionProjectionStorage:true});
  pt.storageReuse=smallerPlan.filter(d=>d.kind==='fuse-residual-norm').map(d=>({block:d.block,afterDutyIndex:d.dutyIndex,
    bytes:113246208,source:'owned-work-inventory-after-gpu-prefix'}));
  const completedProjection=smallerPlan.filter(d=>d.kind==='fuse-attention-linear-range'&&d.rangeIndex===d.rangeCount-1);
  pt.attentionResidualReuse=completedProjection.map(d=>({block:d.block,afterDutyIndex:d.dutyIndex,bytes:113246208,source:'owned-projected-attention-after-gpu-prefix'}));
  pt.attentionProjectionReuse=completedProjection.map(d=>({block:d.block,afterDutyIndex:d.dutyIndex,bytes:113246208,totalRows:27648,
    rowCapacity:128,scratchBytes:1048576,source:'owned-attention-result-after-each-complete-row-last-read'}));
  for(const o of projected.phaseObservations)if(o.phase.startsWith('two-stream-duty-')){
    Object.assign(o.descriptor,{reuseDeadTriplaneStorage:true,reuseAttentionResidualStorage:true,reuseAttentionProjectionStorage:true,linearRowsPerDuty:128});
    o.demand=twoStreamPhaseDemand({...o.descriptor,name:'two-stream-duty'});
  }
  assert.equal(acceptResidentTwoStream(projected).ok,true,'complete positive owned projection rows with actual reduced NEW');
  for(const mutate of [r=>r.twoStream.reuseAttentionProjectionStorage=false,r=>r.twoStream.cooperative.adapterTelemetry.attentionProjectionReuse.pop(),
    r=>r.twoStream.cooperative.adapterTelemetry.attentionProjectionReuse[0].afterDutyIndex--,
    r=>r.twoStream.cooperative.adapterTelemetry.attentionProjectionReuse[0].scratchBytes=0,
    r=>r.twoStream.cooperative.adapterTelemetry.attentionProjectionReuse[0].source='requested-only',
    r=>r.phaseObservations.find(o=>o.phase.startsWith('two-stream-duty-')).descriptor.reuseAttentionProjectionStorage=false,
    r=>r.phaseObservations.find(o=>o.phase.startsWith('two-stream-duty-')).descriptor.linearRowsPerDuty=1,
    r=>r.phaseObservations.find(o=>o.phase.startsWith('two-stream-duty-')).demand.requiredBytes=0]){
    const bad=structuredClone(projected);mutate(bad);assert.equal(acceptResidentTwoStream(bad).ok,false);
  }
  for(const mutate of [r=>delete r.twoStream.reuseAttentionQueryStorage,
    r=>r.twoStream.cooperative.adapterTelemetry.reuseAttentionQueryStorage=false,
    r=>r.requested.reuseAttentionQueryStorage='true',
    r=>delete r.phaseObservations.find(o=>o.phase.startsWith('two-stream-duty-')).descriptor.reuseAttentionQueryStorage,
    r=>r.phaseObservations.find(o=>o.phase.startsWith('two-stream-duty-')).descriptor.reuseAttentionQueryStorage=false,
    r=>r.phaseObservations.find(o=>o.phase.startsWith('two-stream-duty-')).demand.requiredBytes=0]){
    const drift=structuredClone(queryReuse);mutate(drift);
    assert.equal(acceptResidentTwoStream(drift).ok,false,'requested-only/partial/default effective mode or zero demand cannot claim lower NEW');
  }
  const rematerialized=structuredClone(smaller),rematerializedNames=[...smallerNames];
  rematerializedNames.splice(rematerializedNames.indexOf('two-stream-final'),0,
    'two-stream-embedding-rematerialize-weights','two-stream-embedding-rematerialize-rearrange');
  assert.deepEqual(residentTwoStreamExpectedPhases(rows,true),rematerializedNames,
    'explicit rematerialization adds both complete guarded phases without removing model work');
  rematerialized.requested.rematerializeTokenizerEmbedding=true;
  rematerialized.twoStream.rematerializeTokenizerEmbedding=true;
  rematerialized.twoStream.weightPhases.push({status:'completed-retired'});
  for(const index of [0,22])rematerialized.twoStream.weightPhases[index].tensorNames=['tokenizer.embeddings'];
  rematerialized.twoStream.setupTokenizerEmbeddingRetirement={status:'retired',afterGroup:'setup',bufferCount:2,logicalBytes:113246228};
  rematerialized.twoStream.tokenizerEmbeddingRetirement={status:'retired',afterGroup:'final',bufferCount:2,logicalBytes:113246228};
  rematerialized.expectedPhaseOrder=rematerializedNames;rematerialized.phaseObservations=rematerializedNames.map(observation);
  for(const name of ['two-stream-embedding-rematerialize-weights','two-stream-embedding-rematerialize-rearrange']){
    const o=rematerialized.phaseObservations.find(o=>o.phase===name);
    o.descriptor={name,tensors:name.endsWith('weights')?[{name:'tokenizer.embeddings',dtype:1,size:56623104,shape:[3,1024,96,96]}]:[]};
    o.demand=twoStreamPhaseDemand(o.descriptor);
  }
  rematerialized.requested.processBudgetBytes=2147483648;
  for(const o of rematerialized.phaseObservations)o.host.hostFreeBytes=2147483648;
  assert.equal(acceptResidentTwoStream(rematerialized).ok,true);
  const setupReuse=structuredClone(rematerialized);setupReuse.requested.reuseTokenizerSetupStorage=true;
  assert.equal(acceptResidentTwoStream(setupReuse).ok,false,'requested setup reuse cannot certify default allocating route');
  setupReuse.twoStream.reuseTokenizerSetupStorage=true;
  const setupTelemetry=setupReuse.twoStream.cooperative.adapterTelemetry;
  setupTelemetry.reuseTokenizerSetupStorage=true;
  setupTelemetry.tokenizerSetupStorageReuse={status:'completed-prefix',afterDutyIndex:0,bytesPerBuffer:113246208,
    reusedTriplaneBuffers:2,newTriplaneBuffers:1,
    source:'owned-tokenizer-after-groupnorm-read; owned-groupnorm-after-transpose-read'};
  const setupObservation=setupReuse.phaseObservations.find(o=>o.phase==='two-stream-duty-0');
  setupObservation.descriptor={name:'two-stream-duty',duty:{kind:'setup'},
    reuseTokenizerSetupStorage:true,rematerializeTokenizerEmbedding:true};
  setupObservation.demand=twoStreamPhaseDemand(setupObservation.descriptor);
  assert.equal(acceptResidentTwoStream(setupReuse).ok,true);
  for(const mutate of [r=>delete r.twoStream.reuseTokenizerSetupStorage,
    r=>r.twoStream.cooperative.adapterTelemetry.reuseTokenizerSetupStorage=false,
    r=>r.twoStream.cooperative.adapterTelemetry.tokenizerSetupStorageReuse.status='encoded',
    r=>r.twoStream.cooperative.adapterTelemetry.tokenizerSetupStorageReuse.afterDutyIndex=1,
    r=>r.twoStream.cooperative.adapterTelemetry.tokenizerSetupStorageReuse.bytesPerBuffer--,
    r=>r.twoStream.cooperative.adapterTelemetry.tokenizerSetupStorageReuse.newTriplaneBuffers=0,
    r=>r.phaseObservations.find(o=>o.phase==='two-stream-duty-0').descriptor.reuseTokenizerSetupStorage=false,
    r=>r.phaseObservations.find(o=>o.phase==='two-stream-duty-0').descriptor.rematerializeTokenizerEmbedding=false,
    r=>r.phaseObservations.find(o=>o.phase==='two-stream-duty-0').demand.requiredBytes=0,
    r=>r.phaseObservations.find(o=>o.phase==='two-stream-duty-0').demand.workGpuBytes=0]){
    const drift=structuredClone(setupReuse);mutate(drift);assert.equal(acceptResidentTwoStream(drift).ok,false);
  }
  for(const mutate of [r=>delete r.twoStream.rematerializeTokenizerEmbedding,
    r=>r.twoStream.rematerializeTokenizerEmbedding=false,
    r=>delete r.twoStream.tokenizerEmbeddingRetirement,
    r=>r.twoStream.tokenizerEmbeddingRetirement.afterGroup='setup',
    r=>r.twoStream.tokenizerEmbeddingRetirement.logicalBytes--,
    r=>delete r.twoStream.setupTokenizerEmbeddingRetirement,
    r=>r.twoStream.setupTokenizerEmbeddingRetirement.bufferCount=0,
    r=>r.twoStream.weightPhases.pop(),
    r=>r.twoStream.weightPhases[22].tensorNames=['other.tensor'],
    r=>r.phaseObservations.find(o=>o.phase==='two-stream-embedding-rematerialize-weights').demand.requiredBytes=0,
    r=>r.phaseObservations.find(o=>o.phase==='two-stream-embedding-rematerialize-rearrange').demand.workGpuBytes=0,
    r=>r.phaseObservations.splice(r.phaseObservations.findIndex(o=>o.phase==='two-stream-embedding-rematerialize-weights'),1),
    r=>r.requested.rematerializeTokenizerEmbedding='true']){
    const drift=structuredClone(rematerialized);mutate(drift);assert.equal(acceptResidentTwoStream(drift).ok,false);
  }
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

