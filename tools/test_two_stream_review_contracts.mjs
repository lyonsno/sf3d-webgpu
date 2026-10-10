import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {createTwoStreamAttentionDutyPlan} from '../src/lib/two_stream.js';
import {groupTwoStreamDuties} from '../src/lib/cooperative_two_stream.js';
import {acceptResidentTwoStream,twoStreamPhaseDemand} from './resident_two_stream_acceptance.mjs';
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
  const observation=phase=>({phase,verdict:'admitted',host:{source:'live-macos',hostname:'fixture',model:'Mac14,9',processor:'Apple M2 Pro',observerErrors:[],hostFreeBytes:100},
    demand:{requiredBytes:10},process:{coverage:'sampled-owned-process-tree',lastObservation:{runId:'fixture',rootPid:42,status:'observed',sampledAggregatePhysicalFootprintBytes:10}}});
  const valid={status:'passed',phase:'complete',source:{clean:true,revision:'fixture-head',hostname:'fixture'},
    requested:{revision:'fixture-head',weightsSha256:'fixture-weights',processBudgetBytes:100},
    backend:{vendor:'apple',isFallbackAdapter:false},artifact:{sha256:'fixture-bundle',servedSha256:'fixture-bundle',sourceRevision:'fixture-head',kitVersion:'0.1.53'},
    canonicalSource:{sha256:'fixture-weights',etag:'"fixture-weights"',byteLength:200},
    dino:{weightPhases:Array.from({length:25},()=>({status:'completed-retired'})),loadingReport:{sourceETag:'"fixture-weights"',expectedWeightBytes:200},
      cooperative:{status:'succeeded',schedulingMode:'cooperative',queueCompletionAuthority:'per-gpu-duty-prefix-fence',boundaries:[{completedItems:24,totalItems:24,actualRangeCount:24}]}},
    twoStream:{residentFFN:true,linearRowsPerDuty:128,blocks:4,basicBlocks:12,shape:[3,1024,96,96],weightPhases:Array.from({length:23},()=>({status:'completed-retired'})),
      loadingReport:{sourceETag:'"fixture-weights"',expectedWeightBytes:200},cooperative:{status:'succeeded',schedulingMode:'cooperative',
        routeId:'sf3d.image-to-mesh.webgpu-local.v0',manifestId:'sf3d.two-stream-attention-cooperative-boundaries.v0',queueCompletionAuthority:'per-gpu-duty-prefix-fence',
        boundaries:[{boundaryId:'two-stream-attention-duties',completedItems:3350,totalItems:3350,actualRangeCount:3350}],
        adapterTelemetry:{residentFFN:true,linearRowsPerDuty:128,dutyGranularity:'attention-tile',declaredDutyCount:3350}}},
    phaseObservations:names.map(observation),expectedPhaseOrder:names,runId:'fixture',rootPid:42,validationError:null,budget:{cpu:{liveBytes:0},gpu:{liveBytes:0}},
    processObservation:{status:'observed',coverage:'sampled-owned-process-tree',sampledPeakAggregatePhysicalFootprintBytes:20},cleanup:{browser:{exitObserved:true},server:'closed'},
    evidencePaths:{input,output,twoStreamOutput:backboneOutput},inputTensor:{sha256:hash(ib)},inputConsumed:{sha256:hash(ib)},output:{sha256:hash(ob)},
    twoStreamReadbackBytes:27648*1024*4,twoStreamOutput:{sha256:hash(fs.readFileSync(backboneOutput))}};
  const absent=structuredClone(valid);absent.phaseObservations=absent.phaseObservations.slice(0,27);absent.expectedPhaseOrder=names.slice(0,27);
  assert.equal(acceptResidentTwoStream(absent).ok,false,'zero backbone observations cannot pass via a self-reported shortened expected graph');
  assert.equal(acceptResidentTwoStream(valid).ok,true,'synthetic acceptance-policy fixture only');
  for(const mutate of [
    r=>r.expectedPhaseOrder.pop(),r=>r.phaseObservations.pop(),r=>r.phaseObservations.splice(28,1),
    r=>r.twoStream.cooperative.routeId='other.model',r=>r.twoStream.cooperative.manifestId='other.manifest',
    r=>r.twoStream.cooperative.adapterTelemetry.linearRowsPerDuty=1,
    r=>r.twoStream.cooperative.adapterTelemetry.dutyGranularity='stage',
    r=>r.twoStream.cooperative.boundaries[0].boundaryId='other.boundary',
  ]){const drift=structuredClone(valid);mutate(drift);assert.equal(acceptResidentTwoStream(drift).ok,false);}
}finally{fs.rmSync(leaf,{recursive:true});}
console.log('canonical complete backbone observations, effective route/config and pre-output import failure contracts');

