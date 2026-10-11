import assert from 'node:assert/strict';
import {createPostProcessorChannelDutyPlan,describePostProcessorDutyDemand} from '../src/lib/cooperative_post_processor.js';
import {evaluatePhaseHostHeadroom} from './memory_admission.mjs';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {residentTwoStreamExpectedPhases} from './resident_two_stream_acceptance.mjs';
const subject=await import('./resident_post_processor_acceptance.mjs').catch(error=>{
  if(error.code==='ERR_MODULE_NOT_FOUND'&&error.url===new URL('./resident_post_processor_acceptance.mjs',import.meta.url).href)return {};
  throw error;
});
assert.equal(typeof subject.acceptPostProcessorCompletion,'function',
  'a completed backbone must not close the requested postprocessor consumer without its full output and effective graph');
assert.equal(subject.acceptPostProcessorCompletion({requested:{throughPostProcessor:true}}).ok,false);
const plan=createPostProcessorChannelDutyPlan(16),host={source:'live-macos',hostname:'host',
  model:'Mac14,9',processor:'Apple M2 Pro',observerErrors:[],hostFreeBytes:1000000000};
const descriptors=[{name:'post-processor-output-allocation',
  duty:{kind:'output-allocation',workGpuBytes:70778880},tensors:[]}];
for(const duty of plan.duties){
  if(duty.kind==='conv-range'&&duty.rangeIndex===0)descriptors.push({
    name:'post-processor-plane-'+duty.plane+'-'+duty.stageId,
    tensors:['weight','bias'].map(s=>({name:'post_processor.upsample.'+(2*duty.layerIndex)+'.'+s,size:4,dtype:0})),
  });
  descriptors.push({name:'post-processor-duty-'+duty.dutyIndex,
    duty:{...duty,...describePostProcessorDutyDemand(duty)},tensors:[]});
}
descriptors.push({name:'post-processor-output',tensors:[]});
const report={source:{hostname:'host'},requested:{throughPostProcessor:true,throughBackbone:true,
  postChannelsPerDuty:16,processBudgetBytes:1000000000},runId:'current',rootPid:42,
  canonicalSource:{etag:'"source"',byteLength:200},
  postProcessor:{shape:[3,40,384,384],planes:3,convolutionsPerPlane:4,channelsPerDuty:16,
    loadingReport:{sourceETag:'"source"',expectedWeightBytes:200},
    weightPhases:Array.from({length:12},(_,i)=>({status:'completed-retired',
      tensorNames:['weight','bias'].map(s=>'post_processor.upsample.'+(2*(i%4))+'.'+s)})),
    cooperative:{status:'succeeded',schedulingMode:'cooperative',queueCompletionAuthority:'per-gpu-duty-prefix-fence',
      routeId:'sf3d.image-to-mesh.webgpu-local.v0',manifestId:'sf3d.post-processor-channel-cooperative-boundaries.v0',
      boundaries:[{boundaryId:'post-processor-triplane-channel-ranges',completedItems:702,totalItems:702,actualRangeCount:702}],
      adapterTelemetry:{channelsPerDuty:16,dutyGranularity:'channel-range',residentWork:true,stageDuties:plan.duties}}},
  postProcessorReadbackBytes:70778880,postProcessorOutput:{bytes:70778880,finite:17694720,nonzero:17694720},
  phaseObservations:descriptors.map(descriptor=>{
    const demand=subject.postProcessorPhaseDemand(descriptor,16);
    return {phase:descriptor.name,descriptor,demand,host,verdict:'admitted',phaseRequestedAtUnixMs:1,
      hostHeadroom:evaluatePhaseHostHeadroom({host,requiredBytes:demand.requiredBytes}),
      process:{coverage:'sampled-owned-process-tree',sampleCount:3,freshness:{route:'new-probe-after-request',
        requestedAtUnixMs:2,probeAtUnixMs:3,observationIndex:3},lastObservation:{status:'observed',runId:'current',rootPid:42,
        atUnixMs:3,sampledAggregatePhysicalFootprintBytes:100}}};
  }),
};
assert.equal(subject.acceptPostProcessorCompletion(report).ok,true,
  'synthetic local policy fixture only, not native/backend numerical evidence');
for(const mutate of [
  r=>delete r.postProcessor.cooperative.routeId,r=>r.postProcessor.cooperative.routeId='other',
  r=>delete r.postProcessor.cooperative.manifestId,r=>r.postProcessor.cooperative.manifestId='other',
  r=>delete r.postProcessor.cooperative.boundaries[0].boundaryId,
  r=>r.postProcessor.cooperative.boundaries[0].boundaryId='other',
  r=>r.requested.postChannelsPerDuty=32,r=>r.requested.throughBackbone=false,
  r=>delete r.postProcessor,r=>r.postProcessor.shape=[3,40,96,96],
  r=>r.postProcessor.cooperative.schedulingMode='disabled',
  r=>r.postProcessor.cooperative.queueCompletionAuthority='submit-only',
  r=>r.postProcessor.cooperative.adapterTelemetry.residentWork=false,
  r=>r.postProcessor.cooperative.adapterTelemetry.stageDuties.pop(),
  r=>r.postProcessor.cooperative.adapterTelemetry.stageDuties[1].channelStart=1,
  r=>r.postProcessor.weightPhases.pop(),r=>r.postProcessor.weightPhases[0].tensorNames=['cached'],
  r=>r.postProcessor.weightPhases[0].status='cleanup-unresolved',
  r=>r.postProcessor.loadingReport.sourceETag='"other"',
  r=>r.phaseObservations.pop(),r=>r.phaseObservations[3].descriptor.duty.workGpuBytes=0,
  r=>r.phaseObservations[0].process.lastObservation.runId='stale',
  r=>r.phaseObservations[0].process.freshness.requestedAtUnixMs=0,
  r=>delete r.phaseObservations[0].process.freshness.observationIndex,
  r=>r.phaseObservations[0].process.lastObservation.sampledAggregatePhysicalFootprintBytes=1000000000,
  r=>r.phaseObservations[0].host.hostFreeBytes=0,
  r=>r.postProcessorReadbackBytes=1,r=>r.postProcessorOutput.nonzero=0,
  r=>r.postProcessorOutput.finite=1,
]){
  const changed=structuredClone(report);mutate(changed);
  assert.equal(subject.acceptPostProcessorCompletion(changed).ok,false);
}
assert.equal(subject.acceptNativeResidentPostProcessor({status:'passed',phase:'complete'}).ok,false,
  'failure before primary output must still have a refusing terminal judge');
assert.throws(()=>subject.inspectPostProcessorOutput(new URL(import.meta.url).pathname),/partial/);
const additive=structuredClone(report);
for(const phase of additive.phaseObservations){phase.demand.annotation='new compatible diagnostic';phase.descriptor.extra='unrelated';}
assert.equal(subject.acceptPostProcessorCompletion(additive).ok,true,'additive diagnostics do not reinterpret new backing');
const historical={processes:{'old:1':{pid:1}},unavailableProcessObservations:[{pid:2}],
  ...report.phaseObservations[0].process,rawPath:'all-attempts.jsonl',summaryPath:'complete-history.json',
  sampledPeakAggregatePhysicalFootprintBytes:345,transport:{exitStatus:0},newDiagnostic:'preserved'};
const projected=subject.projectFreshProcessEvidence(historical);
const {processes:history,unavailableProcessObservations:missing,...expectedProjection}=historical;
assert.deepEqual(projected,expectedProjection);
assert.ok(historical.processes&&historical.unavailableProcessObservations,'projection does not mutate retained lifetime history');
const backboneNames=residentTwoStreamExpectedPhases(32),postNames=subject.postProcessorExpectedPhases(16);
const whole={requested:{attentionRowsPerDuty:32},expectedPhaseOrder:[...backboneNames,...postNames],
  phaseObservations:[...backboneNames,...postNames].map(phase=>({phase}))};
const prefix=subject.residentBackbonePrefix(whole);
assert.deepEqual(prefix.expectedPhaseOrder,backboneNames);
assert.deepEqual(prefix.phaseObservations.map(p=>p.phase),backboneNames);
assert.equal(whole.phaseObservations.length,backboneNames.length+postNames.length,'the whole downstream plan remains separately checked');
const rematerializedNames=residentTwoStreamExpectedPhases(32,true);
const rematerializedWhole={requested:{attentionRowsPerDuty:32,rematerializeTokenizerEmbedding:true},
  expectedPhaseOrder:[...rematerializedNames,...postNames],
  phaseObservations:[...rematerializedNames,...postNames].map(phase=>({phase}))};
const rematerializedPrefix=subject.residentBackbonePrefix(rematerializedWhole);
assert.deepEqual(rematerializedPrefix.expectedPhaseOrder,rematerializedNames);
assert.deepEqual(rematerializedPrefix.phaseObservations.map(p=>p.phase),rematerializedNames,
  'complete extra guarded tokenizer phases remain in the backbone prefix rather than cutting downstream output');
assert.equal(subject.validateCompleteReadbackChunk({offset:0,bytes:4,receivedBytes:0,totalBytes:8}),4);
assert.equal(subject.validateCompleteReadbackChunk({offset:4,bytes:4,receivedBytes:4,totalBytes:8}),8);
for(const args of [{offset:0,bytes:0,receivedBytes:0,totalBytes:8},
  {offset:0,bytes:3,receivedBytes:0,totalBytes:8},{offset:0,bytes:4,receivedBytes:4,totalBytes:8},
  {offset:4,bytes:4,receivedBytes:0,totalBytes:8},{offset:8,bytes:4,receivedBytes:8,totalBytes:8},
  {offset:NaN,bytes:4,receivedBytes:0,totalBytes:8},
  {offset:0,bytes:524292,receivedBytes:0,totalBytes:70778880}])
  assert.throws(()=>subject.validateCompleteReadbackChunk(args),/readback chunk/);
const leaf=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-post-early-failure-'));
try{
  const output=path.join(leaf,'report.json');
  assert.throws(()=>execFileSync(process.execPath,['tools/smoke_resident_dino.mjs','--through-backbone',
    '--through-postprocessor','--report',output],{stdio:'pipe'}));
  const failure=JSON.parse(fs.readFileSync(output));
  assert.equal(failure.status,'failed');assert.equal(failure.error.lastTrustworthyPhase,'arguments');
  assert.equal(failure.verdict.ok,false);
}finally{fs.rmSync(leaf,{recursive:true});}
console.log('PASS complete postprocessor demand, 29 false-closure paths, additive diagnostics and durable early failure');
