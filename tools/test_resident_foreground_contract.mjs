import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {createWebGpuForegroundService} from '@kaminos/webgpu-inference-kit';
import {validateResidentForegroundRequest,acceptResidentForeground} from './resident_foreground_service.mjs';
import {evaluatePhaseHostHeadroom} from './memory_admission.mjs';

// Synthetic host/process/config fixtures test local refusal policy. Actual
// submission receipts come from the locked common Kit, not a parallel schema.
const requested={throughFullModel:true,processBudgetBytes:2147483648,cpuBytes:268435456,gpuBytes:1073741824,totalBytes:1342177280,
  hostHeadroomPolicy:'darwin-available-memory-estimate-v1',foreground:{repoRoot:'/synthetic/kaminos',revision:'1'.repeat(40),grid:32,sourceDepth:0.125,rendererCpuBytes:20971520,warmupFrames:200}};
validateResidentForegroundRequest(requested);
validateResidentForegroundRequest({...requested,processBudgetBytes:2415919104});
for(const processBudgetBytes of [undefined,null,'2415919104',NaN,2415919105,3*1024**3])
  assert.throws(()=>validateResidentForegroundRequest({...requested,processBudgetBytes}));
for(const change of [r=>r.throughFullModel=false,r=>r.processBudgetBytes++,r=>r.cpuBytes++,r=>r.gpuBytes++,r=>r.totalBytes++,
  r=>r.hostHeadroomPolicy='raw-free-pages-v0',r=>r.foreground.repoRoot='relative',r=>r.foreground.revision='main',
  r=>r.foreground.grid=160,r=>r.foreground.sourceDepth=0.006,r=>r.foreground.rendererCpuBytes=0,r=>r.foreground.warmupFrames=0]){
  const bad=structuredClone(requested);change(bad);assert.throws(()=>validateResidentForegroundRequest(bad));
}
const host={source:'live-macos',platform:'darwin',hostname:'fixture',model:'Mac14,9',processor:'Apple M2 Pro',hostTotalBytes:17179869184,
  hostFreeBytes:100,hostMemoryPressureFreePercent:78,startedAtUnixMs:1000,observedAt:new Date(1002).toISOString(),observerErrors:[],
  availableMemory:{bytes:6442450944,source:'process.availableMemory/uv_get_available_memory',nodeVersion:'v25.9.0',libuvVersion:'1.52.1',executable:'/actual/node',observedAtUnixMs:1001}};
const processEvidence={coverage:'sampled-owned-process-tree',sampleCount:1,
  lastObservation:{status:'observed',runId:'fixture-run',rootPid:1,atUnixMs:1002,sampledAggregatePhysicalFootprintBytes:100000000},
  freshness:{route:'new-probe-after-request',requestedAtUnixMs:1001,probeAtUnixMs:1002,observationIndex:1}};
const admission={verdict:'admitted',phaseRequestedAtUnixMs:1000,host,process:processEvidence,demand:{requiredBytes:requested.totalBytes},
  hostHeadroom:evaluatePhaseHostHeadroom({host,requiredBytes:requested.totalBytes,policy:requested.hostHeadroomPolicy,requestedAtUnixMs:1000})};
const queue={submit(){}};const device={queue};
const service=createWebGpuForegroundService({routeId:'synthetic-native-foreground-contract',device,queue});
const run=await service.beginRun('fixture-run');
const frames=[];
for(let i=1;i<=2;i++){
  const handle=service.request({requestId:'frame-'+i,run(ctx){assert.equal(ctx.device,device);assert.equal(ctx.queue,queue);ctx.submit([{}]);
    return {renderer:'ordinary-volume',status:'submitted',frameCount:i,simStepCount:i,sceneFrameCount:i};}});
  await run.foregroundOpportunities.serviceAtBoundary({invocationId:'fixture',boundaryId:'fixture',dutyId:'fixture-'+i,phase:'gpu-prefix',position:'before-encode'});
  frames.push(await handle.completion);
}
const serviceReport=await run.finish();await service.dispose();
const snapshot={sameDevice:true,sameQueue:true,state:{active:true,error:null,controls:{resolution:32,emitterSourceDepth:0.125}}};
const report={requested,runId:'fixture-run',rootPid:1,ownedBrowserPid:123,source:{hostname:'fixture',revision:'2'.repeat(40)},
  processObservation:{maxFootprintBytes:requested.processBudgetBytes},
  hostPressureGuard:{status:'observed',policy:requested.hostHeadroomPolicy,pressureStopFreePercent:24},
  foregroundRun:{status:'finished',runId:'fixture-run',sameDevice:true,sameQueue:true,frames,serviceReport},
  foreground:{source:{repoRoot:requested.foreground.repoRoot,revision:requested.foreground.revision,trackedClean:true},
    service:{route:'owned-actual-kaminos-handler.v0',ownedPid:124,effective:{source:{repoRoot:requested.foreground.repoRoot,commit:requested.foreground.revision,dirty:false}}},admission,
    compositionModule:{sha256:'3'.repeat(64),servedSha256:'3'.repeat(64),sourceRevision:'2'.repeat(40)},
    before:snapshot,after:snapshot,warmup:{completedFrameCount:200},pageErrors:[],
    captures:{before:{bytes:10,sha256:'before'},after:{bytes:10,sha256:'after'}},deviceRetirement:{route:'exact-owned-browser-exit',ownedPid:123},
    displayAssertion:{route:'owned-caffeinate-process-lifetime',ownedPid:125,aliveAtCapture:true}},
  cleanup:{browser:{exitObserved:true,ownedPid:123},foregroundService:{exitObserved:true,ownedPid:124},displayAssertion:{exitObserved:true,ownedPid:125}}};
assert.deepEqual(acceptResidentForeground(report).errors,[]);
const enlarged=structuredClone(report);
enlarged.requested.processBudgetBytes=2415919104;
enlarged.processObservation.maxFootprintBytes=2415919104;
assert.deepEqual(acceptResidentForeground(enlarged).errors,[],'explicit 2.25-GiB process envelope preserves original live-render guards');
for(const change of [r=>delete r.processObservation.maxFootprintBytes,r=>r.processObservation.maxFootprintBytes=2147483648,
  r=>r.processObservation.maxFootprintBytes='2415919104',r=>r.requested.cpuBytes++,r=>r.requested.gpuBytes++,
  r=>r.requested.totalBytes++,r=>r.hostPressureGuard.pressureStopFreePercent=23,
  r=>r.foreground.admission.process.lastObservation.sampledAggregatePhysicalFootprintBytes=2415919104]){
  const bad=structuredClone(enlarged);change(bad);assert.equal(acceptResidentForeground(bad).ok,false,'changed process or other guard cannot close');
}
// The new complete-consumer branch must constructively accept the same locked
// Kit receipts, not merely reject broken raw references or changed viewports.
const transferred=structuredClone(report);
transferred.modelConnection={status:'connected',effectiveNetworkEnabled:false,hostReferencesPreserved:true,
 targetBefore:'owned-target',targetAfter:'owned-target',urlBefore:'http://owned/',urlAfter:'http://owned/',
 viewportBefore:{width:1280,height:960},viewportAfter:{width:1280,height:960},
 geometryBefore:{width:1280,height:960,deviceScaleFactor:1},geometryAfter:{width:1280,height:960,deviceScaleFactor:1}};
const raw={schema:'sf3d.native-foreground-run-reference.v0',status:'retained',runId:report.runId,
 path:'/owned/complete-foreground.json',bytes:4096,sha256:'4'.repeat(64)};
const auxiliary=(requiredBytes,descriptor)=>({...structuredClone(admission),descriptor,demand:{requiredBytes},
 hostHeadroom:evaluatePhaseHostHeadroom({host,requiredBytes,policy:requested.hostHeadroomPolicy,requestedAtUnixMs:1000})});
transferred.foregroundRunReference=structuredClone(raw);
transferred.foreground.resultModule={sourceRevision:report.source.revision,sha256:'5'.repeat(64),servedSha256:'5'.repeat(64)};
transferred.foreground.resultEvidence={status:'retained',reference:structuredClone(raw),admission:auxiliary(4*raw.bytes)};
transferred.foreground.resultRestoration=auxiliary(6*raw.bytes);
transferred.foreground.captureAdmission=auxiliary(7*1280*960*4,{width:1280,height:960});
assert.deepEqual(acceptResidentForeground(transferred).errors,[],'complete original-geometry transferred consumer is accepted');
for(const mutate of [r=>r.foregroundRunReference.sha256='0'.repeat(64),r=>r.foreground.resultModule.servedSha256='wrong',
 r=>r.foreground.resultEvidence.admission.process.lastObservation.runId='stale',
 r=>r.foreground.resultRestoration.process.freshness.requestedAtUnixMs=999,
 r=>r.foreground.resultRestoration.demand.requiredBytes--,r=>r.foreground.captureAdmission.verdict='refused',
 r=>r.modelConnection.geometryAfter.width=800,r=>r.modelConnection.targetAfter='other']){
 const bad=structuredClone(transferred);mutate(bad);assert.equal(acceptResidentForeground(bad).ok,false,'transferred consumer authority cannot silently drift');
}
for(const hazard of ['viewport','raw-reference']){
 const bad=structuredClone(report);
 if(hazard==='viewport')bad.modelConnection={status:'connected',effectiveNetworkEnabled:false,
  viewportBefore:{width:1280,height:960},viewportAfter:{width:800,height:600},
  geometryBefore:{width:1280,height:960,deviceScaleFactor:1},geometryAfter:{width:800,height:600,deviceScaleFactor:1}};
 else{bad.foreground.resultEvidence={status:'failed'};bad.foregroundRunReference={schema:'sf3d.native-foreground-run-reference.v0',status:'retained',runId:'stale'};}
 assert.equal(acceptResidentForeground(bad).ok,false,'changed viewport or unverified raw handoff cannot close: '+hazard);
}
for(const change of [r=>r.foreground.source.revision='other',r=>r.foreground.service.route='copy',r=>r.foreground.admission.verdict='refused',
  r=>r.foreground.admission.process.lastObservation.runId='stale',r=>r.foreground.admission.process.freshness.requestedAtUnixMs=999,
  r=>r.foreground.admission.process.lastObservation.sampledAggregatePhysicalFootprintBytes=2147483648,
  r=>r.foreground.admission.host.availableMemory.source='os.freemem',r=>r.hostPressureGuard.status='failed',r=>r.foreground.admission.host.hostMemoryPressureFreePercent=24,
  r=>r.foreground.compositionModule.servedSha256='wrong',r=>r.foreground.after.sameDevice=false,r=>r.foreground.after.sameQueue=false,
  r=>r.foreground.after.state.controls.resolution=160,r=>r.foreground.after.state.controls.emitterSourceDepth=0.006,
  r=>r.foreground.warmup.completedFrameCount=199,r=>r.foregroundRun.runId='stale',r=>r.foregroundRun.status='failed',r=>r.foregroundRun.frames=[],
  r=>r.foregroundRun.frames[1].result.simStepCount=1,r=>r.foregroundRun.frames[0].submissions=[],r=>r.foregroundRun.frames[0].result.renderer='alternate-volume',
  r=>r.foreground.captures.after.bytes=0,r=>r.foreground.captures.after.sha256='before',r=>r.foreground.deviceRetirement.ownedPid=456,
  r=>r.cleanup.browser.exitObserved=false,r=>r.cleanup.foregroundService.exitObserved=false,r=>r.foreground.pageErrors.push({message:'error'}),
  r=>r.cleanup.displayAssertion.exitObserved=false,r=>r.foreground.displayAssertion.aliveAtCapture=false]){
  const bad=structuredClone(report);change(bad);assert.equal(acceptResidentForeground(bad).ok,false,'false closure path must refuse');
}
// Exercise the actual native fresh admission conjunction, not a second policy.
const native=fs.readFileSync(new URL('./smoke_resident_dino.mjs',import.meta.url),'utf8');
const start=native.indexOf('const freshObservation=async'),end=native.indexOf('\n  server=http.createServer',start);
assert.ok(start>0&&end>start);
const fresh=vm.runInNewContext(native.slice(start,end)+';freshObservation;',{report:{requested,source:{hostname:'fixture'},runId:'fixture-run'},
  process:{pid:1},monitor:{async sample(){return processEvidence;}},observeMacMemory:()=>host,
  execFileSync:(_cmd,args)=>args[1]==='hw.model'?'Mac14,9':'Apple M2 Pro',evaluatePhaseHostHeadroom,throughPostProcessor:true,
  postWitness:{projectFreshProcessEvidence:value=>value}});
assert.equal((await fresh({name:'foreground-initialization'},{requiredBytes:requested.totalBytes},1000)).verdict,'admitted');
processEvidence.lastObservation.sampledAggregatePhysicalFootprintBytes=2147483648;
assert.equal((await fresh({name:'foreground-initialization'},{requiredBytes:requested.totalBytes},1000)).verdict,'refused','identified NEW plus actual observer-inclusive baseline must fit before allocation');
const leaf=fs.mkdtempSync(path.join(os.tmpdir(),'resident-foreground-arguments-')),reportPath=path.join(leaf,'report.json');
try{
  const result=spawnSync(process.execPath,['tools/smoke_resident_dino.mjs','--report',reportPath,'--foreground-repo-root','/synthetic/kaminos'],{encoding:'utf8'});
  assert.equal(result.status,1);const failed=JSON.parse(fs.readFileSync(reportPath,'utf8'));
  assert.equal(failed.status,'failed');assert.match(failed.error.message,/resident foreground requires/);
  assert.equal(failed.cleanup.browser.status,'not-started');assert.equal(fs.existsSync(reportPath+'.model.glb'),false);
}finally{fs.rmSync(leaf,{recursive:true});}
console.log('Resident foreground exact config, actual locked-Kit receipts, fresh NEW+baseline refusal, false-closure and pre-output failure contracts passed');
