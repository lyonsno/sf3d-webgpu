import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {prepareForegroundDependency} from './foreground_dependency_source.mjs';
import {phaseHostHeadroomIsAdmitted} from './memory_admission.mjs';
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
export function validateResidentForegroundRequest(requested){
  if(!requested?.foreground)return;
  const f=requested.foreground;
  if(!requested.throughFullModel||requested.hostHeadroomPolicy!=='darwin-available-memory-estimate-v1'||requested.processBudgetBytes!==2147483648||
    requested.cpuBytes!==268435456||requested.gpuBytes!==1073741824||requested.totalBytes!==1342177280)
    throw Error('resident foreground requires explicit complete model and unchanged available-memory diagnostic guards');
  if(!path.isAbsolute(f.repoRoot??'')||!(/^[a-f0-9]{40}$/).test(f.revision??'')||f.grid!==32||f.sourceDepth!==0.125||f.rendererCpuBytes!==20971520||f.warmupFrames!==200)
    throw Error('explicit source-pinned previously exercised grid32/depth0.125/20MiB/200-frame foreground variant required; no default substitution');
}
export async function prepareResidentForeground({requested,outputDir}){
  validateResidentForegroundRequest(requested);
  const f=requested.foreground,git=args=>execFileSync('git',args,{cwd:f.repoRoot,encoding:'utf8'}).trim();
  const source={repoRoot:f.repoRoot,revision:git(['rev-parse','HEAD']),trackedClean:git(['status','--porcelain','--untracked-files=no'])===''};
  if(source.revision!==f.revision||!source.trackedClean)throw Error('exact clean tracked foreground source required');
  const html=fs.readFileSync(path.join(f.repoRoot,'sf3d-elfinblue.html'));
  if(!html.equals(execFileSync('git',['show',f.revision+':sf3d-elfinblue.html'],{cwd:f.repoRoot})))throw Error('foreground route capsule differs from exact source');
  const match=html.toString().match(/content="0;\s*url=([^"]+)"/);
  if(!match)throw Error('actual existing ordinary route capsule missing');
  const route=new URL(match[1].replaceAll('&amp;','&'),'http://source.invalid'),originalRoute=route.href;
  route.hash='';route.searchParams.delete('settings_preset');route.searchParams.delete('settings_preset_authority');
  route.searchParams.set('volume_resolution',String(f.grid));route.searchParams.set('volume_emitter_source_depth',String(f.sourceDepth));
  route.searchParams.set('volume_quality_reason','mini-explicit-memory-guard-grid-variant');
  route.hash=new URLSearchParams({composition_module_url:'/resident-foreground-browser.mjs'}).toString();
  const volumeSource=fs.readFileSync(path.join(f.repoRoot,'volume-core.js'),'utf8');
  if(!volumeSource.includes('const FLUID_SLOTS_PER_CELL = 4;')||!volumeSource.includes('const VOLUME_VERTICAL_DOMAIN_EXTENT_MULTIPLIER = 2;'))
    throw Error('ordinary initialization source contract changed');
  const requiredCpuBytes=2*f.grid**3*160;
  if(requiredCpuBytes>f.rendererCpuBytes)throw Error('identified foreground initial stores exceed reservation');
  const dependency=await prepareForegroundDependency({repoRoot:f.repoRoot,revision:f.revision,outputDir:path.join(outputDir,'dependencies')});
  return {source,route:'/foreground/index.html'+route.search+route.hash,originalRoute,
    variant:'explicit grid32 and source-depth0.125 diagnostic; not unchanged authored basin or ordinary full-route admission',
    initialization:{requiredCpuBytes,rendererCpuBytes:f.rendererCpuBytes,meaning:'identified initial component stores; opaque/repeated backing separately observed'},
    dependencies:{[dependency.relative]:dependency.receipt},servedSources:{},dependency};
}

export function acceptResidentForeground(report){
  const errors=[],require=(ok,msg)=>{if(!ok)errors.push(msg);},f=report.foreground,run=report.foregroundRun;
  try{validateResidentForegroundRequest(report.requested);}catch(error){errors.push(error.message);}
  require(f?.source?.revision===report.requested?.foreground?.revision&&f.source.trackedClean,'exact actual foreground source missing');
  require(f?.service?.route==='owned-actual-kaminos-handler.v0'&&f.service.effective?.source?.repoRoot===f.source?.repoRoot&&
    f.service.effective.source.commit===f.source?.revision&&f.service.effective.source.dirty===false,'actual foreground handler identity missing');
  const admission=f?.admission,p=admission?.process;
  require(admission?.verdict==='admitted'&&admission.demand.requiredBytes===report.requested?.totalBytes&&phaseHostHeadroomIsAdmitted(report,admission)&&
    admission.host?.hostname===report.source?.hostname&&admission.host.model==='Mac14,9'&&admission.host.processor==='Apple M2 Pro'&&
    !admission.host.observerErrors?.length&&p?.coverage==='sampled-owned-process-tree'&&p.lastObservation?.status==='observed'&&
    p.lastObservation.runId===report.runId&&p.lastObservation.rootPid===report.rootPid&&p.freshness?.route==='new-probe-after-request'&&
    p.freshness.requestedAtUnixMs>=admission.phaseRequestedAtUnixMs&&p.lastObservation.atUnixMs>=p.freshness.requestedAtUnixMs&&
    p.freshness.probeAtUnixMs===p.lastObservation.atUnixMs&&p.freshness.observationIndex===p.sampleCount&&
    p.lastObservation.sampledAggregatePhysicalFootprintBytes+admission.demand.requiredBytes<=report.requested.processBudgetBytes,'fresh guarded foreground initialization missing');
  require(f?.compositionModule?.servedSha256===f?.compositionModule?.sha256&&!!f?.compositionModule?.sha256&&f.compositionModule.sourceRevision===report.source?.revision,
    'actual source-bound resident composition module missing');
  require(f?.before?.sameDevice===true&&f.before.sameQueue===true&&f.after?.sameDevice===true&&f.after.sameQueue===true,'actual same-device/queue foreground missing');
  require(run?.status==='finished'&&run.runId===report.runId&&run.sameDevice===true&&run.sameQueue===true,'foreground run did not finish on exact shared device');
  require(f?.warmup?.completedFrameCount>=report.requested?.foreground?.warmupFrames&&
    [f.before,f.after].every(s=>s?.state?.active===true&&!s.state.error&&s.state.controls?.resolution===report.requested.foreground.grid&&s.state.controls?.emitterSourceDepth===report.requested.foreground.sourceDepth),
    'foreground warmup/effective source controls/error state mismatch');
  const live=run?.frames?.filter(row=>row.runId===report.runId)??[];
  require(live.length>=2,'no actual in-run foreground frames');
  require(live.every(row=>row.status==='completed'&&row.result?.renderer==='ordinary-volume'&&row.result.status==='submitted'&&
    row.submissions?.some(s=>s.submissionStatus==='queue-submit-returned'&&s.commandBufferCount>0)),'actual foreground submission failed or missing');
  require(['frameCount','simStepCount','sceneFrameCount'].every(k=>live.at(-1)?.result?.[k]>live[0]?.result?.[k]),'actual scene/volume/simulation did not advance during complete model');
  require(f?.captures?.before?.bytes>0&&f?.captures?.after?.bytes>0&&f.captures.before.sha256!==f.captures.after.sha256,'actual before/after render capture missing or unchanged');
  require(f?.deviceRetirement?.route==='exact-owned-browser-exit'&&f.deviceRetirement.ownedPid===report.ownedBrowserPid&&report.cleanup?.browser?.exitObserved===true,
    'live shared host retirement lacks exact owned-browser exit');
  require(report.cleanup?.foregroundService?.exitObserved===true&&report.cleanup.foregroundService.ownedPid===f?.service?.ownedPid,'exact owned foreground handler cleanup missing');
  require(!f?.pageErrors?.length,'foreground page exception present');
  return {ok:!errors.length,errors,authority:'complete native model alongside actual same-device ordinary volume and Three submissions; pixels require separate inspection'};
}
