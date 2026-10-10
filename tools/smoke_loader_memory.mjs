#!/usr/bin/env node
// Ordinary tiny native allocation witness, not inference/benchmark or fit admission.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import http from 'node:http';
import {randomUUID, createHash} from 'node:crypto';
import {execFileSync, spawn} from 'node:child_process';
import puppeteer from 'puppeteer-core';
import {weightFixture} from './fixtures/weight_resource_fixture.mjs';
import {startProcessMemory} from './process_memory_guard.mjs';
import {memoryStopAction, stopOwnedBrowser, ownedBrowserArguments} from './owned_browser_stop.mjs';
import {writeJsonReportAtomic} from './json_report_atomic.mjs';
import {acceptLoaderMemoryWitness,acceptStagedTensorWitness} from './loader_memory_witness_acceptance.mjs';
import {prepareCanonicalTensorSource,closeCanonicalSource} from './canonical_tensor_source.mjs';
import {observeMacMemory} from './memory_admission.mjs';
import {PATCH_TENSOR_NAMES,patchPhaseDemand,checkPatchOutput} from './patch_phase_reference.mjs';
import {installForegroundAcquisition} from './foreground_guard_browser.js';
const arg = name => {const i=process.argv.indexOf(name);return i<0?null:process.argv[i+1];};
const root = arg('--repo-root') ? path.resolve(arg('--repo-root')) : null;
const reportPath = path.resolve(arg('--report') ?? path.join(os.tmpdir(),'sf3d-loader-memory-'+randomUUID()+'.json'));
const patchMode=process.argv.includes('--canonical-patch-phase');
const canonicalMode=patchMode||process.argv.includes('--canonical-tensor-unit');
const sharedMode=process.argv.includes('--shared-allowance');
const foregroundRoot=arg('--foreground-repo-root')?path.resolve(arg('--foreground-repo-root')):null;
const report = {schema:'sf3d.loader-native-memory-witness.v0', status:'running', phase:'arguments', receiver:'mini-wake-and-bake-pit-boss',
  route:'sf3d-loader-native-refusal-synthetic-weights-no-inference.v0', reportPath, runId:randomUUID(),
  requested:{repoRoot:root, revision:arg('--expected-revision'), chrome:arg('--chrome'), processBudgetBytes:Number(arg('--process-budget-bytes'))},
  claim:'real loader refusal/cleanup and owned-process stop only; synthetic weights, no learned compute or M2 fit claim'};
report.evidencePaths={report:reportPath, browserLog:reportPath+'.chrome.log', process:reportPath+'.process.jsonl', processRefusal:reportPath+'.refusal-process.jsonl'};
if(canonicalMode){
  report.route='sf3d-canonical-tensor-ranges-no-inference.v0';
  report.claim='exact identified canonical weight conversion/upload/readback and process stopping; no learned compute, phase-fit or full-route claim';
  Object.assign(report.requested,{weightsPath:arg('--weights'),weightsSha256:arg('--expected-weights-sha256'),cpuBytes:Number(arg('--cpu-budget-bytes')),gpuBytes:Number(arg('--gpu-budget-bytes'))});
}
if(patchMode){report.route='sf3d-canonical-patch-embedding.v0';report.claim='selected canonical learned patch projection on M2 Pro with explicit backing demand, live baseline, process stop and raw tensor replay; not full-model fit or production admission';
  Object.assign(report.requested,{inputPath:arg('--input'),inputSha256:arg('--expected-input-sha256')});}
if(foregroundRoot){
  report.requested.foreground={repoRoot:foregroundRoot,revision:arg('--expected-foreground-revision'),grid:Number(arg('--foreground-grid')),rendererCpuBytes:Number(arg('--renderer-cpu-bytes'))};
  report.requested.totalBytes=Number(arg('--combined-budget-bytes'));
  report.claim+='; actual ordinary flame and selected learned operation on its pre-bound device under one explicit allowance; explicit grid variant, not unchanged basin/full-model fit';
}
fs.mkdirSync(path.dirname(reportPath),{recursive:true});
if(sharedMode){report.requested.sharedAllowance=true;report.claim+='; shared explicit allowance contention on two actual owned devices with compute/readback, not host-wide coverage';}
const persist = async () => writeJsonReportAtomic(reportPath,report);
let browser, child, server, monitor, profile, canonical,foregroundPage;
try {
  await persist();
  if(sharedMode&&canonicalMode)throw Error('shared tiny diagnostic and canonical learned/unit routes must be invoked separately');
  if(foregroundRoot&&(!patchMode||![32,48,64,96,128,136,140,160].includes(report.requested.foreground.grid)||
    !Number.isSafeInteger(report.requested.foreground.rendererCpuBytes)||report.requested.foreground.rendererCpuBytes<1||!report.requested.foreground.revision||
    !Number.isSafeInteger(report.requested.totalBytes)||report.requested.totalBytes<1))
    throw Error('foreground requires canonical patch mode and explicit source revision, supported grid and CPU initialization allowance');
  if(!root || !arg('--chrome') || !arg('--expected-revision') || !Number.isSafeInteger(report.requested.processBudgetBytes) || report.requested.processBudgetBytes<1)
    throw Error('explicit --repo-root, --expected-revision, --chrome and --process-budget-bytes required');
  if(canonicalMode){
    report.requested.tensorNames=patchMode?[...PATCH_TENSOR_NAMES]:JSON.parse(arg('--tensor-names')??'null');
    if(!report.requested.weightsPath||!Number.isSafeInteger(report.requested.cpuBytes)||report.requested.cpuBytes<1||!Number.isSafeInteger(report.requested.gpuBytes)||report.requested.gpuBytes<1)
      throw Error('canonical unit requires explicit weights, tensor names and CPU/GPU diagnostic allowances');
    if(patchMode&&(!report.requested.inputPath||!/^[a-f0-9]{64}$/.test(report.requested.inputSha256??'')))throw Error('learned phase requires explicit input path and SHA256');
  }
  report.phase='source-identity';
  const git = args => execFileSync('git',args,{cwd:root,encoding:'utf8'}).trim();
  report.source={revision:git(['rev-parse','HEAD']), clean:git(['status','--porcelain'])==='', repoRoot:root, hostname:os.hostname()};
  if(!report.source.clean || report.source.revision!==report.requested.revision)throw Error('clean exact requested source required');
  const foregroundSources=new Map();
  if(foregroundRoot){
    const fgGit=args=>execFileSync('git',args,{cwd:foregroundRoot,encoding:'utf8'}).trim();
    report.foreground={source:{repoRoot:foregroundRoot,revision:fgGit(['rev-parse','HEAD']),trackedClean:fgGit(['status','--porcelain','--untracked-files=no'])==='',
      untracked:fgGit(['ls-files','--others','--exclude-standard']).split('\n').filter(Boolean)},servedSources:{},visibility:'headed-independent-browser'};
    if(!report.foreground.source.trackedClean||report.foreground.source.revision!==report.requested.foreground.revision)throw Error('exact tracked foreground source required');
    const html=fs.readFileSync(path.join(foregroundRoot,'sf3d-elfinblue.html'),'utf8');
    if(html!==fgGit(['show',report.foreground.source.revision+':sf3d-elfinblue.html'])+'\n'&&html.trim()!==fgGit(['show',report.foreground.source.revision+':sf3d-elfinblue.html']))throw Error('foreground route capsule differs from commit');
    const match=html.match(/content="0;url=([^"]+)"/);if(!match)throw Error('existing ordinary foreground route capsule required');
    const route=new URL(match[1].replaceAll('&amp;','&'),'http://source.invalid');
    report.foreground.originalRoute=route.href;report.foreground.originalGrid=Number(route.searchParams.get('volume_resolution'));
    route.hash='';route.searchParams.delete('settings_preset');route.searchParams.delete('settings_preset_authority');
    route.searchParams.set('volume_resolution',String(report.requested.foreground.grid));
    route.searchParams.set('volume_quality_reason','mini-explicit-memory-guard-grid-variant');
    report.foreground.route='/foreground/index.html'+route.search;
    report.foreground.variant='same recorded ordinary source controls with explicitly changed grid; no accepted immutable basin identity claimed';
    // Pin the source-level large initialization cliff before navigating. The
    // observed source has a tall (2*g^3) domain, sixteen fluid floats/cell,
    // paired front/quench/pressure arrays, sidecar and scalar/texture stores.
    // 160 component bytes/cell covers those identified initialization stores;
    // metadata, repeated transient arrays, padding and private backing remain
    // observed, not certified by this lower initialization calculation.
    const volumeSource=fs.readFileSync(path.join(foregroundRoot,'volume-core.js'),'utf8');
    if(!volumeSource.includes('const FLUID_SLOTS_PER_CELL = 4;')||!volumeSource.includes('const VOLUME_VERTICAL_DOMAIN_EXTENT_MULTIPLIER = 2;'))throw Error('ordinary initialization source contract changed; recalculate before launch');
    report.foreground.initialization={cells:2*report.requested.foreground.grid**3,knownComponentBytesPerCell:160,
      requiredCpuBytes:2*report.requested.foreground.grid**3*160,requestedCpuBytes:report.requested.foreground.rendererCpuBytes,
      meaning:'identified initial component stores only; not an upper bound on opaque/repeated browser backing'};
    if(report.foreground.initialization.requiredCpuBytes>report.requested.foreground.rendererCpuBytes||
      report.requested.foreground.rendererCpuBytes>=report.requested.cpuBytes)throw Error('foreground initialization CPU reservation insufficient before navigation');
    report.foreground.preflight={host:observeMacMemory(),combinedExplicitBytes:report.requested.totalBytes};
    if(report.foreground.preflight.host.source!=='live-macos'||report.foreground.preflight.host.observerErrors.length||
      report.foreground.preflight.host.hostFreeBytes<report.requested.totalBytes)throw Error('current live host headroom insufficient for combined explicit foreground allowance before browser launch');
    Object.assign(report.evidencePaths,{foregroundBefore:reportPath+'.foreground-before.png',foregroundAfter:reportPath+'.foreground-after.png'});
  }
  report.browserExecutable=fs.realpathSync(arg('--chrome'));
  if(/\/Google Chrome\.app\//.test(report.browserExecutable))throw Error('independent browser required; installed GUI Chrome cannot run headlessly');
  report.phase='fixture-preparation';
  if(canonicalMode){canonical=await prepareCanonicalTensorSource({...report.requested,expectedSha256:report.requested.weightsSha256,referenceMode:patchMode?'source-file':'inline-words'});report.canonicalSource=canonical.receipt;await persist();}
  const fixtures={f32:weightFixture().bytes, f16:weightFixture({tensorShapes:new Map([['image_tokenizer.image_mean',[300000]]]),fp16Names:new Set(['image_tokenizer.image_mean'])}).bytes};
  const digest = bytes => createHash('sha256').update(bytes).digest('hex');
  let phaseImage;
  if(patchMode){
    phaseImage=fs.readFileSync(report.requested.inputPath);
    if(digest(phaseImage)!==report.requested.inputSha256)throw Error('phase input SHA256 mismatch');
    if(!phaseImage.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]))||phaseImage.toString('ascii',12,16)!=='IHDR')throw Error('identified PNG phase input required');
    const width=phaseImage.readUInt32BE(16),height=phaseImage.readUInt32BE(20);
    if(width<1||height<1)throw Error('positive source image dimensions required');
    report.inputArtifact={path:fs.realpathSync(report.requested.inputPath),sha256:digest(phaseImage),byteLength:phaseImage.length,width,height};
    report.phaseDemand=patchPhaseDemand(report.canonicalSource,report.inputArtifact);
    if(report.phaseDemand.cpuBytes>report.requested.cpuBytes||report.phaseDemand.gpuBytes>report.requested.gpuBytes)throw Error('explicit phase backing demand exceeds caller allowance');
    Object.assign(report.evidencePaths,{phaseInput:reportPath+'.input.f32',phaseOutput:reportPath+'.output.f32'});
  }
  report.fixtures=Object.fromEntries(Object.entries(fixtures).map(([name,bytes])=>{
    const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
    return [name,{bytes:bytes.length,headerBytes:view.getUint32(12,true),firstTensorBytes:view.getUint32(16+156,true),sha256:digest(bytes),meaning:'synthetic flat-v1 loader fixture; not canonical model weights'}];
  }));
  const sources=new Map();
  const served=['src/lib/weights.js','src/lib/gpu.js','src/lib/loader_memory_budget.js','src/lib/flat_tensor_ranges.js'];
  if(sharedMode)served.push('tools/shared_allowance_browser.js');
  if(foregroundRoot)served.push('tools/foreground_guard_browser.js');
  if(patchMode)served.push('src/lib/sf3d_backbone.js','src/lib/preprocess_core.js','tools/patch_phase_browser.js',
    ...['patch_embed_dinov2','layernorm_vit','attention','linear','linear_gelu','layerscale','activations'].map(n=>'src/shaders/'+n+'.wgsl'));
  for(const relative of served) {
    const bytes=fs.readFileSync(path.join(root,relative)), committed=execFileSync('git',['show',report.source.revision+':'+relative],{cwd:root});
    if(digest(bytes)!==digest(committed))throw Error('served source differs from commit: '+relative);
    sources.set('/'+relative,bytes);
  }
  report.servedSources=Object.fromEntries([...sources].map(([name,bytes])=>[name,digest(bytes)]));
  server=http.createServer((req,res)=>{
    const name=new URL(req.url,'http://localhost').pathname;
    if(foregroundRoot&&name.startsWith('/foreground/')){
      try{
        const relative=decodeURIComponent(name.slice('/foreground/'.length)),file=path.resolve(foregroundRoot,relative);
        if(path.relative(foregroundRoot,file).startsWith('..')||/\.(bin|glb|gltf)$/i.test(file))throw Error('unadmitted model/asset source refused by selected consumer host');
        let data=foregroundSources.get(relative);
        if(!data){
          const committed=execFileSync('git',['show',report.foreground.source.revision+':'+relative],{cwd:foregroundRoot,stdio:['ignore','pipe','pipe']});
          data=fs.readFileSync(file);
          if(digest(data)!==digest(committed))throw Error('foreground served source differs from commit: '+relative);
          foregroundSources.set(relative,data);report.foreground.servedSources[relative]=digest(data);
        }
        const ext=path.extname(file),type={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.png':'image/png','.svg':'image/svg+xml','.jpg':'image/jpeg','.json':'application/json'}[ext]??'application/octet-stream';
        res.writeHead(200,{'Content-Type':type,'Content-Length':data.length,'Cache-Control':'no-store'}).end(data);return;
      }catch(error){res.writeHead(503,{'Content-Type':'text/plain','Cache-Control':'no-store'}).end(error.message);return;}
    }
    if(name==='/'){res.setHeader('content-type','text/html');res.end('<title>SF3D native loader refusal — no inference</title>');return;}
    if(name==='/favicon.ico'){res.writeHead(204).end();return;}
    if(name==='/canonical-weights.bin'&&canonical){canonical.serve(req,res);return;}
    if(patchMode&&name==='/phase-admission'){
      try{
      const host={...observeMacMemory(),model:execFileSync('sysctl',['-n','hw.model'],{encoding:'utf8'}).trim(),processor:execFileSync('sysctl',['-n','machdep.cpu.brand_string'],{encoding:'utf8'}).trim()};
      report.phaseAdmission={authority:'reversible-selected-phase-only',host,processBaseline:report.processObservation?.lastObservation,
        requiredBytes:report.phaseDemand.requiredBytes,demand:report.phaseDemand,
        verdict:host.hostname===report.source.hostname&&host.model==='Mac14,9'&&host.processor==='Apple M2 Pro'&&!host.observerErrors.length&&
          host.hostFreeBytes>=report.phaseDemand.requiredBytes&&report.processObservation?.lastObservation?.status==='observed'&&
          report.processObservation?.coverage==='sampled-owned-process-tree'&&!report.memorySafety?'admitted':'refused'};
      writeJsonReportAtomic(reportPath,report);
      res.writeHead(report.phaseAdmission.verdict==='admitted'?200:409,{'Content-Type':'application/json','Cache-Control':'no-store'}).end(JSON.stringify(report.phaseAdmission));return;
      }catch(error){report.phaseAdmission={verdict:'refused',error:error.message};res.writeHead(409).end(error.message);return;}
    }
    if(patchMode&&name==='/phase-image'){res.writeHead(200,{'Content-Type':'image/png','Cache-Control':'no-store'}).end(phaseImage);return;}
    if(patchMode&&req.method==='POST'&&['/phase-input.f32','/phase-output.f32'].includes(name)){
      const expected=name==='/phase-input.f32'?3145728:5312512,chunks=[];let received=0,failed=false;
      req.on('data',chunk=>{received+=chunk.length;if(received>expected){failed=true;res.writeHead(409).end('phase tensor exceeds exact production shape');req.destroy();}else chunks.push(chunk);});
      req.on('end',()=>{try{if(failed)return;if(received!==expected)throw Error('partial phase tensor');
        fs.writeFileSync(name==='/phase-input.f32'?report.evidencePaths.phaseInput:report.evidencePaths.phaseOutput,Buffer.concat(chunks));res.writeHead(200).end();
      }catch(error){res.writeHead(500).end(error.message);}});return;
    }
    const bytes=name.startsWith('/fixture/')?fixtures[name.slice(9)]:sources.get(name);
    if(!bytes){res.writeHead(404).end();return;}
    const raw=name.endsWith('.wgsl')&&new URL(req.url,'http://localhost').searchParams.has('raw');
    const servedBytes=raw?Buffer.from('export default '+JSON.stringify(bytes.toString())+';'):bytes;
    res.setHeader('content-type',name.endsWith('.js')||raw?'text/javascript':'application/octet-stream');res.setHeader('content-length',servedBytes.length);res.setHeader('cache-control','no-store');res.end(servedBytes);
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  report.url='http://127.0.0.1:'+server.address().port+'/';
  report.phase='process-guard';
  const script=path.join(root,'tools/process_memory.py');
  report.observer={script,sha256:digest(fs.readFileSync(script)),python:'/usr/bin/python3',periodMs:1000};
  const stop=memoryStopAction({child:()=>child,report,persist});
  monitor=await startProcessMemory({python:report.observer.python,script,runId:report.runId,rawPath:reportPath+'.process.jsonl',maxFootprintBytes:report.requested.processBudgetBytes,onUnsafe:stop});
  report.phase='browser-launch';await persist();
  profile=await fs.promises.mkdtemp(path.join(os.tmpdir(),'sf3d-loader-browser-'));
  const allowed=['HOME','TMPDIR','PATH','LANG','LC_ALL','LC_CTYPE','__CF_USER_TEXT_ENCODING'];
  const env=Object.fromEntries(allowed.filter(name=>process.env[name]!=null).map(name=>[name,process.env[name]]));
  report.childEnvironment={policy:'positive-allowlist',names:Object.keys(env),valuesRecorded:false};
  const args=await ownedBrowserArguments(puppeteer,{profile,headless:!foregroundRoot});report.browserArguments=args;
  if(report.memorySafety)throw Error('memory guard prevented browser launch');
  // Own the child synchronously, before awaiting its endpoint. A threshold
  // crossing during launch can stop this exact child, not a pending promise.
  child=spawn(report.browserExecutable,args,{env,stdio:['ignore','ignore','pipe']});report.ownedBrowserPid=child.pid;
  const endpoint=await new Promise((resolve,reject)=>{
    let stderr='';
    const onData=chunk=>{stderr+=chunk;fs.appendFileSync(report.evidencePaths.browserLog,chunk);const match=stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(match){cleanup();resolve(match[1]);}};
    const onExit=(code,signal)=>{cleanup();reject(Error('owned browser exited before endpoint: '+code+'/'+signal));};
    const onError=error=>{cleanup();reject(error);};
    const cleanup=()=>{child.stderr.off('data',onData);child.off('exit',onExit);child.off('error',onError);};
    child.stderr.on('data',onData);child.once('exit',onExit);child.once('error',onError);
  });
  child.stderr.on('data',chunk=>fs.appendFileSync(report.evidencePaths.browserLog,chunk));
  if(report.memorySafety)throw Error('memory guard intervened during browser launch');
  browser=await puppeteer.connect({browserWSEndpoint:endpoint});report.browserVersion=await browser.version();
  const page=await browser.newPage();await page.goto(report.url);
  if(foregroundRoot){
    await page.setViewport({width:1280,height:960});
    await page.evaluateOnNewDocument(installForegroundAcquisition,{allowance:{cpuBytes:report.requested.cpuBytes,gpuBytes:report.requested.gpuBytes,totalBytes:report.requested.totalBytes},rendererCpuBytes:report.requested.foreground.rendererCpuBytes});
    report.phase='ordinary-foreground-initialization';await persist();
    foregroundPage=page;
    report.foreground.console=[];report.foreground.pageErrors=[];
    page.on('console',message=>report.foreground.console.push({type:message.type(),text:message.text()}));
    page.on('pageerror',error=>report.foreground.pageErrors.push(error.message));
    let failStartup;
    const startupFailure=new Promise((_resolve,reject)=>{failStartup=error=>reject(Error('foreground startup exception: '+error.message));});
    page.on('pageerror',failStartup);
    try{
      await Promise.race([page.goto(new URL(report.foreground.route,report.url).href,{waitUntil:'networkidle0',timeout:0}),startupFailure]);
      await Promise.race([page.waitForFunction(()=>window.__kaminosVolumePrototype?.debugState().active||window.__kaminosVolumePrototype?.debugState().error,{timeout:0}),startupFailure]);
    }finally{page.off('pageerror',failStartup);}
    const initial=await page.evaluate(()=>window.__kaminosVolumePrototype.debugState());
    if(!initial.active||initial.error)throw Error('ordinary foreground initialization failed: '+initial.error);
    await page.waitForFunction(()=>window.__kaminosVolumePrototype.debugState().frameCount>=3||window.__kaminosVolumePrototype.debugState().error,{timeout:0});
    const settled=await page.evaluate(()=>window.__kaminosVolumePrototype.debugState());if(settled.error)throw Error('ordinary foreground frame failed: '+settled.error);
    await page.screenshot({path:report.evidencePaths.foregroundBefore});
  }
  report.phase='native-loader-refusal';await persist();
  if(canonicalMode){
    report.phase='native-canonical-tensor-unit';await persist();
    if(patchMode){
      report.phase='native-canonical-patch-phase';
      report.processObservation=await monitor.sample();
      const baseline=report.processObservation;
      if(baseline.status!=='running'||baseline.runId!==report.runId||baseline.rootPid!==process.pid||
        baseline.lastObservation?.runId!==report.runId||baseline.lastObservation?.rootPid!==process.pid||
        baseline.lastObservation?.status!=='observed'||baseline.coverage!=='sampled-owned-process-tree'||report.memorySafety)
        throw Error('current successful owned-process baseline required before learned allocation');
      report.phaseProcessBaseline=baseline;
      await persist();
      if(foregroundRoot)report.foreground.before=await page.evaluate(async()=>{const p=window.__kaminosVolumePrototype,g=await window.__miniForegroundGuardPromise,s=p.debugState(),context=p.foregroundGpuContext();
        return{active:s.active,error:s.error,renderer:context.renderer,grid:s.simGrid,frameCount:s.frameCount,simStepCount:s.simStepCount,submissions:g.queueSubmissions.length,budget:g.snapshot()};});
      Object.assign(report,await page.evaluate(async config=>{const {runPatchPhase}=await import('/tools/patch_phase_browser.js');
        const hostGpu=config.requested.foreground?(await window.__miniForegroundGuardPromise).forDevice(window.__kaminosVolumePrototype.foregroundGpuContext().device):null;
        const result=await runPatchPhase(config,hostGpu);
        if(hostGpu)result.foregroundSameDevice=window.__kaminosVolumePrototype.foregroundGpuContext().device===hostGpu.device;
        if(hostGpu)result.foregroundDeviceIndex=hostGpu.deviceIndex;
        return result;},
        {requested:report.requested,source:report.canonicalSource,demand:report.phaseDemand,input:report.inputArtifact}));
      Object.assign(report.patchPhase,checkPatchOutput({source:report.canonicalSource,inputPath:report.evidencePaths.phaseInput,outputPath:report.evidencePaths.phaseOutput}));
      if(foregroundRoot){
        report.foreground.sameDevice=report.foregroundSameDevice;
        report.foreground.after=await page.evaluate(async()=>{const p=window.__kaminosVolumePrototype,g=await window.__miniForegroundGuardPromise,s=p.debugState(),context=p.foregroundGpuContext();
          return{active:s.active,error:s.error,renderer:context.renderer,grid:s.simGrid,frameCount:s.frameCount,simStepCount:s.simStepCount,submissions:g.queueSubmissions.length,budget:g.snapshot()};});
        await page.screenshot({path:report.evidencePaths.foregroundAfter});
        report.patchPhase.hostBudgetAfterPhase=report.patchPhase.budget;
        report.foreground.terminalBudget=await page.evaluate(async()=>{const guard=await window.__miniForegroundGuardPromise;window.__kaminosVolumePrototype.dispose();return guard.retire();});
        // Preserve raw allocation identity separately from the terminal snapshot.
        report.foreground.textureEvents=await page.evaluate(async()=>{const g=await window.__miniForegroundGuardPromise;return g.children.flatMap(row=>row.memoryBudget.events.filter(e=>e.kind==='host-texture-allocated'));});
        report.patchPhase.budget=report.foreground.terminalBudget.children[report.foregroundDeviceIndex]?.budget;
        foregroundPage=null;
      }
    }else{
    Object.assign(report,await page.evaluate(async config=>{
      const {loadWeightTensorUnit}=await import('/src/lib/weights.js'),{createLoaderMemoryBudget}=await import('/src/lib/loader_memory_budget.js');
      const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('actual WebGPU adapter required');
      const info=adapter.info,backend={vendor:info.vendor,architecture:info.architecture,description:info.description,isFallbackAdapter:info.isFallbackAdapter??adapter.isFallbackAdapter};
      if(backend.isFallbackAdapter!==false||!/apple/i.test(backend.vendor))throw Error('nonfallback Apple route required');
      const device=await adapter.requestDevice(),budget=createLoaderMemoryBudget(config.requested);budget.bindOwnedDevice(device);device.pushErrorScope('validation');
      let unit;const readbacks=[];
      try{
        unit=await loadWeightTensorUnit(device,'/canonical-weights.bin',config.requested.tensorNames,{memoryBudget:budget,expectedWeightBytes:config.source.byteLength,expectedSourceETag:config.source.etag});
        for(const [name,buffer]of unit.tensors){
          const output=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
          try{
            const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,output,0,buffer.size);device.queue.submit([encoder.finish()]);
            await output.mapAsync(GPUMapMode.READ);readbacks.push({name,bytes:buffer.size,f32Words:Array.from(new Uint32Array(output.getMappedRange()))});output.unmap();
          }finally{output.destroy();}
        }
        const loadingReport=unit.loadingReport;unit.dispose();unit=null;
        const validation=await device.popErrorScope();
        return {backend,tensorUnit:{loadingReport,readbacks,budget:budget.snapshot(),events:budget.events},validationError:validation?.message??null};
      }finally{unit?.dispose();device.destroy();budget.restore();}
    },{requested:report.requested,source:report.canonicalSource}));
    }
  }else{
  Object.assign(report,await page.evaluate(async fixtures=>{
    const {loadWeights}=await import('/src/lib/weights.js'),{createLoaderMemoryBudget}=await import('/src/lib/loader_memory_budget.js');
    const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('actual WebGPU adapter required');
    const info=adapter.info, backend={vendor:info.vendor,architecture:info.architecture,description:info.description,isFallbackAdapter:info.isFallbackAdapter??adapter.isFallbackAdapter};
    if(backend.isFallbackAdapter!==false || !/apple/i.test(backend.vendor))throw Error('nonfallback Apple route required');
    const device=await adapter.requestDevice();device.pushErrorScope('validation');
    const cases={},configs={sourceRefusal:{name:'f32',cpuBytes:fixtures.f32.bytes-1,gpuBytes:8},
      // Allow native response chunking to require a raw/header copy; refuse
      // the much larger FP32 conversion, not an incidental stream boundary.
      conversionRefusal:{name:'f16',cpuBytes:fixtures.f16.bytes+fixtures.f16.firstTensorBytes+2*fixtures.f16.headerBytes,gpuBytes:8},
      gpuRefusal:{name:'f32',cpuBytes:3*fixtures.f32.bytes,gpuBytes:8}};
    try {
      for(const [key,config]of Object.entries(configs)){
        const budget=createLoaderMemoryBudget(config);budget.bindOwnedDevice(device);let fetchCount=0,error,weights;
        const original=globalThis.fetch;globalThis.fetch=(...args)=>{fetchCount++;return original(...args);};
        try{weights=await loadWeights(device,'/fixture/'+config.name,null,{memoryBudget:budget,expectedWeightBytes:fixtures[config.name].bytes});}
        catch(e){error={name:e.name,message:e.message,memoryBudget:e.memoryBudget??null};}
        finally{globalThis.fetch=original;weights?.dispose();}
        cases[key]={requested:config,fetchCount,error,budget:budget.snapshot(),events:budget.events};budget.restore();
      }
      // Deliberate post-guard usability control, not charged to a restored guard.
      const input=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      const output=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      device.queue.writeBuffer(input,0,new Uint32Array([7,11,13,17]));const e=device.createCommandEncoder();e.copyBufferToBuffer(input,0,output,0,16);device.queue.submit([e.finish()]);
      await output.mapAsync(GPUMapMode.READ);const deviceControl=Array.from(new Uint32Array(output.getMappedRange()));output.unmap();input.destroy();output.destroy();
      const validation=await device.popErrorScope();return{backend,cases,deviceControl,validationError:validation?.message??null};
    }finally{device.destroy();}
  },report.fixtures));
  }
  if(sharedMode){
    report.phase='native-shared-allowance';await persist();
    report.sharedAllowance=await page.evaluate(async()=>{const {runSharedAllowanceWitness}=await import('/tools/shared_allowance_browser.js');return runSharedAllowanceWitness();});
  }
  report.phase='native-process-stop';report.processObservation=await monitor.stop();monitor=null;await persist();
  // Force refusal below an already observed tiny-run charge; never provoke growth/OOM.
  const observed=report.processObservation.lastObservation.sampledAggregatePhysicalFootprintBytes;
  const stopThreshold=Math.max(1,Math.floor(observed/4));
  try {
    const unexpected=await startProcessMemory({python:report.observer.python,script,runId:report.runId+'-refusal',rawPath:reportPath+'.refusal-process.jsonl',maxFootprintBytes:stopThreshold,onUnsafe:stop});
    await unexpected.stop();throw Error('expected native process refusal did not occur');
  } catch(error) {if(error.memorySummary?.safety?.reason!=='process-footprint-budget')throw error;report.processRefusal=error.memorySummary;}
  report.status='passed';
}catch(error){report.status='failed';report.error={name:error.name,message:error.message,stack:error.stack};}
finally{
  report.cleanup={};
  if(foregroundPage){try{report.foreground.failedCleanup=await foregroundPage.evaluate(async()=>{window.__kaminosVolumePrototype?.dispose();const g=await window.__miniForegroundGuardPromise;return g?.retire();});}catch(error){report.cleanup.foregroundError=error.message;}}
  try{if(monitor)report.processObservation=await monitor.stop();}catch(error){report.cleanup.observerError=error.message;report.status='failed';}
  try{report.cleanup.browser=child?await stopOwnedBrowser(child):{status:'not-started',exitObserved:true};}catch(error){report.cleanup.browser={error:error.message};report.status='failed';}
  if(server)await new Promise(resolve=>server.close(resolve));report.cleanup.server='closed';
  closeCanonicalSource(report,canonical);
  try{if(profile && report.cleanup.browser?.exitObserved)await fs.promises.rm(profile,{recursive:true});}
  catch(error){report.cleanup.profileError=error.message;report.status='failed';}
  report.verdict=canonicalMode?acceptStagedTensorWitness(report):acceptLoaderMemoryWitness(report);if(!report.verdict.ok)report.status='failed';
  report.terminalAt=new Date().toISOString();await persist();console.log(JSON.stringify({status:report.status,phase:report.phase,report:reportPath,errors:report.verdict.errors}));
  if(report.status!=='passed')process.exitCode=1;
}
