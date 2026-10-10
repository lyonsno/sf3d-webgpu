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
const arg = name => {const i=process.argv.indexOf(name);return i<0?null:process.argv[i+1];};
const root = arg('--repo-root') ? path.resolve(arg('--repo-root')) : null;
const reportPath = path.resolve(arg('--report') ?? path.join(os.tmpdir(),'sf3d-loader-memory-'+randomUUID()+'.json'));
const patchMode=process.argv.includes('--canonical-patch-phase');
const canonicalMode=patchMode||process.argv.includes('--canonical-tensor-unit');
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
fs.mkdirSync(path.dirname(reportPath),{recursive:true});
const persist = async () => writeJsonReportAtomic(reportPath,report);
let browser, child, server, monitor, profile, canonical;
try {
  await persist();
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
      writeJsonReportAtomic(reportPath,report).then(()=>{res.writeHead(report.phaseAdmission.verdict==='admitted'?200:409,{'Content-Type':'application/json','Cache-Control':'no-store'}).end(JSON.stringify(report.phaseAdmission));},error=>res.writeHead(500).end(error.message));return;
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
  const args=await ownedBrowserArguments(puppeteer,{profile});report.browserArguments=args;
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
      await persist();
      Object.assign(report,await page.evaluate(async config=>{const {runPatchPhase}=await import('/tools/patch_phase_browser.js');return runPatchPhase(config);},
        {requested:report.requested,source:report.canonicalSource,demand:report.phaseDemand,input:report.inputArtifact}));
      Object.assign(report.patchPhase,checkPatchOutput({source:report.canonicalSource,inputPath:report.evidencePaths.phaseInput,outputPath:report.evidencePaths.phaseOutput}));
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
