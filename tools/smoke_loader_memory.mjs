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
import {acceptLoaderMemoryWitness} from './loader_memory_witness_acceptance.mjs';
const arg = name => {const i=process.argv.indexOf(name);return i<0?null:process.argv[i+1];};
const root = arg('--repo-root') ? path.resolve(arg('--repo-root')) : null;
const reportPath = path.resolve(arg('--report') ?? path.join(os.tmpdir(),'sf3d-loader-memory-'+randomUUID()+'.json'));
const report = {schema:'sf3d.loader-native-memory-witness.v0', status:'running', phase:'arguments', receiver:'mini-wake-and-bake-pit-boss',
  route:'sf3d-loader-native-refusal-synthetic-weights-no-inference.v0', reportPath, runId:randomUUID(),
  requested:{repoRoot:root, revision:arg('--expected-revision'), chrome:arg('--chrome'), processBudgetBytes:Number(arg('--process-budget-bytes'))},
  claim:'real loader refusal/cleanup and owned-process stop only; synthetic weights, no learned compute or M2 fit claim'};
report.evidencePaths={report:reportPath, browserLog:reportPath+'.chrome.log', process:reportPath+'.process.jsonl', processRefusal:reportPath+'.refusal-process.jsonl'};
fs.mkdirSync(path.dirname(reportPath),{recursive:true});
const persist = async () => writeJsonReportAtomic(reportPath,report);
let browser, child, server, monitor, profile;
try {
  await persist();
  if(!root || !arg('--chrome') || !arg('--expected-revision') || !Number.isSafeInteger(report.requested.processBudgetBytes) || report.requested.processBudgetBytes<1)
    throw Error('explicit --repo-root, --expected-revision, --chrome and --process-budget-bytes required');
  report.phase='source-identity';
  const git = args => execFileSync('git',args,{cwd:root,encoding:'utf8'}).trim();
  report.source={revision:git(['rev-parse','HEAD']), clean:git(['status','--porcelain'])==='', repoRoot:root, hostname:os.hostname()};
  if(!report.source.clean || report.source.revision!==report.requested.revision)throw Error('clean exact requested source required');
  report.browserExecutable=fs.realpathSync(arg('--chrome'));
  if(/\/Google Chrome\.app\//.test(report.browserExecutable))throw Error('independent browser required; installed GUI Chrome cannot run headlessly');
  report.phase='fixture-preparation';
  const fixtures={f32:weightFixture().bytes, f16:weightFixture({tensorShapes:new Map([['image_tokenizer.image_mean',[300000]]]),fp16Names:new Set(['image_tokenizer.image_mean'])}).bytes};
  const digest = bytes => createHash('sha256').update(bytes).digest('hex');
  report.fixtures=Object.fromEntries(Object.entries(fixtures).map(([name,bytes])=>{
    const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
    return [name,{bytes:bytes.length,headerBytes:view.getUint32(12,true),firstTensorBytes:view.getUint32(16+156,true),sha256:digest(bytes),meaning:'synthetic flat-v1 loader fixture; not canonical model weights'}];
  }));
  const sources=new Map();
  for(const relative of ['src/lib/weights.js','src/lib/gpu.js','src/lib/loader_memory_budget.js']) {
    const bytes=fs.readFileSync(path.join(root,relative)), committed=execFileSync('git',['show',report.source.revision+':'+relative],{cwd:root});
    if(digest(bytes)!==digest(committed))throw Error('served source differs from commit: '+relative);
    sources.set('/'+relative,bytes);
  }
  report.servedSources=Object.fromEntries([...sources].map(([name,bytes])=>[name,digest(bytes)]));
  server=http.createServer((req,res)=>{
    const name=new URL(req.url,'http://localhost').pathname;
    if(name==='/'){res.setHeader('content-type','text/html');res.end('<title>SF3D native loader refusal — no inference</title>');return;}
    if(name==='/favicon.ico'){res.writeHead(204).end();return;}
    const bytes=name.startsWith('/fixture/')?fixtures[name.slice(9)]:sources.get(name);
    if(!bytes){res.writeHead(404).end();return;}
    res.setHeader('content-type',name.endsWith('.js')?'text/javascript':'application/octet-stream');res.setHeader('content-length',bytes.length);res.setHeader('cache-control','no-store');res.end(bytes);
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
  if(profile && report.cleanup.browser?.exitObserved)await fs.promises.rm(profile,{recursive:true});
  report.verdict=acceptLoaderMemoryWitness(report);if(!report.verdict.ok)report.status='failed';
  report.terminalAt=new Date().toISOString();await persist();console.log(JSON.stringify({status:report.status,phase:report.phase,report:reportPath,errors:report.verdict.errors}));
  if(report.status!=='passed')process.exitCode=1;
}
