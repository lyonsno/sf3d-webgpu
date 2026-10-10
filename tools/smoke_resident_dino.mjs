#!/usr/bin/env node
// Source-pinned complete native encoder experiment. Full SF3D remains held.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import http from 'node:http';
import {execFileSync,spawn} from 'node:child_process';
import {createHash,randomUUID} from 'node:crypto';
import puppeteer from 'puppeteer-core';
import {prepareCanonicalTensorSource,closeCanonicalSource} from './canonical_tensor_source.mjs';
import {PATCH_TENSOR_NAMES} from './patch_phase_reference.mjs';
import {parseFlatWeightHeader} from '../src/lib/flat_tensor_ranges.js';
import {observeMacMemory} from './memory_admission.mjs';
import {startProcessMemory} from './process_memory_guard.mjs';
import {memoryStopAction,stopOwnedBrowser,ownedBrowserArguments} from './owned_browser_stop.mjs';
import {writeJsonReportAtomic} from './json_report_atomic.mjs';
import {dinoPhaseDemand,inspectDinoOutput,acceptResidentDino} from './resident_dino_acceptance.mjs';
const arg=name=>{const i=process.argv.indexOf(name);return i<0?null:process.argv[i+1];};
const root=arg('--repo-root')?path.resolve(arg('--repo-root')):null;
const requestedReportPath=path.resolve(arg('--report')??path.join(os.tmpdir(),'sf3d-dino-'+randomUUID()+'.json'));
const occupied=fs.existsSync(requestedReportPath)||fs.existsSync(requestedReportPath+'.input.f32')||fs.existsSync(requestedReportPath+'.output.f32');
const reportPath=occupied?requestedReportPath+'.refused-'+randomUUID()+'.json':requestedReportPath;
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const report={schema:'sf3d.native-resident-dino.v0',status:'running',phase:'arguments',runId:randomUUID(),rootPid:process.pid,
  receiver:'mini-wake-and-bake-pit-boss',claim:'complete native DINO only; no full SF3D/GLB, reference parity, physical fit or rendering-composition claim',
  requested:{report:requestedReportPath,repoRoot:root,revision:arg('--expected-revision'),weightsPath:arg('--weights'),weightsSha256:arg('--expected-weights-sha256'),
    input:arg('--input'),inputSha256:arg('--expected-input-sha256'),chrome:arg('--chrome'),processBudgetBytes:Number(arg('--process-budget-bytes')),
    cpuBytes:128*1024*1024,gpuBytes:384*1024*1024,totalBytes:512*1024*1024},
  evidencePaths:{report:reportPath,input:reportPath+'.input.f32',output:reportPath+'.output.f32',process:reportPath+'.process.jsonl',browserLog:reportPath+'.chrome.log'},
  phaseObservations:[]};
fs.mkdirSync(path.dirname(reportPath),{recursive:true});
const persist=()=>writeJsonReportAtomic(reportPath,report);
let source,monitor,child,browser,server,profile;
try{
  await persist();
  if(occupied)throw Error('requested evidence paths already exist; retained previous evidence without reuse or overwrite');
  if(!root||!report.requested.revision||!report.requested.chrome||!report.requested.input||
    !Number.isSafeInteger(report.requested.processBudgetBytes)||report.requested.processBudgetBytes<1)throw Error('explicit source, browser, input and process diagnostic allowance required');
  const git=args=>execFileSync('git',args,{cwd:root,encoding:'utf8'}).trim();
  report.source={repoRoot:root,revision:git(['rev-parse','HEAD']),clean:git(['status','--porcelain'])==='',hostname:os.hostname()};
  if(!report.source.clean||report.source.revision!==report.requested.revision)throw Error('clean exact requested source required');
  report.browserExecutable=fs.realpathSync(report.requested.chrome);
  if(/\/Google Chrome\.app\//.test(report.browserExecutable))throw Error('independent browser required');
  report.phase='immutable-source';await persist();
  source=await prepareCanonicalTensorSource({weightsPath:report.requested.weightsPath,expectedSha256:report.requested.weightsSha256,
    tensorNames:[...PATCH_TENSOR_NAMES],cpuBytes:report.requested.cpuBytes,gpuBytes:report.requested.gpuBytes,referenceMode:'source-file'});
  report.canonicalSource=source.receipt;
  const fd=fs.openSync(report.requested.weightsPath,'r'),header=Buffer.alloc(source.receipt.headerBytes);
  try{if(fs.readSync(fd,header,0,header.length,0)!==header.length)throw Error('complete header required');}finally{fs.closeSync(fd);}
  if(digest(header)!==source.receipt.headerSha256)throw Error('header source identity drift');
  const table=parseFlatWeightHeader(header.buffer.slice(header.byteOffset,header.byteOffset+header.length),source.receipt.byteLength).tensors;
  const image=fs.readFileSync(report.requested.input);
  if(digest(image)!==report.requested.inputSha256||image.toString('ascii',12,16)!=='IHDR')throw Error('canonical PNG input identity required');
  report.inputArtifact={sha256:digest(image),byteLength:image.length,width:image.readUInt32BE(16),height:image.readUInt32BE(20)};
  if(report.inputArtifact.width!==503||report.inputArtifact.height!==503)throw Error('source-fixed preprocessing dimensions changed');
  report.phase='library-build';await persist();
  const log=fs.openSync(reportPath+'.build.log','w');
  try{execFileSync(process.execPath,[path.join(root,'node_modules/vite/bin/vite.js'),'build','-c','vite.lib.config.js'],{cwd:root,stdio:['ignore',log,log]});}finally{fs.closeSync(log);}
  const artifact=fs.readFileSync(path.join(root,'dist-lib/sf3d-producer.js'));
  report.artifact={sha256:digest(artifact),kitVersion:JSON.parse(fs.readFileSync(path.join(root,'node_modules/@kaminos/webgpu-inference-kit/package.json'))).version,
    entry:'/dist-lib/sf3d-producer.js',sourceRevision:report.source.revision,lockSha256:digest(fs.readFileSync(path.join(root,'package-lock.json')))};
  const order=['preprocess','camera',...Array.from({length:24},(_,i)=>'dino-block-'+i),'dino-output'];
  const readBody=async req=>{const chunks=[];for await(const chunk of req)chunks.push(chunk);return Buffer.concat(chunks);};
  server=http.createServer(async(req,res)=>{
    try{
      const name=new URL(req.url,'http://localhost').pathname;
      if(name==='/canonical-weights.bin'){source.serve(req,res);return;}
      if(name==='/'){res.writeHead(200,{'Content-Type':'text/html','Cache-Control':'no-store'}).end('<title>Complete SF3D DINO encoder experiment</title><h1>Complete DINO encoder — guarded experiment</h1><p id="status">No model work started</p><img id="input">');return;}
      if(name==='/favicon.ico'){res.writeHead(204).end();return;}
      if(name===report.artifact.entry){report.artifact.servedSha256=digest(artifact);res.writeHead(200,{'Content-Type':'text/javascript','Cache-Control':'no-store'}).end(artifact);return;}
      if(name==='/image.png'){res.writeHead(200,{'Content-Type':'image/png','Cache-Control':'no-store'}).end(image);return;}
      if(name==='/phase'&&req.method==='POST'){
        const phase=JSON.parse((await readBody(req)).toString());
        if(phase.name!==order[report.phaseObservations.length]||report.memorySafety||report.phaseObservations.some(o=>o.verdict!=='admitted'))throw Error('phase order or safety hold prevents allocation');
        for(const tensor of phase.tensors??[]){const observed=table.get(tensor.name);
          if(!observed||observed.size!==tensor.size||observed.offset!==tensor.offset||observed.dtype!==tensor.dtype)throw Error('effective tensor source metadata mismatch');}
        const demand=dinoPhaseDemand(phase),processObservation=await monitor.sample();
        const host={...observeMacMemory(),model:execFileSync('sysctl',['-n','hw.model'],{encoding:'utf8'}).trim(),processor:execFileSync('sysctl',['-n','machdep.cpu.brand_string'],{encoding:'utf8'}).trim()};
        const processRow=processObservation.lastObservation;
        const observation={phase:phase.name,descriptor:phase,demand,host,process:processObservation,
          authority:'reversible encoder phase with observed baseline and diagnostic process stop; not production physical fit',
          verdict:host.hostname===report.source.hostname&&host.model==='Mac14,9'&&host.processor==='Apple M2 Pro'&&!host.observerErrors.length&&
            host.hostFreeBytes>=demand.requiredBytes&&processObservation.coverage==='sampled-owned-process-tree'&&processRow?.status==='observed'&&
            processRow.runId===report.runId&&processRow.rootPid===process.pid&&processRow.sampledAggregatePhysicalFootprintBytes+demand.requiredBytes<=report.requested.processBudgetBytes&&!report.memorySafety?'admitted':'refused'};
        report.phaseObservations.push(observation);report.phase='phase-'+phase.name;await persist();
        res.writeHead(observation.verdict==='admitted'?200:409,{'Content-Type':'application/json','Cache-Control':'no-store'}).end(JSON.stringify(observation));return;
      }
      if(req.method==='POST'&&['/input.f32','/output.f32'].includes(name)){
        const bytes=await readBody(req),expected=name==='/input.f32'?3*512*512*4:1297*1024*4;
        if(bytes.length!==expected)throw Error('partial or conflicting full tensor shape');
        fs.writeFileSync(name==='/input.f32'?report.evidencePaths.input:report.evidencePaths.output,bytes);
        res.writeHead(200).end();return;
      }
      res.writeHead(404).end();
    }catch(error){const message=String(error?.message??error);report.lastServerError={message,phase:report.phase};await persist();res.writeHead(409).end(message);}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  report.url='http://127.0.0.1:'+server.address().port+'/';
  report.phase='process-guard';await persist();
  monitor=await startProcessMemory({python:'/usr/bin/python3',script:path.join(root,'tools/process_memory.py'),runId:report.runId,
    rawPath:report.evidencePaths.process,maxFootprintBytes:report.requested.processBudgetBytes,onUnsafe:memoryStopAction({child:()=>child,report,persist})});
  profile=await fs.promises.mkdtemp(path.join(os.tmpdir(),'sf3d-resident-dino-browser-'));
  const allowed=['HOME','TMPDIR','PATH','LANG','LC_ALL','LC_CTYPE','__CF_USER_TEXT_ENCODING'];
  const env=Object.fromEntries(allowed.filter(n=>process.env[n]!=null).map(n=>[n,process.env[n]]));
  const args=await ownedBrowserArguments(puppeteer,{profile,headless:false});report.browserArguments=args;
  report.phase='browser-launch';await persist();if(report.memorySafety)throw Error('memory guard prevents launch');
  child=spawn(report.browserExecutable,args,{env,stdio:['ignore','ignore','pipe']});report.ownedBrowserPid=child.pid;
  const endpoint=await new Promise((resolve,reject)=>{
    let stderr='';
    const cleanup=()=>{child.stderr.off('data',data);child.off('exit',exited);child.off('error',error);};
    const data=chunk=>{stderr+=chunk;fs.appendFileSync(report.evidencePaths.browserLog,chunk);const match=stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(match){cleanup();resolve(match[1]);}};
    const exited=(code,signal)=>{cleanup();reject(Error('owned browser exited '+code+'/'+signal));};
    const error=err=>{cleanup();reject(err);};child.stderr.on('data',data);child.once('exit',exited);child.once('error',error);
  });
  child.stderr.on('data',chunk=>fs.appendFileSync(report.evidencePaths.browserLog,chunk));
  browser=await puppeteer.connect({browserWSEndpoint:endpoint,protocolTimeout:0});report.browserVersion=await browser.version();
  const page=await browser.newPage();await page.setViewport({width:1280,height:960});await page.goto(report.url,{timeout:0});
  report.phase='native-dino';await persist();
  Object.assign(report,await page.evaluate(async config=>{
    const {createLoaderMemoryBudget,runResidentDino,preprocessImage}=await import('/dist-lib/sf3d-producer.js');
    const observe=async phase=>{const response=await fetch('/phase',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(phase)});
      if(!response.ok)throw Error('fresh phase observation refused: '+await response.text());};
    const persistTensor=async(url,bytes)=>{const response=await fetch(url,{method:'POST',body:bytes});if(!response.ok)throw Error('complete tensor persistence failed: '+await response.text());};
    const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('WebGPU adapter unavailable');
    const info=adapter.info,backend={vendor:info.vendor,architecture:info.architecture,description:info.description,isFallbackAdapter:info.isFallbackAdapter??adapter.isFallbackAdapter};
    if(backend.isFallbackAdapter!==false||!/apple/i.test(backend.vendor))throw Error('actual nonfallback Apple adapter required');
    const budget=createLoaderMemoryBudget(config.requested),device=await budget.requestOwnedDevice(adapter,{requiredLimits:{maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
    let imageLease,bitmap,staging,failure,failed=false;
    device.pushErrorScope('validation');
    try{
      await observe({name:'preprocess',tensors:[],width:config.input.width,height:config.input.height});
      imageLease=budget.reserveCpu(80*1024*1024,'native-condition-image-and-readback');
      const imageResponse=await fetch('/image.png');bitmap=await createImageBitmap(await imageResponse.blob());
      if(bitmap.width!==config.input.width||bitmap.height!==config.input.height)throw Error('decoded canonical dimensions changed');
      const chw=await preprocessImage(bitmap);await persistTensor('/input.f32',new Uint8Array(chw.buffer));
      const dino=await runResidentDino({device,memoryBudget:budget,imageChw:chw,weightsUrl:'/canonical-weights.bin',
        expectedWeightBytes:config.source.byteLength,expectedSourceETag:config.source.etag,onBeforePhase:observe,
        onProgress:p=>{document.querySelector('#status').textContent='DINO '+(p.completedItems??0)+'/24 blocks';},
        async withResult(result){
          staging=device.createBuffer({size:result.tokensBuf.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,label:'complete-dino-readback'});
          const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(result.tokensBuf,0,staging,0,result.tokensBuf.size);device.queue.submit([encoder.finish()]);
          await staging.mapAsync(GPUMapMode.READ);await persistTensor('/output.f32',new Uint8Array(staging.getMappedRange()));staging.unmap();staging.destroy();staging=null;
        }});
      imageLease.release();imageLease=null;const validation=await device.popErrorScope();
      if(validation)throw Error('native validation: '+validation.message);
      document.querySelector('#status').textContent='All24 DINO blocks completed; full SF3D not yet run';
      return {backend,dino,validationError:null,budget:budget.snapshot()};
    }catch(error){failed=true;failure=error;throw error;}
    finally{
      bitmap?.close();
      try{await device.queue.onSubmittedWorkDone();staging?.destroy();imageLease?.release();device.destroy();budget.restore();}
      catch(error){throw failed?new AggregateError([failure,error],'native DINO failed and cleanup failed',{cause:failure}):error;}
    }
  },{requested:report.requested,source:report.canonicalSource,input:report.inputArtifact}));
  report.output=inspectDinoOutput(report.evidencePaths.output);report.phase='complete';report.status='passed';
}catch(error){report.status='failed';report.error={message:String(error?.message??error),stack:error?.stack,lastTrustworthyPhase:report.phase};}
finally{
  report.cleanup={};
  try{if(monitor)report.processObservation=await monitor.stop();}catch(error){report.cleanup.observerError=error.message;report.status='failed';}
  try{report.cleanup.browser=child?await stopOwnedBrowser(child):{status:'not-started',exitObserved:true};}catch(error){report.cleanup.browser={error:error.message};report.status='failed';}
  try{if(server)await new Promise(resolve=>server.close(resolve));report.cleanup.server='closed';}catch(error){report.cleanup.serverError=error.message;report.status='failed';}
  closeCanonicalSource(report,source);
  try{if(profile&&report.cleanup.browser?.exitObserved)await fs.promises.rm(profile,{recursive:true});}catch(error){report.cleanup.profileError=error.message;report.status='failed';}
  report.verdict=acceptResidentDino(report);if(!report.verdict.ok)report.status='failed';report.terminalAt=new Date().toISOString();await persist();
  console.log(JSON.stringify({status:report.status,phase:report.phase,report:reportPath,errors:report.verdict.errors}));
  if(report.status!=='passed')process.exitCode=1;
}
