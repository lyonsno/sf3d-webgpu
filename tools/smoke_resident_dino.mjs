#!/usr/bin/env node
// Source-pinned native experiment. Backbone invocation requires:
// node --import ./tools/wgsl-raw-loader-register.mjs tools/smoke_resident_dino.mjs --through-backbone ...
// Explicit full-model diagnostic does not reopen ordinary/eager admission.
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
import {observeMacMemory,evaluatePhaseHostHeadroom,startHostPressureGuard,AVAILABLE_MEMORY_DIAGNOSTIC_POLICY} from './memory_admission.mjs';
import {startProcessMemory} from './process_memory_guard.mjs';
import {memoryStopAction,stopOwnedBrowser,ownedBrowserArguments} from './owned_browser_stop.mjs';
import {writeJsonReportAtomic} from './json_report_atomic.mjs';
import {dinoPhaseDemand,inspectDinoInputBytes,inspectDinoOutput,acceptResidentDino} from './resident_dino_acceptance.mjs';
import {twoStreamPhaseDemand,inspectTwoStreamOutput,acceptResidentTwoStream,residentTwoStreamExpectedPhases} from './resident_two_stream_acceptance.mjs';
const throughBackbone=process.argv.includes('--through-backbone');
const throughPostProcessor=process.argv.includes('--through-postprocessor');
const throughFullModel=process.argv.includes('--through-full-model');
let groups=null,postWitness=null,tensorTable,artifactContract;
let createArtifactPhaseContract,sourceIntakePhaseDemand,inspectCompleteGlb,acceptNativeArtifact,persistArtifactDelivery,RESIDENT_ARTIFACT_CONFIG;
const arg=name=>{const i=process.argv.indexOf(name);return i<0?null:process.argv[i+1];};
const root=arg('--repo-root')?path.resolve(arg('--repo-root')):null;
const requestedReportPath=path.resolve(arg('--report')??path.join(os.tmpdir(),'sf3d-dino-'+randomUUID()+'.json'));
const occupied=['','.input.f32','.output.f32','.triplane.f32','.postprocessor.f32','.model.glb','.mesh.f32','.faces.u32'].some(s=>fs.existsSync(requestedReportPath+s));
const reportPath=occupied?requestedReportPath+'.refused-'+randomUUID()+'.json':requestedReportPath;
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const report={schema:throughFullModel?'sf3d.native-resident-artifact.v0':throughPostProcessor?'sf3d.native-resident-postprocessor.v0':throughBackbone?'sf3d.native-resident-backbone.v0':'sf3d.native-resident-dino.v0',status:'running',phase:'arguments',runId:randomUUID(),rootPid:process.pid,
  receiver:'mini-wake-and-bake-pit-boss',claim:throughPostProcessor?'complete native DINO/backbone/postprocessor; no mesh/material/GLB, reference parity, physical fit or rendering-composition claim':throughBackbone?'complete native DINO plus existing two-stream backbone; no full SF3D/GLB, reference parity, physical fit or rendering-composition claim':'complete native DINO only; no full SF3D/GLB, reference parity, physical fit or rendering-composition claim',
  requested:{report:requestedReportPath,repoRoot:root,revision:arg('--expected-revision'),weightsPath:arg('--weights'),weightsSha256:arg('--expected-weights-sha256'),
    input:arg('--input'),inputSha256:arg('--expected-input-sha256'),chrome:arg('--chrome'),processBudgetBytes:Number(arg('--process-budget-bytes')),
    throughBackbone,attentionRowsPerDuty:Number(arg('--attention-rows-per-duty')??128),
    throughPostProcessor,throughFullModel,postChannelsPerDuty:Number(arg('--post-channels-per-duty')??16),
    reuseDeadTriplaneStorage:process.argv.includes('--reuse-dead-triplane-storage'),
    reuseAttentionResidualStorage:process.argv.includes('--reuse-attention-residual-storage'),
    hostHeadroomPolicy:arg('--host-headroom-policy')??'raw-free-pages-v0',
    cpuBytes:(throughBackbone?256:128)*1024*1024,gpuBytes:(throughBackbone?1024:384)*1024*1024,totalBytes:(throughBackbone?1280:512)*1024*1024},
  allowanceBasis:throughBackbone?'canonical embedding source56,623,104bytes FP16: response/destination/conversion bound226,492,416CPUbytes; GPU persistent embedding/triplane/latent, one selected stage, Q/K/V, caller-declared complete-query score scratch and reusable128-row FFN scratch; diagnostic only; unchanged2GiB stop and fresh raw host-free gates':'source-fixed encoder backing',
  evidencePaths:{report:reportPath,input:reportPath+'.input.f32',output:reportPath+'.output.f32',twoStreamOutput:reportPath+'.triplane.f32',process:reportPath+'.process.jsonl',browserLog:reportPath+'.chrome.log'},
  phaseObservations:[]};
fs.mkdirSync(path.dirname(reportPath),{recursive:true});
const persist=()=>writeJsonReportAtomic(reportPath,report);
let source,monitor,hostMonitor,child,browser,server,profile,safetyStop;
const stopForSafety=safety=>{
  if(!safetyStop)safetyStop=memoryStopAction({child:()=>child,report,persist})(safety);
  return safetyStop;
};
try{
  await persist();
  if(occupied)throw Error('requested evidence paths already exist; retained previous evidence without reuse or overwrite');
  if(throughFullModel){
    report.claim='complete canonical native model and textured GLB; no reference-parity, opaque physical-capacity or living-foreground claim';
    Object.assign(report.evidencePaths,{glb:reportPath+'.model.glb',mesh:reportPath+'.mesh.f32',faces:reportPath+'.faces.u32'});
    if(!throughBackbone||!throughPostProcessor)throw Error('full model requires complete backbone and postprocessor');
    report.phase='full-consumer-graph-import';await persist();
    ({createArtifactPhaseContract,sourceIntakePhaseDemand,inspectCompleteGlb,acceptNativeArtifact,persistArtifactDelivery}=await import('./resident_artifact_acceptance.mjs'));
    ({RESIDENT_ARTIFACT_CONFIG}=await import('../src/lib/resident_artifact.js'));
  }
  if(throughPostProcessor){
    if(!throughBackbone)throw Error('postprocessor requires complete backbone invocation');
    postWitness=await import('./resident_post_processor_acceptance.mjs');
    postWitness.postProcessorExpectedPhases(report.requested.postChannelsPerDuty);
    report.evidencePaths.postProcessorOutput=reportPath+'.postprocessor.f32';
  }
  if(report.requested.reuseDeadTriplaneStorage&&!throughBackbone)throw Error('dead-triplane reuse requires complete backbone invocation');
  if(report.requested.reuseAttentionResidualStorage&&(!throughBackbone||!report.requested.reuseDeadTriplaneStorage))
    throw Error('attention residual reuse requires explicit owned dead-triplane backbone route');
  if(!['raw-free-pages-v0',AVAILABLE_MEMORY_DIAGNOSTIC_POLICY].includes(report.requested.hostHeadroomPolicy))throw Error('unknown explicit host headroom policy; no fallback');
  if(report.requested.hostHeadroomPolicy===AVAILABLE_MEMORY_DIAGNOSTIC_POLICY){
    if(!throughBackbone||report.requested.processBudgetBytes!==2147483648)throw Error('available-memory diagnostic requires complete backbone and unchanged2GiB process stop');
    report.allowanceBasis+='; explicitly selected OS availability estimate, independent historical24% pressure stop, no ordinary full-route admission';
  }
  if(!root||!report.requested.revision||!report.requested.chrome||!report.requested.input||
    !Number.isSafeInteger(report.requested.processBudgetBytes)||report.requested.processBudgetBytes<1)throw Error('explicit source, browser, input and process diagnostic allowance required');
  if(throughBackbone){
    report.phase='backbone-graph-import';await persist();
    const backboneModule=await import('../src/lib/two_stream.js');
    groups=(await import('../src/lib/cooperative_two_stream.js')).groupTwoStreamDuties(
      backboneModule.createTwoStreamAttentionDutyPlan(1297,{residentFFN:true,linearRowsPerDuty:128,attentionRowsPerDuty:report.requested.attentionRowsPerDuty}));
  }
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
  const table=tensorTable=parseFlatWeightHeader(header.buffer.slice(header.byteOffset,header.byteOffset+header.length),source.receipt.byteLength).tensors;
  if(throughFullModel)artifactContract=createArtifactPhaseContract({table,headerBytes:source.receipt.headerBytes});
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
  if(throughFullModel)report.tetAssets=['grid','tets'].map(kind=>{
    const url=kind==='grid'?'/tets/_grid_vertices.bin':'/tets/indices.bin',bytes=fs.readFileSync(path.join(root,'public',url)),c=RESIDENT_ARTIFACT_CONFIG;
    const expectedBytes=kind==='grid'?c.gridBytes:c.tetBytes,expectedHash=kind==='grid'?c.gridSha256:c.tetSha256;
    if(bytes.length!==expectedBytes||digest(bytes)!==expectedHash)throw Error('clean source canonical complete tet asset mismatch');
    return {url,bytes:bytes.length,sha256:digest(bytes),sourceRevision:report.source.revision};});
  const order=['preprocess','camera',...Array.from({length:24},(_,i)=>'dino-block-'+i),'dino-output'];
  if(throughBackbone){
    order.push('two-stream-embedding-weights','two-stream-embedding-rearrange');
    for(const group of groups){
      order.push('two-stream-'+group.stageId);
      for(const duty of group.duties)if(twoStreamPhaseDemand({name:'two-stream-duty',duty,attentionRowsPerDuty:report.requested.attentionRowsPerDuty,reuseDeadTriplaneStorage:report.requested.reuseDeadTriplaneStorage,reuseAttentionResidualStorage:report.requested.reuseAttentionResidualStorage}).requiredBytes>0)
        order.push('two-stream-duty-'+duty.dutyIndex);
    }
    order.push('two-stream-output');
    if(order.join(',')!==residentTwoStreamExpectedPhases(report.requested.attentionRowsPerDuty).join(','))throw Error('effective source graph differs from approved complete backbone cliffs');
  }
  report.expectedPhaseOrder=order;
  if(throughPostProcessor)order.push(...postWitness.postProcessorExpectedPhases(report.requested.postChannelsPerDuty));
  if(throughFullModel){
    order.unshift('device-acquisition');
    for(const [before,scope]of [['camera','dino'],['two-stream-embedding-weights','two-stream'],['post-processor-output-allocation','post-processor']])
      order.splice(order.indexOf(before),0,scope+'-source-weight-header-prefix',scope+'-source-weight-header');
    report.artifactPrefixPhaseCount=order.length;
  }
  const readBody=async req=>{const chunks=[];for await(const chunk of req)chunks.push(chunk);return Buffer.concat(chunks);};
  server=http.createServer(async(req,res)=>{
    try{
      const name=new URL(req.url,'http://localhost').pathname;
      if(name==='/canonical-weights.bin'){source.serve(req,res);return;}
      if(name==='/'){res.writeHead(200,{'Content-Type':'text/html','Cache-Control':'no-store'}).end('<title>Guarded SF3D native diagnostic</title><h1>'+ (throughFullModel?'Complete canonical model':'Complete DINO encoder') +' — guarded diagnostic</h1><p id="status">No model work started</p><img id="input">');return;}
      if(name==='/favicon.ico'){res.writeHead(204).end();return;}
      if(name==='/backend'&&req.method==='POST'){
        const backend=JSON.parse((await readBody(req)).toString());
        if(report.backend)throw Error('native backend identity already fixed');
        report.backend=backend;await persist();
        if(backend.isFallbackAdapter!==false||!/apple/i.test(backend.vendor??''))throw Error('actual nonfallback Apple route required before model allocation');
        res.writeHead(200).end();return;
      }
      if(name===report.artifact.entry){report.artifact.servedSha256=digest(artifact);res.writeHead(200,{'Content-Type':'text/javascript','Cache-Control':'no-store'}).end(artifact);return;}
      if(name==='/image.png'){res.writeHead(200,{'Content-Type':'image/png','Cache-Control':'no-store'}).end(image);return;}
      if(throughFullModel&&['/tets/_grid_vertices.bin','/tets/indices.bin'].includes(name)){
        const filename=path.join(root,'public',name),bytes=fs.readFileSync(filename);
        const entry=report.tetAssets?.find(a=>a.url===name);
        if(!entry||digest(bytes)!==entry.sha256||bytes.length!==entry.bytes)throw Error('pinned tet asset changed before serving');
        res.writeHead(200,{'Content-Type':'application/octet-stream','Content-Length':bytes.length,'Cache-Control':'no-store'}).end(bytes);return;
      }
      if(name==='/phase'&&req.method==='POST'){
        const phaseRequestedAtUnixMs=Date.now();
        const phase=JSON.parse((await readBody(req)).toString());
        const inPrefix=report.phaseObservations.length<order.length;
        if(!report.backend||report.backend.isFallbackAdapter!==false||(inPrefix&&phase.name!==order[report.phaseObservations.length])||(!inPrefix&&!throughFullModel)||report.memorySafety||report.phaseObservations.some(o=>o.verdict!=='admitted'))throw Error('effective backend, phase order or safety hold prevents allocation');
        for(const tensor of phase.tensors??[]){const observed=table.get(tensor.name);
          if(!observed||observed.size!==tensor.size||observed.offset!==tensor.offset||observed.dtype!==tensor.dtype)throw Error('effective tensor source metadata mismatch');}
        let demand;
        if(!inPrefix)demand=artifactContract.accept(phase);
        else if(throughFullModel&&phase.name==='device-acquisition'){
          if(phase.requiredBytes!==0||phase.rangeCpuBytes!==0||phase.workGpuBytes!==0||phase.tensors?.length)throw Error('device acquisition backing descriptor changed');
          demand={weightGpuBytes:0,rangeCpuBytes:0,workGpuBytes:0,requiredBytes:0,components:[]};
        }else if(throughFullModel&&/^(dino|two-stream|post-processor)-source-/.test(phase.name))demand=sourceIntakePhaseDemand(phase,source.receipt.headerBytes);
        else if(phase.name.startsWith('two-stream-')){
          if(!throughBackbone)throw Error('requested route does not include backbone');
          if(phase.name.startsWith('two-stream-duty-')){
            const index=Number(phase.name.slice('two-stream-duty-'.length)),expected=groups.flatMap(g=>g.duties)[index];
            if(!expected||phase.duty?.dutyIndex!==expected.dutyIndex||phase.duty?.dutyId!==expected.dutyId||
              phase.duty?.kind!==expected.kind||phase.duty?.rowStart!==expected.rowStart||
              phase.duty?.rowCount!==expected.rowCount||phase.duty?.direction!==expected.direction)
              throw Error('effective complete backbone duty identity mismatch');
            if(phase.attentionRowsPerDuty!==report.requested.attentionRowsPerDuty)throw Error('effective attention allocation granule mismatch');
            if(phase.reuseDeadTriplaneStorage!==report.requested.reuseDeadTriplaneStorage)throw Error('effective owned-storage reuse differs from requested demand');
            if(phase.reuseAttentionResidualStorage!==report.requested.reuseAttentionResidualStorage)throw Error('effective attention residual reuse differs from requested demand');
            let normX=true;
            if(expected.kind==='fuse-prepare'){
              const prefix='backbone.main_blocks.'+expected.block+'.fuse_block_'+expected.direction+'.norm_x';
              const weight=table.has(prefix+'.weight'),bias=table.has(prefix+'.bias');
              if(weight!==bias||phase.normX!==weight)throw Error('actual optional normalization differs from canonical complete source metadata');
              normX=weight;
            }
            demand=twoStreamPhaseDemand({name:'two-stream-duty',duty:expected,attentionRowsPerDuty:report.requested.attentionRowsPerDuty,normX,reuseDeadTriplaneStorage:report.requested.reuseDeadTriplaneStorage,reuseAttentionResidualStorage:report.requested.reuseAttentionResidualStorage});
          }else demand=twoStreamPhaseDemand(phase);
        }else if(phase.name.startsWith('post-processor-')){
          if(!throughPostProcessor)throw Error('requested route does not include postprocessor');
          demand=postWitness.postProcessorPhaseDemand(phase,report.requested.postChannelsPerDuty);
        }else demand=dinoPhaseDemand(phase);
        const processObservation=await monitor.sample({fresh:true,includeLifetimeHistory:!throughPostProcessor});
        const host={...observeMacMemory(),model:execFileSync('sysctl',['-n','hw.model'],{encoding:'utf8'}).trim(),processor:execFileSync('sysctl',['-n','machdep.cpu.brand_string'],{encoding:'utf8'}).trim()};
        const processRow=processObservation.lastObservation;
        const hostHeadroom=evaluatePhaseHostHeadroom({host,requiredBytes:demand.requiredBytes,policy:report.requested.hostHeadroomPolicy,requestedAtUnixMs:phaseRequestedAtUnixMs});
        // Full raw process journal and terminal lifetime summary retain all
        // data. Per-phase evidence needs the exact fresh row, not another copy
        // of every historical observer process accumulated by the monitor.
        const observation={phase:phase.name,phaseRequestedAtUnixMs,descriptor:phase,demand,host,hostHeadroom,
          process:throughPostProcessor?postWitness.projectFreshProcessEvidence(processObservation):processObservation,
          authority:'reversible encoder phase with observed baseline and diagnostic process stop; not production physical fit',
          verdict:host.hostname===report.source.hostname&&host.model==='Mac14,9'&&host.processor==='Apple M2 Pro'&&!host.observerErrors.length&&
            hostHeadroom.verdict==='admitted'&&processObservation.coverage==='sampled-owned-process-tree'&&processRow?.status==='observed'&&
            processRow.runId===report.runId&&processRow.rootPid===process.pid&&
            processObservation.freshness?.route==='new-probe-after-request'&&
            processObservation.freshness.requestedAtUnixMs>=phaseRequestedAtUnixMs&&
            processRow.atUnixMs>=processObservation.freshness.requestedAtUnixMs&&
            processObservation.freshness.probeAtUnixMs===processRow.atUnixMs&&
            processObservation.freshness.observationIndex===processObservation.sampleCount&&
            processRow.sampledAggregatePhysicalFootprintBytes+demand.requiredBytes<=report.requested.processBudgetBytes&&!report.memorySafety?'admitted':'refused'};
        report.phaseObservations.push(observation);report.phase='phase-'+phase.name;
        if(throughFullModel)report.expectedPhaseOrder=[...order,...artifactContract.received.map(r=>r.name)];
        await persist();
        res.writeHead(observation.verdict==='admitted'?200:409,{'Content-Type':'application/json','Cache-Control':'no-store'}).end(JSON.stringify(observation));return;
      }
      if(throughFullModel&&req.method==='POST'&&['/model.glb','/mesh.f32','/faces.u32'].includes(name)){
        const c=artifactContract.context,bytes=await readBody(req);
        if(name==='/model.glb'){
          await persistArtifactDelivery({filename:report.evidencePaths.glb,bytes,report,key:'glbOutput',persist,inspect:input=>{
            if(!artifactContract.complete||input.length!==c.glbBytes)throw Error('full GLB persistence before complete admitted consumer graph');
            return inspectCompleteGlb(input,{meshInput:report.evidencePaths.mesh,facesInput:report.evidencePaths.faces,numVertices:c.numVertices,numFaces:c.numFaces});
          }});
        }else{
          const isMesh=name==='/mesh.f32',expected=12*(isMesh?c.numVertices:c.numFaces);
          await persistArtifactDelivery({filename:isMesh?report.evidencePaths.mesh:report.evidencePaths.faces,bytes,report,key:isMesh?'meshOutput':'faceOutput',persist,inspect:input=>{
            if(report.phaseObservations.at(-1)?.phase!=='artifact-geometry-persist')throw Error('actual geometry persistence lacks its fresh new-backing gate');
            if(input.length!==expected)throw Error('partial complete geometry persistence');
            for(let i=0;i<input.length;i+=4)if(isMesh?!Number.isFinite(input.readFloatLE(i)):input.readUInt32LE(i)>=c.numVertices)throw Error('invalid actual complete mesh');
            return {};
          }});
        }
        await persist();res.writeHead(200).end();return;
      }
      if(['/triplane.f32','/postprocessor.f32'].includes(name)&&req.method==='POST'){
        const post=name==='/postprocessor.f32';
        if(post?!throughPostProcessor:!throughBackbone)throw Error('complete output not requested');
        const countKey=post?'postProcessorReadbackBytes':'twoStreamReadbackBytes';
        const outputPath=post?report.evidencePaths.postProcessorOutput:report.evidencePaths.twoStreamOutput;
        const offset=Number(req.headers['x-byte-offset']),bytes=await readBody(req),expected=post?70778880:3*1024*96*96*4;
        if(throughPostProcessor)postWitness.validateCompleteReadbackChunk({offset,bytes:bytes.length,receivedBytes:report[countKey]??0,totalBytes:expected});
        else if(offset!==(report[countKey]??0)||offset%4||!bytes.length||bytes.length%4||
          bytes.length>128*1024*4||offset+bytes.length>expected)throw Error('partial, skipped, duplicate or oversized readback chunk');
        // Append all bytes in order; no result cap, cached file or overwrite.
        if(offset===0)fs.writeFileSync(outputPath,bytes,{flag:'wx'});
        else fs.appendFileSync(outputPath,bytes);
        report[countKey]=offset+bytes.length;
        await persist();res.writeHead(200).end();return;
      }
      if(req.method==='POST'&&['/input.f32','/output.f32'].includes(name)){
        const bytes=await readBody(req),expected=name==='/input.f32'?3*512*512*4:1297*1024*4;
        if(bytes.length!==expected)throw Error('partial or conflicting full tensor shape');
        if(name==='/input.f32'){
          if(report.inputTensor)throw Error('complete transformed input already fixed');
          report.inputTensor=inspectDinoInputBytes(bytes);
        }
        fs.writeFileSync(name==='/input.f32'?report.evidencePaths.input:report.evidencePaths.output,bytes);
        await persist();
        res.writeHead(200).end();return;
      }
      res.writeHead(404).end();
    }catch(error){const message=String(error?.message??error);report.lastServerError={message,phase:report.phase};await persist();res.writeHead(409).end(message);}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  report.url='http://127.0.0.1:'+server.address().port+'/';
  report.phase='process-guard';await persist();
  monitor=await startProcessMemory({python:'/usr/bin/python3',script:path.join(root,'tools/process_memory.py'),runId:report.runId,
    rawPath:report.evidencePaths.process,maxFootprintBytes:report.requested.processBudgetBytes,onUnsafe:stopForSafety});
  if(report.requested.hostHeadroomPolicy===AVAILABLE_MEMORY_DIAGNOSTIC_POLICY){
    report.evidencePaths.hostPressure=reportPath+'.host.jsonl';
    try{hostMonitor=await startHostPressureGuard({rawPath:report.evidencePaths.hostPressure,onUnsafe:stopForSafety});}
    catch(error){report.hostPressureGuard=error.hostPressureSummary;throw error;}
  }
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
    const {createLoaderMemoryBudget,runResidentDino,runResidentTwoStream,disposeResidentTwoStream,runResidentPostProcessor,TwoStreamBackbone,preprocessImage,
      preprocessConditionImage,runResidentArtifact,disposeResidentArtifact}=await import('/dist-lib/sf3d-producer.js');
    const observe=async phase=>{const response=await fetch('/phase',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(phase)});
      if(!response.ok)throw Error('fresh phase observation refused: '+await response.text());};
    const persistTensor=async(url,bytes)=>{const response=await fetch(url,{method:'POST',body:bytes});if(!response.ok)throw Error('complete tensor persistence failed: '+await response.text());};
    const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('WebGPU adapter unavailable');
    const info=adapter.info,backend={vendor:info.vendor,architecture:info.architecture,description:info.description,isFallbackAdapter:info.isFallbackAdapter??adapter.isFallbackAdapter};
    const backendResponse=await fetch('/backend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(backend)});
    if(!backendResponse.ok)throw Error('native backend identity refused: '+await backendResponse.text());
    if(backend.isFallbackAdapter!==false||!/apple/i.test(backend.vendor))throw Error('actual nonfallback Apple adapter required');
    if(config.requested.throughFullModel)await observe({name:'device-acquisition',tensors:[],workGpuBytes:0,rangeCpuBytes:0,requiredBytes:0});
    const budget=createLoaderMemoryBudget(config.requested),device=await budget.requestOwnedDevice(adapter,{requiredLimits:{maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
    let imageLease,bitmap,staging,failure,failed=false,twoStream,postProcessor,fullArtifact,artifactHandle,decodedGlbImages,upstreamRetirement;
    const sourceGuard=scope=>config.requested.throughFullModel?phase=>observe({...phase,name:scope+'-source-'+phase.name}):undefined;
    device.pushErrorScope('validation');
    try{
      await observe({name:'preprocess',tensors:[],width:config.input.width,height:config.input.height});
      imageLease=budget.reserveCpu(80*1024*1024,'native-condition-image-and-readback');
      const imageResponse=await fetch('/image.png');bitmap=await createImageBitmap(await imageResponse.blob());
      if(bitmap.width!==config.input.width||bitmap.height!==config.input.height)throw Error('decoded canonical dimensions changed');
      const condition=config.requested.throughFullModel?await preprocessConditionImage(bitmap):{chw:await preprocessImage(bitmap)};
      const chw=condition.chw,inputBytes=new Uint8Array(chw.buffer,chw.byteOffset,chw.byteLength);
      const inputHash=await crypto.subtle.digest('SHA-256',inputBytes);
      const inputConsumed={sha256:Array.from(new Uint8Array(inputHash),b=>b.toString(16).padStart(2,'0')).join(''),bytes:inputBytes.byteLength};
      await persistTensor('/input.f32',inputBytes);
      if(config.requested.throughBackbone){
        bitmap.close();bitmap=null;imageLease.release();
        imageLease=budget.reserveCpu(chw.byteLength+(condition.rgba?.byteLength??0),'complete-preprocessed-CHW-and-condition-RGBA');
      }
      const dino=await runResidentDino({device,memoryBudget:budget,imageChw:chw,weightsUrl:'/canonical-weights.bin',
        expectedWeightBytes:config.source.byteLength,expectedSourceETag:config.source.etag,onBeforePhase:observe,
        onBeforeSourceIntake:sourceGuard('dino'),
        onProgress:p=>{document.querySelector('#status').textContent='DINO '+(p.completedItems??0)+'/24 blocks';},
        async withResult(result){
          staging=device.createBuffer({size:result.tokensBuf.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,label:'complete-dino-readback'});
          const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(result.tokensBuf,0,staging,0,result.tokensBuf.size);device.queue.submit([encoder.finish()]);
          await staging.mapAsync(GPUMapMode.READ);await persistTensor('/output.f32',new Uint8Array(staging.getMappedRange()));staging.unmap();staging.destroy();staging=null;
          if(config.requested.throughBackbone){
            const backbone=new TwoStreamBackbone(device);backbone.init();
            twoStream=await runResidentTwoStream({device,backbone,memoryBudget:budget,weightsUrl:'/canonical-weights.bin',
              expectedWeightBytes:config.source.byteLength,expectedSourceETag:config.source.etag,
              onBeforeSourceIntake:sourceGuard('two-stream'),
              imageTokensBuf:result.tokensBuf,N_img:result.N,onBeforePhase:observe,attentionRowsPerDuty:config.requested.attentionRowsPerDuty,
              reuseDeadTriplaneStorage:config.requested.reuseDeadTriplaneStorage,
              reuseAttentionResidualStorage:config.requested.reuseAttentionResidualStorage,
              async onBeforeDuty(duty,state){
                if(duty.kind==='output')return observe({name:'two-stream-output',tensors:[]});
                // Fresh observation at every new work-storage cliff; uniform
                // backing is accounted up front, not a needless host probe for
                // every allocation-free attention/FFN row dispatch.
                if(config.allocatingDutyIndices.includes(duty.dutyIndex))
                  await observe({name:'two-stream-duty-'+duty.dutyIndex,duty,tensors:[],attentionRowsPerDuty:state.attentionRowsPerDuty,
                    reuseDeadTriplaneStorage:state.reuseDeadTriplaneStorage,
                    reuseAttentionResidualStorage:state.reuseAttentionResidualStorage,
                    ...(duty.kind==='fuse-prepare'?{normX:!!state.weights.mainBlocks[duty.block][duty.direction==='in'?'fuseBlockIn':'fuseBlockOut'].normX}:{})});
              },
              onProgress:p=>{document.querySelector('#status').textContent='Actual complete two-stream backbone '+(p.completedItems??0)+' duties';},
              async withResult(output){
                const chunkBytes=128*1024*4;
                staging=device.createBuffer({size:chunkBytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,label:'complete-backbone-stream-readback'});
                for(let offset=0;offset<output.buffer.size;offset+=chunkBytes){
                  const bytes=Math.min(chunkBytes,output.buffer.size-offset),encoder=device.createCommandEncoder();
                  encoder.copyBufferToBuffer(output.buffer,offset,staging,0,bytes);device.queue.submit([encoder.finish()]);
                  await staging.mapAsync(GPUMapMode.READ);
                  const response=await fetch('/triplane.f32',{method:'POST',headers:{'X-Byte-Offset':String(offset)},body:new Uint8Array(staging.getMappedRange(),0,bytes)});
                  if(!response.ok)throw Error('complete backbone persistence failed: '+await response.text());
                  staging.unmap();
                }
                staging.destroy();staging=null;
                if(config.requested.throughPostProcessor){
                  postProcessor=await runResidentPostProcessor({
                    device,postProcessor:{device},memoryBudget:budget,
                    weightsUrl:'/canonical-weights.bin',expectedWeightBytes:config.source.byteLength,
                    expectedSourceETag:config.source.etag,triplanesBuf:output.buffer,
                    onBeforeSourceIntake:sourceGuard('post-processor'),
                    channelsPerDuty:config.requested.postChannelsPerDuty,onBeforePhase:observe,
                    onBeforeDuty:duty=>observe({name:duty.kind==='output-allocation'?
                      'post-processor-output-allocation':'post-processor-duty-'+duty.dutyIndex,duty,tensors:[]}),
                    onProgress:p=>{document.querySelector('#status').textContent='Complete postprocessor '+(p.completedItems??0)+' duties';},
                    async withResult(result){
                      await observe({name:'post-processor-output',tensors:[]});
                      const chunkBytes=128*1024*4;
                      staging=device.createBuffer({size:chunkBytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,label:'complete-postprocessor-stream-readback'});
                      for(let offset=0;offset<result.buffer.size;offset+=chunkBytes){
                        const bytes=Math.min(chunkBytes,result.buffer.size-offset),encoder=device.createCommandEncoder();
                        encoder.copyBufferToBuffer(result.buffer,offset,staging,0,bytes);device.queue.submit([encoder.finish()]);
                        await staging.mapAsync(GPUMapMode.READ);
                        const response=await fetch('/postprocessor.f32',{method:'POST',headers:{'X-Byte-Offset':String(offset)},body:new Uint8Array(staging.getMappedRange(),0,bytes)});
                        if(!response.ok)throw Error('complete postprocessor persistence failed: '+await response.text());
                        staging.unmap();
                      }
                      staging.destroy();staging=null;
                      if(config.requested.throughFullModel){
                        // The complete postprocessor prefix and raw readback
                        // no longer borrow the backbone input. Retire through
                        // its existing owner, not the borrowed post/device.
                        await disposeResidentTwoStream(backbone);
                        upstreamRetirement={consumer:'complete-postprocessor-and-persisted-readback',
                          route:'existing-owned-two-stream-disposer',backboneOutputBytes:output.buffer.size,
                          disposition:'returned',authority:'logical owned retirement only; next fresh kernel sample governs physical margin'};
                        artifactHandle={device};
                        fullArtifact=await runResidentArtifact({device,handle:artifactHandle,memoryBudget:budget,
                          triplanesBuf:result.buffer,conditionRgba:condition.rgba,weightsUrl:'/canonical-weights.bin',
                          expectedWeightBytes:config.source.byteLength,expectedSourceETag:config.source.etag,
                          onBeforePhase:observe,onBeforeDuty:observe,
                          onProgress:phase=>{document.querySelector('#status').textContent='Actual complete model: '+phase;},
                          async onGeometry(mesh){
                            const bytes=mesh.vertices.byteLength+mesh.faces.byteLength;
                            await observe({name:'artifact-geometry-persist',tensors:[],rangeCpuBytes:2*bytes,workGpuBytes:0,requiredBytes:2*bytes});
                            const lease=budget.reserveCpu(2*bytes,'actual-complete-mesh-persistence');
                            try{await persistTensor('/mesh.f32',new Uint8Array(mesh.vertices.buffer,mesh.vertices.byteOffset,mesh.vertices.byteLength));
                              await persistTensor('/faces.u32',new Uint8Array(mesh.faces.buffer,mesh.faces.byteOffset,mesh.faces.byteLength));
                            }finally{lease.release();}
                          },
                          async withResult({glb,mesh}){
                            const view=new DataView(glb),jsonLength=view.getUint32(12,true),binStart=28+jsonLength;
                            const json=JSON.parse(new TextDecoder().decode(new Uint8Array(glb,20,jsonLength)));
                            const imageViews=json.images.map(image=>json.bufferViews[image.bufferView]);
                            const imageBytes=2*Math.max(...imageViews.map(v=>v.byteLength));
                            await observe({name:'artifact-glb-image-conformance',tensors:[],rangeCpuBytes:imageBytes,workGpuBytes:0,requiredBytes:imageBytes});
                            const imageConformanceLease=budget.reserveCpu(imageBytes,'embedded-JPEG-source-copies');
                            try{decodedGlbImages=[];
                              for(const image of imageViews){const bytes=new Uint8Array(glb,binStart+image.byteOffset,image.byteLength);
                                const hash=await crypto.subtle.digest('SHA-256',bytes);
                                const decoded=await createImageBitmap(new Blob([bytes],{type:'image/jpeg'}));
                                try{if(decoded.width!==1024||decoded.height!==1024)throw Error('actual complete embedded texture dimensions changed');
                                  decodedGlbImages.push({source:'live-browser-decoded-embedded-JPEG',width:decoded.width,height:decoded.height,
                                    sha256:Array.from(new Uint8Array(hash),b=>b.toString(16).padStart(2,'0')).join('')});
                                }finally{decoded.close();}
                              }
                            }finally{imageConformanceLease.release();}
                            const persistenceBytes=2*glb.byteLength+mesh.vertices.byteLength+mesh.faces.byteLength+2*65536;
                            await observe({name:'artifact-glb-persist',tensors:[],rangeCpuBytes:persistenceBytes,workGpuBytes:0,requiredBytes:persistenceBytes});
                            const lease=budget.reserveCpu(persistenceBytes,'actual-complete-GLB-persistence-and-raw-geometry-join');
                            try{await persistTensor('/model.glb',new Uint8Array(glb));}finally{lease.release();}
                          }});
                      }
                    },
                  });
                }
              },
            });
          }
        }});
      imageLease.release();imageLease=null;const validation=await device.popErrorScope();
      if(validation)throw Error('native validation: '+validation.message);
      document.querySelector('#status').textContent=config.requested.throughFullModel?
        'Actual complete canonical model and textured GLB exported; foreground composition not yet witnessed':config.requested.throughPostProcessor?
        'Complete DINO/backbone/postprocessor; mesh/material/GLB not yet run':config.requested.throughBackbone?
        'All24 DINO blocks and complete four-block backbone completed; full mesh/material/GLB not yet run':'All24 DINO blocks completed; full SF3D not yet run';
      return {backend,dino,twoStream,postProcessor,fullArtifact,decodedGlbImages,upstreamRetirement,inputConsumed,validationError:null,budget:budget.snapshot()};
    }catch(error){failed=true;failure=error;throw error;}
    finally{
      bitmap?.close();
      try{if(artifactHandle)await disposeResidentArtifact(artifactHandle);await device.queue.onSubmittedWorkDone();staging?.destroy();imageLease?.release();device.destroy();budget.restore();}
      catch(error){throw failed?new AggregateError([failure,error],'native DINO failed and cleanup failed',{cause:failure}):error;}
    }
  },{requested:report.requested,source:report.canonicalSource,input:report.inputArtifact,
    allocatingDutyIndices:groups?.flatMap(g=>g.duties).filter(d=>twoStreamPhaseDemand({name:'two-stream-duty',duty:d,attentionRowsPerDuty:report.requested.attentionRowsPerDuty,reuseDeadTriplaneStorage:report.requested.reuseDeadTriplaneStorage,reuseAttentionResidualStorage:report.requested.reuseAttentionResidualStorage}).requiredBytes>0).map(d=>d.dutyIndex)??[]}));
  if(throughBackbone)report.twoStreamOutput=inspectTwoStreamOutput(report.evidencePaths.twoStreamOutput);
  if(throughPostProcessor)report.postProcessorOutput=postWitness.inspectPostProcessorOutput(report.evidencePaths.postProcessorOutput);
  if(throughFullModel)Object.assign(report.glbOutput,inspectCompleteGlb(report.evidencePaths.glb));
  report.output=inspectDinoOutput(report.evidencePaths.output);report.phase='complete';report.status='passed';
}catch(error){report.status='failed';report.error={message:String(error?.message??error),stack:error?.stack,lastTrustworthyPhase:report.phase};}
finally{
  report.cleanup={};
  try{if(hostMonitor)report.hostPressureGuard=await hostMonitor.stop();if(safetyStop)await safetyStop;}
  catch(error){report.cleanup.hostObserverError=error.message;report.status='failed';}
  try{if(monitor)report.processObservation=await monitor.stop();}catch(error){report.cleanup.observerError=error.message;report.status='failed';}
  try{report.cleanup.browser=child?await stopOwnedBrowser(child):{status:'not-started',exitObserved:true};}catch(error){report.cleanup.browser={error:error.message};report.status='failed';}
  try{if(server)await new Promise(resolve=>server.close(resolve));report.cleanup.server='closed';}catch(error){report.cleanup.serverError=error.message;report.status='failed';}
  closeCanonicalSource(report,source);
  try{if(profile&&report.cleanup.browser?.exitObserved)await fs.promises.rm(profile,{recursive:true});}catch(error){report.cleanup.profileError=error.message;report.status='failed';}
  report.verdict=throughFullModel?(acceptNativeArtifact?acceptNativeArtifact(report,tensorTable):{ok:false,errors:['full model witness unavailable before graph import completed']}):throughPostProcessor?(postWitness?postWitness.acceptNativeResidentPostProcessor(report):{ok:false,errors:['postprocessor witness unavailable before arguments completed']}):throughBackbone?acceptResidentTwoStream(report):acceptResidentDino(report);if(!report.verdict.ok)report.status='failed';report.terminalAt=new Date().toISOString();await persist();
  console.log(JSON.stringify({status:report.status,phase:report.phase,report:reportPath,errors:report.verdict.errors}));
  if(report.status!=='passed')process.exitCode=1;
}
