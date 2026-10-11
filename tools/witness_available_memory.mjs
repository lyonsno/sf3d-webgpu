#!/usr/bin/env node
// Read-only effective-route conformance; never grants allocation authority.
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {observeMacMemory,evaluatePhaseHostHeadroom,AVAILABLE_MEMORY_DIAGNOSTIC_POLICY} from './memory_admission.mjs';
import {writeJsonReportAtomic} from './json_report_atomic.mjs';
const output=process.argv[2]?path.resolve(process.argv[2]):null;
if(!output)throw Error('caller-owned conformance report path required');
if(fs.existsSync(output))throw Error('existing conformance evidence retained; choose a new output path');
fs.mkdirSync(path.dirname(output),{recursive:true});
const report={schema:'sf3d.available-memory-conformance.v0',status:'running',phase:'runtime',output,
  runtime:{node:process.version,libuv:process.versions.uv,executable:fs.realpathSync(process.execPath),platform:process.platform},
  authority:'live effective-route observation against pinned canonical sources; not physical WebGPU capacity or model admission'};
const persist=()=>writeJsonReportAtomic(output,report);
try{
  await persist();
  if(process.platform!=='darwin'||process.version!=='v25.9.0'||process.versions.uv!=='1.52.1')throw Error('canonical observed Node25.9/libuv1.52.1 Darwin route required; no fallback');
  report.phase='canonical-sources';await persist();
  report.sources=[];
  const urls=[['libuv-darwin.c','https://raw.githubusercontent.com/libuv/libuv/v1.52.1/src/unix/darwin.c'],
    ['node-process-methods.cc','https://raw.githubusercontent.com/nodejs/node/v25.9.0/src/node_process_methods.cc']];
  const bodies=[];
  for(const [name,url] of urls){
    const response=await fetch(url);if(!response.ok||response.url!==url)throw Error('exact canonical source HTTP route failed: '+url);
    const bytes=Buffer.from(await response.arrayBuffer()),target=output+'.'+name;
    fs.writeFileSync(target,bytes,{flag:'wx'});report.sources.push({url,effectiveUrl:response.url,path:target,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});
    bodies.push(bytes.toString('utf8'));await persist();
  }
  const available=bodies[0].match(/uint64_t uv_get_available_memory\(void\)\s*\{([\s\S]*?)\n\}/)?.[1];
  const free=bodies[0].match(/uint64_t uv_get_free_memory\(void\)\s*\{([\s\S]*?)\n\}/)?.[1];
  const binding=bodies[1].match(/(?:static )?void GetAvailableMemory\([^]*?\n\}/)?.[0];
  if(!available?.includes('info.free_count + (uint64_t) info.inactive_count + (uint64_t) info.purgeable_count')||
    !available.includes('sysconf(_SC_PAGESIZE)')||!available.includes('return 0;')||
    !free?.includes('info.free_count * sysconf(_SC_PAGESIZE)')||!binding?.includes('uv_get_available_memory()'))throw Error('canonical field or Node binding contract drift');
  report.phase='live-observation';await persist();
  const rawCommand=command=>({command,at:new Date().toISOString(),stdout:execFileSync(command,[],{encoding:'utf8'})});
  report.before=rawCommand('vm_stat');const requestedAtUnixMs=Date.now(),calls=[];
  const native=process.availableMemory;
  report.observation=observeMacMemory({availableMemory:()=>{const atUnixMs=Date.now(),bytes=native.call(process);calls.push({atUnixMs,bytes});return bytes;}});
  report.calls=calls;report.after=rawCommand('vm_stat');
  report.decision=evaluatePhaseHostHeadroom({host:report.observation,requiredBytes:0,policy:AVAILABLE_MEMORY_DIAGNOSTIC_POLICY,requestedAtUnixMs});
  if(calls.length!==1||calls[0].bytes!==report.observation.availableMemory.bytes||report.decision.verdict!=='admitted')throw Error('live direct field binding, freshness or pressure witness failed');
  report.limit='vm_stat snapshots and Node observation are sequential, not atomic; availability is an OS estimate, not demonstrated unified-GPU headroom';
  report.status='passed';report.phase='complete';
}catch(error){report.status='failed';report.error={message:error.message,lastTrustworthyPhase:report.phase};process.exitCode=1;}
finally{report.terminalAt=new Date().toISOString();await persist();console.log(JSON.stringify({status:report.status,phase:report.phase,report:output,available:report.observation?.availableMemory,decision:report.decision,error:report.error}));}
