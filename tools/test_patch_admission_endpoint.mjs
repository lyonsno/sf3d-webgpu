// Execute the actual HTTP branch without launching Chromium or allocating GPU
// memory. Injected host data tests local endpoint policy, not Darwin capacity.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import {writeJsonReportAtomic} from './json_report_atomic.mjs';
const source=fs.readFileSync(new URL('./smoke_loader_memory.mjs',import.meta.url),'utf8');
const start=source.indexOf("    if(patchMode&&name==='/phase-admission'){");
const end=source.indexOf("    if(patchMode&&name==='/phase-image')");
assert.ok(start>=0&&end>start,'actual phase HTTP admission branch must be exercised');
const handler=vm.compileFunction(source.slice(start,end),['patchMode','name','report','observeMacMemory','execFileSync','writeJsonReportAtomic','reportPath','res']);
const out=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-phase-admission-'));
try{
  const reportPath=path.join(out,'report.json');
  const report={source:{hostname:'owner'},phaseDemand:{requiredBytes:100},processObservation:{coverage:'sampled-owned-process-tree',lastObservation:{status:'observed'}}};
  const probe=()=>({hostname:'owner',hostFreeBytes:200,observerErrors:[]});
  const exec=(_tool,args)=>args.includes('hw.model')?'Mac14,9':'Apple M2 Pro';
  const response=()=>({code:null,body:null,writeHead(code){this.code=code;return this;},end(body){this.body=body;return this;}});
  const res=response();handler(true,'/phase-admission',report,probe,exec,writeJsonReportAtomic,reportPath,res);
  assert.equal(res.code,200,'healthy actual admission endpoint must persist and respond before learned allocation');
  assert.equal(JSON.parse(res.body).verdict,'admitted');assert.equal(JSON.parse(fs.readFileSync(reportPath)).phaseAdmission.verdict,'admitted');
  const refused=response();handler(true,'/phase-admission',report,()=>({...probe(),hostFreeBytes:99}),exec,writeJsonReportAtomic,reportPath,refused);
  assert.equal(refused.code,409);assert.equal(JSON.parse(refused.body).verdict,'refused');
  const io=response();handler(true,'/phase-admission',report,probe,exec,()=>{throw Error('report persistence failed');},reportPath,io);
  assert.equal(io.code,409);assert.match(io.body,/report persistence failed/,'failed persistence cannot authorize browser allocation');
}finally{fs.rmSync(out,{recursive:true,force:true});}
console.log('Actual phase HTTP endpoint persists and returns admission; low memory and report failure refuse before learned allocation.');
