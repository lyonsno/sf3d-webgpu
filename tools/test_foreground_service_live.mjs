// HTTP-only conformance against the actual pinned serving implementation.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {startForegroundService,validateForegroundService} from './foreground_service.mjs';
const arg=name=>{const i=process.argv.indexOf(name);return i<0?null:process.argv[i+1];};
const root=arg('--repo-root'),revision=arg('--revision'),out=arg('--out');
if(!root||!revision||!out)throw Error('explicit --repo-root --revision --out required');
fs.mkdirSync(out,{recursive:true});
const report={status:'running',phase:'actual-server-start',requested:{root,revision},route:'HTTP-only-real-Kaminos-handler-no-GPU'};
const write=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));
let service;write();
try{
  service=await startForegroundService({repoRoot:root,revision,outputDir:out});report.service=service.receipt;
  report.phase='actual-schema-and-api-conformance';
  const responses=[];report.responses=responses;
  for(const url of ['/index.html','/volume-settings-preset-schema-v2.json','/api/runtime-config','/api/forge-host/registry','/node_modules/three-mesh-bvh/build/index.module.js']){
    const response=await fetch(service.receipt.origin+url),bytes=Buffer.from(await response.arrayBuffer());
    fs.writeFileSync(path.join(out,'observed-'+responses.length+'.raw'),bytes);
    responses.push({url,status:response.status,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});
    assert.equal(response.status,200,url);
    if(url==='/index.html'){
      const size=Number(execFileSync('git',['cat-file','-s',revision+':index.html'],{cwd:root,encoding:'utf8'}));
      const committed=execFileSync('git',['show',revision+':index.html'],{cwd:root,maxBuffer:size});
      assert.equal(bytes.length,size);assert.deepEqual(bytes,committed);
    }
    if(url==='/volume-settings-preset-schema-v2.json')assert.equal(JSON.parse(bytes).identity,'kaminos-volume-settings-preset-schema-v2');
  }
  const effective=service.receipt.effective;
  for(const mutate of [x=>x.source.repoRoot='/wrong',x=>x.source.commit='a'.repeat(40),x=>x.sharedBasinStore='/shared',x=>delete x.volumeSettingsStore]){
    const bad=structuredClone(effective);mutate(bad);assert.throws(()=>validateForegroundService(bad,{repoRoot:root,revision,outputDir:out}),/identity mismatch/);
  }
  report.phase='actual-private-origin-refusal';
  report.refusals=[];
  for(const [url,method]of [['/lib/sf3d/weights.bin','GET'],['/lib/sf3d/%77eights%2ebin','GET'],['/api/read?root=pipeline-runs&path=weights.bin','GET'],['/api/delete-scene?name=x','GET'],['/api/run-pipeline','POST']]){
    const response=await fetch(service.receipt.origin+url,{method});const body=await response.text();
    report.refusals.push({url,method,status:response.status,body});assert.equal(response.status,409);
  }
  report.status='passed';
}catch(error){report.status='failed';report.error={message:error.message,stack:error.stack};}
finally{if(service)report.cleanup=await service.close();report.terminalAt=new Date().toISOString();write();}
console.log(JSON.stringify({status:report.status,report:path.join(out,'report.json'),error:report.error?.message}));
if(report.status!=='passed')process.exitCode=1;
