// Non-allocating source-serving witness: HEAD never consumes model payload.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createServer} from 'vite';
import {execFileSync} from 'node:child_process';
const root=path.resolve(new URL('..',import.meta.url).pathname);
const reportPath=path.resolve(process.argv[2]??path.join(os.tmpdir(),'sf3d-m2-source-refusal-'+Date.now()+'.json'));
const report={schema:'sf3d.m2-source-refusal.v0',status:'running',phase:'host-identity',repoRoot:root,
  payloadConsumed:false,requestedMethod:'HEAD',requestedPath:'/weights.bin',hostname:os.hostname()};
let server;
try{
  report.revision=execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim();
  report.clean=execFileSync('git',['status','--porcelain'],{cwd:root,encoding:'utf8'}).trim()==='';
  const expectedIndex=process.argv.indexOf('--expected-revision');
  if(expectedIndex>=0){report.expectedRevision=process.argv[expectedIndex+1];assert.equal(report.revision,report.expectedRevision);assert.equal(report.clean,true,'closing native source witness requires clean exact source');}
  report.machine={platform:process.platform,totalBytes:os.totalmem(),model:execFileSync('sysctl',['-n','hw.model'],{encoding:'utf8'}).trim()};
  if(report.machine.platform!=='darwin'||report.machine.model!=='Mac14,9')throw Error('this native witness requires the actual M2 Pro host');
  report.phase='server';server=await createServer({root,configFile:path.join(root,'vite.config.js'),server:{host:'127.0.0.1',port:0,open:false}});await server.listen();
  report.url='http://127.0.0.1:'+server.httpServer.address().port+'/weights.bin';
  report.phase='source-request';const response=await fetch(report.url,{method:'HEAD'});
  report.response={status:response.status,contentLength:response.headers.get('content-length'),authority:response.headers.get('x-sf3d-memory-authority')};
  assert.equal(response.status,503,'actual M2 app source route must refuse the unadmitted full model before serving payload');
  assert.equal(report.response.authority,'circuit-breaker-only');report.status='passed';
  report.requests=[];
  for(const url of ['/weights.bin?raw','/%77eights.bin','/public/weights.bin','/@fs/'+fs.realpathSync(path.join(root,'public/weights.bin'))]){
    const response=await fetch(new URL(url,report.url));
    if(response.status!==503){await response.body?.cancel();assert.fail('unadmitted model alias served: '+url);}
    const receipt=await response.json();assert.equal(receipt.verdict,'refused');assert.equal(receipt.authority,'circuit-breaker-only');
    report.requests.push({method:'GET',url,status:response.status,receipt});
  }
}catch(error){report.status='failed';report.error={message:error.message,stack:error.stack};throw error;}
finally{await server?.close();fs.mkdirSync(path.dirname(reportPath),{recursive:true});fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+'\n');}
