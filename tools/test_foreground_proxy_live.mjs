// Replayable HTTP conformance of the proxy against the selected real handler.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {createHash} from 'node:crypto';
import {startForegroundService} from './foreground_service.mjs';
const arg=name=>process.argv[process.argv.indexOf(name)+1];
const root=arg('--repo-root'),revision=arg('--revision'),out=arg('--out'),mode=arg('--case');
const report={status:'running',phase:'actual-proxy-start',requested:{root,revision,mode},route:'HTTP-only-actual-handler-proxy-no-GPU'};
fs.mkdirSync(out,{recursive:true});
const write=()=>fs.writeFileSync(path.join(out,'proxy-report.json'),JSON.stringify(report,null,2));
let service,proxy,other;const pending=[];write();
try{
  service=await startForegroundService({repoRoot:root,revision,outputDir:out});report.service=service.receipt;
  proxy=http.createServer((req,res)=>pending.push(service.proxy(req,res)));
  await new Promise(resolve=>proxy.listen(0,'127.0.0.1',resolve));
  const origin='http://127.0.0.1:'+proxy.address().port;
  if(mode==='concurrent'){
    report.phase='concurrent-retained-bytes';
    const urls=['/index.html','/volume-settings-preset-schema-v2.json','/api/runtime-config'];
    const responses=await Promise.all(urls.map(async url=>{const response=await fetch(origin+url);assert.equal(response.status,200);return Buffer.from(await response.arrayBuffer());}));
    await Promise.all(pending);
    const records=service.receipt.network;
    assert.equal(new Set(records.map(r=>r.rawPath)).size,3,'each concurrent response needs a unique retained path');
    for(let i=0;i<3;i++){
      const record=records.find(r=>r.url===urls[i]);assert.equal(record.complete,true);
      const retained=fs.readFileSync(record.rawPath);assert.deepEqual(retained,responses[i]);
      assert.equal(record.sha256,createHash('sha256').update(retained).digest('hex'));
    }
  }else if(mode==='write-failure'){
    report.phase='retention-failure-authority';
    // EISDIR is a real evidence write failure, not a mocked successful stream.
    fs.mkdirSync(path.join(out,'response-1.raw'));
    try{await(await fetch(origin+'/api/runtime-config')).arrayBuffer();}catch{}
    await Promise.all(pending);
    const record=service.receipt.network[0];
    assert.equal(record.complete,false,'failed raw output cannot retain a complete claim');
    assert.match(record.evidenceError,/EISDIR/);
  }else if(mode==='origin'){
    report.phase='private-origin-confinement';let contacts=0;
    other=http.createServer((_req,res)=>{contacts++;res.end('wrong-origin');});
    await new Promise(resolve=>other.listen(0,'127.0.0.1',resolve));
    const target='//127.0.0.1:'+other.address().port+'/api/read?root=pipeline-runs&path=weights.bin';
    const response=await new Promise((resolve,reject)=>{http.get(origin+target,res=>{res.resume();res.on('end',()=>resolve(res));}).on('error',reject);});
    await Promise.all(pending);
    assert.equal(response.statusCode,409,'authority-bearing target must be refused');
    assert.equal(contacts,0,'second origin must receive no request');report.secondOriginContacts=contacts;
  }else if(mode==='layout'){
    report.phase='actual-owned-layout-handler';
    const response=await fetch(origin+'/api/volume-cockpit-layouts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({layout:{},activate:false})});
    const body=await response.json();await Promise.all(pending);
    assert.equal(response.status,400,'actual handler must validate the supplied layout, not refuse the write route');
    assert.equal(body.storePath,path.join(out,'layouts'));
    assert.equal(body.failurePhase,'cockpit-layout-write');
    assert.notEqual(body.error,'cockpit layout write requires exactly layout and activate inputs','proxy must preserve POST body/content length');
    report.layoutValidation=body;
  }else throw Error('explicit --case concurrent|write-failure|origin|layout required');
  report.status='passed';
}catch(error){report.status='failed';report.error={message:error.message,stack:error.stack};}
finally{
  if(proxy)await new Promise(resolve=>proxy.close(resolve));
  if(other)await new Promise(resolve=>other.close(resolve));
  if(service)report.cleanup=await service.close();write();
}
console.log(JSON.stringify({status:report.status,mode,report:path.join(out,'proxy-report.json'),error:report.error?.message}));
if(report.status!=='passed')process.exitCode=1;
