import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {modelSourceCircuitBreaker} from './model_source_circuit_breaker.mjs';
const root=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-source-breaker-'));
const observed={platform:'darwin',hostTotalBytes:16*1024**3,source:'live-macos'};
const result={verdict:'would-admit',effective:{planId:'darwin-16gib-full-route-circuit-breaker-v0'},decision:{reasons:[]}};
try{
  fs.mkdirSync(path.join(root,'public'));fs.writeFileSync(path.join(root,'public/weights.bin'),'local policy fixture, not canonical weights');
  fs.symlinkSync('weights.bin',path.join(root,'public/alias.bin'));
  const exercise=({url='/weights.bin',method='GET',observation=observed,admit=()=>result,preview=false,base='/'}={})=>{
    let handler,passed=false;const plugin=modelSourceCircuitBreaker({observe:()=>observation,admit});
    plugin.configResolved({root,publicDir:path.join(root,'public'),base,rawBase:base});
    const returned=plugin[preview?'configurePreviewServer':'configureServer']({middlewares:{use(fn){handler=fn;return ()=>assert.fail('Connect app must not become a Vite post-hook');}}});
    assert.equal(returned,undefined,'Vite configure hook must not return the Connect app as a post-hook');
    const res={code:null,headers:null,body:null,writeHead(code,headers){this.code=code;this.headers=headers;return this;},end(body){this.body=body;}};
    handler({url,method},res,()=>{passed=true;});return{passed,...res};
  };
  for(const url of ['/weights.bin','/weights.bin?raw','/public/weights.bin','/alias.bin','/%77eights.bin','/@fs/'+path.join(root,'public/weights.bin')]){
    const r=exercise({url});assert.equal(r.code,503,url);assert.equal(r.passed,false);const receipt=JSON.parse(r.body);
    assert.equal(receipt.memoryAdmission.verdict,'refused');assert.equal(receipt.memoryAdmission.diagnosticVerdict,'would-admit');
    assert.equal(receipt.sourcePath,fs.realpathSync(path.join(root,'public/weights.bin')));assert.equal(receipt.repoRoot,root);
  }
  const head=exercise({method:'HEAD',preview:true});assert.equal(head.code,503);assert.equal(head.body,undefined);
  for(const preview of [false,true])for(const effectivePath of ['/weights.bin','/%77eights.bin','/alias.bin','/public/weights.bin','/@fs/'+path.join(root,'public/weights.bin')]){
    const url='/sf3d'+effectivePath,r=exercise({url,preview,base:'/sf3d/'});
    assert.equal(r.code,503,'Vite base-prefixed model must refuse before later base stripping: '+url);
    const receipt=JSON.parse(r.body);assert.equal(receipt.requestedPath,decodeURIComponent(url));
    assert.equal(receipt.effectivePath,decodeURIComponent(effectivePath));assert.equal(receipt.base,'/sf3d/');
  }
  assert.equal(head.headers['X-SF3D-Memory-Authority'],'circuit-breaker-only');
  assert.equal(exercise({observation:{...observed,hostTotalBytes:64*1024**3}}).passed,true,'unmatched source host preserves existing route, not positive M2 authority');
  assert.equal(exercise({url:'/demo_chair.png'}).passed,true,'ordinary assets remain unaffected');
  for(const observation of [{...observed,hostTotalBytes:null},{...observed,source:'replay-fixture'}])
    assert.equal(exercise({observation}).code,503,'unknown/replayed host must not become an unmatched-host permission');
  const failed=exercise({admit:()=>{throw Error('observer/source unavailable');}});assert.equal(failed.code,503);assert.match(failed.body,/observer\/source unavailable/);
  const lostAuthority=exercise({admit:()=>({...result,effective:{planId:'wrong'}})});assert.equal(lostAuthority.code,503);assert.match(lostAuthority.body,/lost refusal authority/);
}finally{fs.rmSync(root,{recursive:true,force:true});}
console.log('M2 source breaker refuses before payload on dev/preview, source aliases and diagnostic would-admit; unrelated assets/unmatched host remain separate.');
