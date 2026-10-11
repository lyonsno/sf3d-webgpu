import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import {createHash} from 'node:crypto';
import {prepareCanonicalTensorSource,closeCanonicalSource} from './canonical_tensor_source.mjs';
import {weightFixture} from './fixtures/weight_resource_fixture.mjs';
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-response-close-'));
const file=path.join(tmp,'weights.bin'),fixture=weightFixture();
fs.writeFileSync(file,fixture.bytes);
const open=fs.openSync,close=fs.closeSync;
let source,server,responseFd,escaped=null,settled,anchorClosed=false;
try{
  source=await prepareCanonicalTensorSource({weightsPath:file,
    expectedSha256:createHash('sha256').update(fixture.bytes).digest('hex'),
    tensorNames:['image_tokenizer.image_mean'],cpuBytes:1048576,gpuBytes:16384});
  fs.openSync=(...args)=>{const fd=open(...args);if(args[0]===file)responseFd=fd;return fd;};
  fs.closeSync=fd=>{if(fd===responseFd)throw Error('injected response descriptor close failure');return close(fd);};
  const completed=new Promise(resolve=>{settled=resolve;});
  server=http.createServer((req,res)=>{
    Promise.resolve(source.serve(req,res)).then(settled,error=>{escaped=error;settled();});
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const url='http://127.0.0.1:'+server.address().port;
  const headers={Range:'bytes=0-15','If-Match':source.receipt.etag};
  // The body may already have reached the client when cleanup fails. Either
  // transport outcome is lawful; source/terminal authority must still fail.
  try{const r=await fetch(url,{headers});await r.arrayBuffer();}catch{}
  await completed;
  assert.equal(escaped,null,'response descriptor failure must not escape an unawaited native handler');
  fs.openSync=open;fs.closeSync=close;
  assert.equal(source.receipt.rangeCleanupFailures.length,1);
  assert.equal(source.receipt.rangeCleanupFailures[0].fd,responseFd);
  assert.match(source.receipt.rangeCleanupFailures[0].message,/descriptor close failure/);
  const later=await fetch(url,{headers});assert.equal(later.status,409);
  assert.match(await later.text(),/range descriptor cleanup failed/);
  const report={status:'passed',cleanup:{}};
  closeCanonicalSource(report,source);anchorClosed=true;source=null;
  assert.equal(report.status,'failed','completed bytes cannot conceal unresolved source cleanup');
  assert.match(report.cleanup.canonicalSourceError,/range descriptor cleanup failed/);
}finally{
  fs.openSync=open;fs.closeSync=close;
  if(server)await new Promise(resolve=>server.close(resolve));
  try{source?.close();}catch{}
  // Injection threw before the real close, so this exact fixture descriptor
  // remains owned/open and may be closed without guessing an OS failure.
  if(responseFd!==undefined)try{close(responseFd);}catch{}
  fs.rmSync(tmp,{recursive:true,force:true});
}
assert.equal(anchorClosed,true);
console.log('actual Node HTTP response close failure stays owned, poisons later ranges and lowers terminal source authority without escaping cleanup');
