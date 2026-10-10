import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import {createHash} from 'node:crypto';
import {prepareCanonicalTensorSource,closeCanonicalSource} from './canonical_tensor_source.mjs';
import {weightFixture} from './fixtures/weight_resource_fixture.mjs';
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-canonical-range-')),p=path.join(tmp,'fixture.bin'),fixture=weightFixture();
fs.writeFileSync(p,fixture.bytes);
let source,server;
try{
  source=await prepareCanonicalTensorSource({weightsPath:p,expectedSha256:createHash('sha256').update(fixture.bytes).digest('hex'),
    tensorNames:['image_tokenizer.image_mean'],cpuBytes:1048576,gpuBytes:16384});
  server=http.createServer((req,res)=>source.serve(req,res));await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const url='http://127.0.0.1:'+server.address().port;
  for(const range of ['bytes=0-15','bytes=0-175']){
    const response=await fetch(url,{headers:{Range:range,'If-Match':source.receipt.etag}});
    assert.equal(response.status,206,'completed response must not destroy the held source descriptor needed by the next range');
    assert.equal(response.headers.get('etag'),source.receipt.etag);
    const bytes=new Uint8Array(await response.arrayBuffer());assert.deepEqual(bytes,fixture.bytes.slice(0,bytes.length));
  }
  source.close();source=null;
}finally{
  if(server)await new Promise(resolve=>server.close(resolve));
  try{source?.close();}catch{}
  fs.rmSync(tmp,{recursive:true});
}
console.log('actual Node HTTP range response ownership preserves the held canonical descriptor across requests');
const report={status:'passed',cleanup:{}};
closeCanonicalSource(report,{close(){throw Error('injected descriptor close failure');}});
assert.equal(report.status,'failed');assert.match(report.cleanup.canonicalSourceError,/descriptor close failure/);
console.log('canonical source cleanup failure lowers status without suppressing final report persistence');
