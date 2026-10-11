import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import {syncBuiltinESMExports} from 'node:module';
import {weightFixture} from './fixtures/weight_resource_fixture.mjs';

const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-source-hash-backing-'));
const file=path.join(tmp,'weights.bin'),fixture=weightFixture();
const bytes=Buffer.concat([fixture.bytes,Buffer.alloc(3*65536+37,49)]);
const originalHash=crypto.createHash,originalRead=fs.readSync;
const expected=originalHash('sha256').update(bytes).digest('hex'),records=[];
crypto.createHash=function(...args){
  const hash=originalHash(...args),update=hash.update,digest=hash.digest;
  const row={bytes:0,backings:new Set(),digest:null};records.push(row);
  hash.update=function(chunk,...rest){if(ArrayBuffer.isView(chunk)){row.bytes+=chunk.byteLength;row.backings.add(chunk.buffer);}return update.call(this,chunk,...rest);};
  hash.digest=function(...rest){const value=digest.call(this,...rest);row.digest=value;return value;};
  return hash;
};
syncBuiltinESMExports();
let source;
try{
  fs.writeFileSync(file,bytes);
  const {prepareCanonicalTensorSource}=await import('./canonical_tensor_source.mjs');
  const args={weightsPath:file,expectedSha256:expected,tensorNames:['image_tokenizer.image_mean'],cpuBytes:1048576,gpuBytes:16384,referenceMode:'source-file'};
  source=await prepareCanonicalTensorSource(args);
  const full=records.find(row=>row.digest===expected&&row.bytes===bytes.length);
  assert.ok(full,'every source byte must reach the independent full-file SHA256');
  assert.equal(full.backings.size,1,'full-file hashing must reuse one backing, not retain/churn a new source-sized succession of chunks');
  assert.equal([...full.backings][0].byteLength,65536,'hash backing must remain one 64KiB IO buffer, without capping source bytes');
  assert.equal(source.receipt.byteLength,bytes.length);assert.equal(source.receipt.sha256,expected);
  source.close();source=null;
  fs.readSync=function(fd,buffer,offset,length,position){return originalRead(fd,buffer,offset,buffer.length===65536?Math.min(length,6000):length,position);};
  syncBuiltinESMExports();
  source=await prepareCanonicalTensorSource(args);
  assert.equal(source.receipt.sha256,expected,'legal short reads must still hash the complete ordered source');
  source.close();source=null;
  fs.readSync=function(fd,buffer,offset,length,position){
    if(buffer.length===65536&&position>=65536)return 0;
    return originalRead(fd,buffer,offset,length,position);
  };
  syncBuiltinESMExports();
  await assert.rejects(prepareCanonicalTensorSource(args),/canonical full source truncated during hash/,'early EOF must not become a partial canonical digest');
}finally{
  try{source?.close();}catch{}
  crypto.createHash=originalHash;fs.readSync=originalRead;syncBuiltinESMExports();
  fs.rmSync(tmp,{recursive:true});
}
console.log('complete canonical SHA256 uses one reused backing, handles short reads and refuses partial EOF');
