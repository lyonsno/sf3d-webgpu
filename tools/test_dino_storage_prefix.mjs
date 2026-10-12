import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {createNativeDinoObservation} from './native_dino_observation.mjs';
const sha=b=>createHash('sha256').update(b).digest('hex');
const bytes=Buffer.alloc(4);bytes.writeFloatLE(1);
const info={name:'camera_embedder.linear.weight',shape:[1],size:2,dtype:1,offset:0};
const phase={phase:'camera',verdict:'admitted',descriptor:{name:'camera',tensors:[info]}};
const request={kind:'weights',sourceETag:'"source"',tensors:[{...info,byteLength:4}]};
for(const failure of ['throw','zero']){
 if(process.argv.includes('--zero-only')&&failure!=='zero')continue;
 const out=fs.mkdtempSync(path.join(os.tmpdir(),'dino-prefix-'+failure+'-'));
 const c=createNativeDinoObservation({table:new Map([[info.name,info]]),sourceETag:'"source"',
  expectedWeightHashes:new Map([[info.name,sha(bytes)]]),outputDir:out});
 const admit=async()=>({verdict:'admitted'});await c.admit(request,phase,admit);await c.admitTensor(0,0,admit);
 const real=fs.writeSync;let calls=0;
 fs.writeSync=(fd,chunk,offset)=>{
  calls++;if(calls===1)return real(fd,chunk,offset,1);
  if(failure==='zero')return 0;
  throw Error('injected storage failure after actual positive short write');
 };
 try{await assert.rejects(c.persistTensor(0,0,bytes),/storage failure|partial raw evidence write/);}
 finally{fs.writeSync=real;}
 assert.equal(calls,2,'no retry after indeterminate failure');
 const failed=c.observations[0].failedDeliveries[0],retained=fs.readFileSync(failed.path);
 assert.equal(retained.length,1);assert.equal(failed.bytes,retained.length,'receipt counts actual stored prefix');
 assert.equal(failed.sha256,sha(retained),'receipt hashes actual stored prefix');
 assert.equal(failed.receivedBytes,4,'received and retained bytes remain distinct');
 assert.deepEqual(retained,bytes.subarray(0,1));assert.equal(c.observations[0].outputs.length,0);
 assert.equal(c.observations[0].status,'failed');
}
console.log('PASS actual short-write failure/zero receipts retain exact prefix hash/count, no retry or success');
