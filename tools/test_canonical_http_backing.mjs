import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import {createHash} from 'node:crypto';
import {prepareCanonicalTensorSource} from './canonical_tensor_source.mjs';
import {weightFixture} from './fixtures/weight_resource_fixture.mjs';
const hash=b=>createHash('sha256').update(b).digest('hex');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'sf3d-http-backing-'));
const file=path.join(tmp,'weights.bin');
const bytes=Buffer.concat([Buffer.from(weightFixture().bytes),Buffer.alloc(5*65536+197,0xa7)]);
fs.writeFileSync(file,bytes);
const expected=hash(bytes);
const originalRead=fs.read;
let source,server,requests=[],shortReads=false,earlyEof=false,writeFailure=false;
try{
  source=await prepareCanonicalTensorSource({weightsPath:file,expectedSha256:expected,
    tensorNames:['image_tokenizer.image_mean'],cpuBytes:1048576,gpuBytes:16384});
  // Real Node HTTP writes, with a deliberately delayed completion callback.
  // An overwrite before that callback corrupts this independent snapshot.
  server=http.createServer((req,res)=>{
    const observation={backings:new Set(),chunks:0,callbackBytes:[]};requests.push(observation);
    observation.finished=new Promise(resolve=>res.once('finish',resolve));
    const write=res.write.bind(res);
    res.write=(chunk,callback)=>{
      observation.backings.add(chunk.buffer);observation.chunks++;
      const snapshot=Buffer.from(chunk);
      return write(chunk,error=>setTimeout(()=>{
        try{assert.deepEqual(chunk,snapshot,'response must not overwrite a backing before its write callback');}
        catch(e){callback?.(e);return;}
        observation.callbackBytes.push(Buffer.from(chunk));callback?.(writeFailure?Error('injected transport write failure'):error);
      },2));
    };
    source.serve(req,res);
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const url='http://127.0.0.1:'+server.address().port;
  const get=async(start,end)=>{
    const response=await fetch(url,{headers:{Range:'bytes='+start+'-'+end,'If-Match':source.receipt.etag}});
    assert.equal(response.status,206);assert.equal(response.headers.get('etag'),source.receipt.etag);
    assert.equal(response.headers.get('content-range'),'bytes '+start+'-'+end+'/'+bytes.length);
    assert.equal(Number(response.headers.get('content-length')),end-start+1);
    const result=Buffer.from(await response.arrayBuffer());
    assert.equal(hash(result),hash(bytes.subarray(start,end+1)));return result;
  };
  await get(0,bytes.length-1);
  await requests[0].finished;
  assert.ok(requests[0].chunks>3);
  assert.equal(requests[0].backings.size,1,'complete HTTP range must reuse one response-owned backing');
  assert.equal(hash(Buffer.concat(requests[0].callbackBytes)),expected);
  fs.read=(fd,buffer,offset,length,position,callback)=>{
    if(earlyEof&&position>=65536){callback(null,0,buffer);return;}
    return originalRead(fd,buffer,offset,shortReads?Math.min(length,6000):length,position,callback);
  };
  shortReads=true;await get(17,bytes.length-19);shortReads=false;
  assert.equal([...requests[0].backings][0],[...requests[1].backings][0],
    'completed sequential ranges reuse the released source-owned backing');
  const start=requests.length;
  await Promise.all([get(0,bytes.length-1),get(13,bytes.length-37)]);
  assert.equal(requests[start].backings.size,1);assert.equal(requests[start+1].backings.size,1);
  assert.notEqual([...requests[start].backings][0],[...requests[start+1].backings][0],
    'concurrent responses must not share mutable IO storage');
  for(const headers of [
    {Range:'bytes=0-15','If-Match':'wrong'},
    {Range:'bytes=0-'+bytes.length,'If-Match':source.receipt.etag},
    {Range:'bytes=9-1','If-Match':source.receipt.etag},
    {'If-Match':source.receipt.etag}]){
    const response=await fetch(url,{headers});assert.ok([412,416].includes(response.status));await response.text();
  }
  // Actual transport abort must close only the response, not the source anchor.
  await new Promise(resolve=>{
    const request=http.get(url,{headers:{Range:'bytes=0-'+(bytes.length-1),'If-Match':source.receipt.etag}},response=>{
      response.once('data',()=>{response.destroy();resolve();});
    });request.on('error',resolve);
  });
  await get(0,15);
  earlyEof=true;
  await assert.rejects(get(0,bytes.length-1),/terminated|socket|other side|fetch failed/i);
  earlyEof=false;await get(0,15);
  writeFailure=true;
  await assert.rejects(get(0,bytes.length-1),/terminated|socket|other side|fetch failed/i);
  writeFailure=false;await get(0,15);
  const stamp=fs.statSync(file);fs.utimesSync(file,stamp.atime,new Date(stamp.mtimeMs+1000));
  const drift=await fetch(url,{headers:{Range:'bytes=0-15','If-Match':source.receipt.etag}});
  assert.equal(drift.status,409);assert.match(await drift.text(),/source changed/);
  assert.throws(()=>source.close(),/source changed/);source=null;
}finally{
  fs.read=originalRead;
  if(server)await new Promise(resolve=>server.close(resolve));
  try{source?.close();}catch{}
  fs.rmSync(tmp,{recursive:true,force:true});
}
console.log('actual Node HTTP complete range: one response backing, awaited write ownership, full ordered bytes, short reads, concurrent separation, abort/EOF/drift refusal and anchor continuity');
