import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {parseFlatWeightHeader} from '../src/lib/flat_tensor_ranges.js';
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const stamp=stat=>[stat.dev,stat.ino,stat.size,stat.mtimeMs,stat.ctimeMs].join(':');

// Separate bit-based reference for the preserved source bytes. It does not
// call the runtime's scalar converter or share its arithmetic implementation.
function expandedWords(bytes,dtype){
  if(dtype===0)return Array.from({length:bytes.length/4},(_,i)=>bytes.readUInt32LE(4*i));
  return Array.from({length:bytes.length/2},(_,i)=>{
    const h=bytes.readUInt16LE(2*i),sign=(h&0x8000)<<16,exp=(h>>>10)&31,mant=h&1023;
    if(exp===31)return (sign|0x7f800000|(mant?0x400000:0))>>>0;
    if(exp>0)return (sign|((exp+112)<<23)|(mant<<13))>>>0;
    if(mant===0)return sign>>>0;
    const highest=31-Math.clz32(mant);
    return (sign|((highest+103)<<23)|((mant-(1<<highest))<<(23-highest)))>>>0;
  });
}

export async function prepareCanonicalTensorSource({weightsPath,expectedSha256,tensorNames,cpuBytes,gpuBytes,referenceMode='inline-words'}){
  if(!['inline-words','source-file'].includes(referenceMode))throw Error('explicit supported canonical reference mode required');
  if(!/^[a-f0-9]{64}$/.test(expectedSha256??''))throw Error('explicit expected canonical SHA256 required');
  if(!Array.isArray(tensorNames)||!tensorNames.length||new Set(tensorNames).size!==tensorNames.length)throw Error('explicit unique tensor names required');
  const fd=fs.openSync(weightsPath,'r');
  try{
    const before=fs.fstatSync(fd),identity=stamp(before),rangeCleanupFailures=[];
    let rangeCleanupFailure;
    const check=()=>{if(rangeCleanupFailure)throw rangeCleanupFailure;if(stamp(fs.fstatSync(fd))!==identity)throw Error('canonical source changed during witness');};
    const prefix=Buffer.alloc(16);if(fs.readSync(fd,prefix,0,16,0)!==16)throw Error('canonical prefix truncated');
    const headerSize=prefix.readUInt32LE(12);
    if(headerSize!==16+160*prefix.readUInt32LE(8)||headerSize>before.size||2*headerSize>cpuBytes)
      throw Error('canonical header exceeds declared tensor-unit CPU allowance');
    const header=Buffer.alloc(headerSize);if(fs.readSync(fd,header,0,headerSize,0)!==headerSize)throw Error('canonical header truncated');
    const {tensors}=parseFlatWeightHeader(header.buffer.slice(header.byteOffset,header.byteOffset+header.length),before.size);
    const units=[];let totalGpuBytes=0;
    for(const name of tensorNames){
      const info=tensors.get(name);if(!info)throw Error('canonical tensor missing: '+name);
      const expandedBytes=info.size*(info.dtype===1?2:1);totalGpuBytes+=expandedBytes;
      if(2*info.size+expandedBytes>cpuBytes||totalGpuBytes+expandedBytes>gpuBytes)
        throw Error('canonical unit/reference/readback exceeds declared diagnostic allowance');
      const bytes=Buffer.alloc(info.size);if(fs.readSync(fd,bytes,0,bytes.length,info.offset)!==bytes.length)throw Error('canonical tensor truncated');
      units.push({name,...info,sha256:digest(bytes),expandedBytes,...(referenceMode==='inline-words'?{rawHex:bytes.toString('hex'),expectedF32Words:expandedWords(bytes,info.dtype)}:{})});
    }
    const hash=createHash('sha256');
    // Hash every byte through one reusable IO backing before any GPU launch.
    // update() consumes synchronously; no source-sized succession of retained
    // chunk allocations is needed. Short reads are valid, partial EOF is not.
    const hashBuffer=Buffer.alloc(65536);
    for(let offset=0;offset<before.size;){
      const count=fs.readSync(fd,hashBuffer,0,Math.min(hashBuffer.length,before.size-offset),offset);
      if(count===0)throw Error('canonical full source truncated during hash');
      hash.update(hashBuffer.subarray(0,count));offset+=count;
    }
    const sha256=hash.digest('hex');check();
    if(sha256!==expectedSha256)throw Error('canonical artifact SHA256 mismatch');
    const etag='"sha256-'+sha256+'"';
    let closed=false;const idleResponseBuffers=[];
    return {receipt:{path:fs.realpathSync(weightsPath),sha256,byteLength:before.size,headerSha256:digest(header),headerBytes:headerSize,referenceMode,
      tensorCount:tensors.size,etag,units,totalGpuBytes,cpuBytes,gpuBytes,rangeCleanupFailures,sourceStat:{dev:before.dev,ino:before.ino,size:before.size,mtimeMs:before.mtimeMs,ctimeMs:before.ctimeMs}},
      async serve(req,res){
        let responseFd,buffer,reusable=false;
        try{
          check();const match=req.headers.range?.match(/^bytes=(\d+)-(\d+)$/);
          if(req.headers['if-match']!==etag){res.writeHead(412).end();return;}
          if(!match){res.writeHead(416).end();return;}
          const start=Number(match[1]),end=Number(match[2]);
          if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start<0||end<start||end>=before.size){res.writeHead(416).end();return;}
          // Each response owns its checked descriptor and exclusive IO backing.
          // A completed write callback releases that backing for the next read;
          // successful responses return it for later ranges of the same source.
          // Aborted/error writes never lend their backing to another response.
          responseFd=fs.openSync(weightsPath,'r');
          if(stamp(fs.fstatSync(responseFd))!==identity)throw Error('canonical range descriptor changed');
          buffer=idleResponseBuffers.pop()??Buffer.alloc(65536);
          res.on('error',()=>{});
          res.writeHead(206,{'Content-Type':'application/octet-stream','Content-Length':end-start+1,'Content-Range':`bytes ${start}-${end}/${before.size}`,ETag:etag,'Cache-Control':'no-store'});
          for(let position=start;position<=end;){
            if(res.destroyed)throw Error('canonical range response closed');
            const count=await new Promise((resolve,reject)=>fs.read(responseFd,buffer,0,
              Math.min(buffer.length,end-position+1),position,(error,count)=>error?reject(error):resolve(count)));
            if(count===0)throw Error('canonical range truncated during response');
            if(res.destroyed)throw Error('canonical range response closed');
            await new Promise((resolve,reject)=>{
              const onClose=()=>reject(Error('canonical range response closed during write'));
              res.once('close',onClose);
              try{res.write(buffer.subarray(0,count),error=>{
                res.off('close',onClose);error?reject(error):resolve();
              });}catch(error){res.off('close',onClose);reject(error);}
            });
            position+=count;
          }
          reusable=true;res.end();
        }catch(error){if(res.headersSent||res.destroyed)res.destroy(error);else res.writeHead(409).end(error.message);}
        finally{
          if(responseFd!==undefined)try{fs.closeSync(responseFd);}
          catch(error){
            // A body can be complete before descriptor retirement fails. Keep
            // the unresolved exact fd and poison all later/source-close claims;
            // never leak an async rejection past the native response owner or
            // retry an indeterminate OS close against a possibly recycled fd.
            rangeCleanupFailures.push({fd:responseFd,message:String(error.message??error),disposition:'unresolved-close; exact process exit remains owner'});
            rangeCleanupFailure??=Error('canonical range descriptor cleanup failed: '+error.message,{cause:error});
            reusable=false;if(!res.destroyed)res.destroy(rangeCleanupFailure);
          }
          if(reusable)idleResponseBuffers.push(buffer);
        }
      },close(){if(closed)return;closed=true;try{check();}finally{fs.closeSync(fd);}}};
  }catch(error){fs.closeSync(fd);throw error;}
}

export function closeCanonicalSource(report,source){
  try{source?.close();}
  catch(error){report.cleanup.canonicalSourceError=error.message;report.status='failed';}
}
