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

export async function prepareCanonicalTensorSource({weightsPath,expectedSha256,tensorNames,cpuBytes,gpuBytes}){
  if(!/^[a-f0-9]{64}$/.test(expectedSha256??''))throw Error('explicit expected canonical SHA256 required');
  if(!Array.isArray(tensorNames)||!tensorNames.length||new Set(tensorNames).size!==tensorNames.length)throw Error('explicit unique tensor names required');
  const fd=fs.openSync(weightsPath,'r');
  try{
    const before=fs.fstatSync(fd),identity=stamp(before);
    const check=()=>{if(stamp(fs.fstatSync(fd))!==identity)throw Error('canonical source changed during witness');};
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
      units.push({name,...info,rawHex:bytes.toString('hex'),sha256:digest(bytes),expandedBytes,expectedF32Words:expandedWords(bytes,info.dtype)});
    }
    const hash=createHash('sha256');
    for await(const chunk of fs.createReadStream(weightsPath,{fd,autoClose:false,start:0}))hash.update(chunk);
    const sha256=hash.digest('hex');check();
    if(sha256!==expectedSha256)throw Error('canonical artifact SHA256 mismatch');
    const etag='"sha256-'+sha256+'"';
    return {receipt:{path:fs.realpathSync(weightsPath),sha256,byteLength:before.size,headerSha256:digest(header),headerBytes:headerSize,
      tensorCount:tensors.size,etag,units,totalGpuBytes,cpuBytes,gpuBytes,sourceStat:{dev:before.dev,ino:before.ino,size:before.size,mtimeMs:before.mtimeMs,ctimeMs:before.ctimeMs}},
      serve(req,res){
        try{
          check();const match=req.headers.range?.match(/^bytes=(\d+)-(\d+)$/);
          if(req.headers['if-match']!==etag){res.writeHead(412).end();return;}
          if(!match){res.writeHead(416).end();return;}
          const start=Number(match[1]),end=Number(match[2]);
          if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start<0||end<start||end>=before.size){res.writeHead(416).end();return;}
          res.writeHead(206,{'Content-Type':'application/octet-stream','Content-Length':end-start+1,'Content-Range':`bytes ${start}-${end}/${before.size}`,ETag:etag,'Cache-Control':'no-store'});
          const stream=fs.createReadStream(weightsPath,{fd,autoClose:false,start,end});
          stream.on('error',error=>res.destroy(error));res.on('close',()=>stream.destroy());stream.pipe(res);
        }catch(error){res.writeHead(409).end(error.message);}
      },close(){fs.closeSync(fd);}};
  }catch(error){fs.closeSync(fd);throw error;}
}
