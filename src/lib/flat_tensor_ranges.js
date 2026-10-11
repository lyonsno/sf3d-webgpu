// Flat-v1 source adapter. Kit 0.1.52 chunks upload unchanged bytes and its
// verified custody does not expose a conversion hook. Keep SF3D's existing
// mixed-F16/F32 conversion; do not mislabel packed F16 as an F32 kit bundle.
// Explicit reservations are not a bound on network, GC or driver backing.
const ENTRY_SIZE=160, MAGIC=0x33445346;

export function parseFlatWeightHeader(buffer, totalBytes=Infinity) {
  const view=new DataView(buffer);
  if(buffer.byteLength<16 || view.getUint32(0,true)!==MAGIC)throw Error('Invalid weight file magic/header');
  if(view.getUint32(4,true)!==1)throw Error('Unsupported weight file version');
  const count=view.getUint32(8,true),headerSize=view.getUint32(12,true);
  if(headerSize!==16+count*ENTRY_SIZE || headerSize!==buffer.byteLength || headerSize>totalBytes)
    throw Error('weight header exceeds or contradicts received bytes');
  const tensors=new Map(), intervals=[];
  for(let i=0;i<count;i++){
    const off=16+i*ENTRY_SIZE,bytes=new Uint8Array(buffer,off,128),end=bytes.indexOf(0);
    const name=new TextDecoder('utf-8',{fatal:true}).decode(bytes.subarray(0,end<0?128:end));
    const dtype=view.getUint32(off+128,true),rank=view.getUint32(off+132,true);
    // The canonical converter records the full rank but stores four axes only.
    // Never read the offset/size fields as axes or invent the omitted axes.
    const shape=Array.from({length:Math.min(rank,4)},(_,d)=>view.getUint32(off+136+4*d,true));
    const offset=headerSize+view.getUint32(off+152,true),size=view.getUint32(off+156,true),width=dtype===0?4:2;
    if(!name || tensors.has(name) || (dtype!==0 && dtype!==1) || size===0 || size%width || offset%width || offset+size>totalBytes)
      throw Error('invalid/duplicate/out-of-bounds flat tensor: '+name);
    if(rank<=4 && shape.reduce((n,d)=>n*d,1)*width!==size)throw Error('tensor shape/size conflict: '+name);
    tensors.set(name,{dtype,shape,declaredRank:rank,shapeComplete:rank<=4,offset,size});
    intervals.push({name,offset,size});
  }
  intervals.sort((a,b)=>a.offset-b.offset);
  for(let i=1;i<intervals.length;i++)if(intervals[i].offset<intervals[i-1].offset+intervals[i-1].size)
    throw Error('overlapping flat tensors: '+intervals[i].name);
  return {tensors,headerSize};
}

export async function createFlatTensorRangeSource(url,{memoryBudget,expectedWeightBytes,expectedSourceETag,onProgress,onBeforeSourceIntake}={}) {
  if(!Number.isSafeInteger(expectedWeightBytes) || expectedWeightBytes<16)throw TypeError('explicit expectedWeightBytes required');
  if(typeof expectedSourceETag!=='string' || !/^"[^"\r\n]+"$/.test(expectedSourceETag))
    throw TypeError('explicit strong expectedSourceETag required');
  if(onBeforeSourceIntake!==undefined&&typeof onBeforeSourceIntake!=='function')
    throw TypeError('source intake admission callback must be a function');
  const report={mode:'tensor-ranges',sourceUrl:String(url),sourceETag:expectedSourceETag,expectedWeightBytes,
    sourceIdentityAuthority:'HTTP strong ETag/If-Match plus exact ranges; not an independently verified whole-artifact digest',
    maximumSourceUnitBytes:0,receivedBytes:0,ranges:[],physicalMemoryMeasured:false};
  const read=async (offset,size,label)=>{
    if(label==='weight-header-prefix'||label==='weight-header')await onBeforeSourceIntake?.({
      name:label,tensors:[],storageKind:'cpu-source',sourceOffset:offset,sourceUnitBytes:size,
      rangeCpuBytes:2*size,workGpuBytes:0,requiredBytes:2*size,
      authority:'identified destination plus response unit; not network/GC physical-fit authority'});
    memoryBudget.setPhase(label);
    // Own destination and allow an entire requested response unit concurrently.
    // Reserve both BEFORE fetch; overlong server data is rejected, never appended.
    let lease,receiveLease,reader;
    try{
      lease=memoryBudget.reserveCpu(size,label+':destination');
      receiveLease=memoryBudget.reserveCpu(size,label+':response');
      const bytes=new Uint8Array(size),end=offset+size-1;
      const response=await fetch(url,{headers:{Range:`bytes=${offset}-${end}`,'If-Match':expectedSourceETag},cache:'no-store'});
      reader=response.body?.getReader();
      if(response.status!==206 || response.headers.get('etag')!==expectedSourceETag ||
        response.headers.get('content-range')!==`bytes ${offset}-${end}/${expectedWeightBytes}` ||
        (response.headers.has('content-length') && Number(response.headers.get('content-length'))!==size) || !reader)
        throw Error('weight range response/source identity mismatch (206, exact range/length and strong ETag required)');
      let received=0;
      while(true){
        const {done,value}=await reader.read();if(done)break;
        if(!(value instanceof Uint8Array) || value.length>size-received)throw Error('weight range bytes exceed requested unit');
        bytes.set(value,received);received+=value.length;
      }
      if(received!==size)throw Error('weight range bytes truncated');
      report.maximumSourceUnitBytes=Math.max(report.maximumSourceUnitBytes,size);
      report.receivedBytes+=received;report.ranges.push({label,offset,size});
      onProgress?.(report.receivedBytes,expectedWeightBytes);
      const destinationLease=lease;
      const result={bytes,release:()=>destinationLease.release()};lease=null;
      return result;
    }finally{
      lease?.release();receiveLease?.release();
      try{await reader?.cancel();}catch{}try{reader?.releaseLock();}catch{}
    }
  };
  const prefix=await read(0,16,'weight-header-prefix');
  let headerSize;
  try{
    const view=new DataView(prefix.bytes.buffer),count=view.getUint32(8,true);
    headerSize=view.getUint32(12,true);
    if(view.getUint32(0,true)!==MAGIC || view.getUint32(4,true)!==1 || headerSize!==16+count*ENTRY_SIZE || headerSize>expectedWeightBytes)
      throw Error('invalid flat weight header prefix');
  }finally{prefix.release();prefix.bytes=null;}
  const header=await read(0,headerSize,'weight-header');
  let tensors;
  try{({tensors}=parseFlatWeightHeader(header.bytes.buffer,expectedWeightBytes));}
  finally{header.release();header.bytes=null;}
  report.headerBytes=headerSize;report.tensorCount=tensors.size;
  report.incompleteShapeNames=[...tensors].filter(([,info])=>!info.shapeComplete).map(([name])=>name);
  return {tensors,report,readTensor:async name=>{
    const info=tensors.get(name);if(!info)throw Error('Missing weight: '+name);
    return read(info.offset,info.size,'weight-tensor:'+name);
  }};
}
