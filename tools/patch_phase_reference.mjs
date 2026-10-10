import fs from 'node:fs';
import {createHash} from 'node:crypto';
export const PATCH_TENSOR_NAMES = Object.freeze([
  'image_tokenizer.model.embeddings.patch_embeddings.projection.weight',
  'image_tokenizer.model.embeddings.patch_embeddings.projection.bias',
  'image_tokenizer.model.embeddings.cls_token',
  'image_tokenizer.model.embeddings.position_embeddings',
]);

// Count all explicit source/conversion/transport stores even after logical
// retirement. This plan does not equate destroy()/GC eligibility with physical
// reclamation. The post-launch host/process baseline covers existing backing;
// new opaque browser/driver growth remains observed, not certified here.
export function patchPhaseDemand(source,{width,height,byteLength=0}){
  const cpu=[
    {name:'encoded-image-response-blob',bytes:2*byteLength},
    {name:'range-prefix-and-header-destination-response',bytes:32+2*source.headerBytes},
    ...source.units.map(u=>({name:'cumulative-source-response-conversion:'+u.name,bytes:2*u.size+(u.dtype===1?u.expandedBytes:0)})),
    {name:'decoded-image-canvas-and-rgba-readback',bytes:3*width*height*4},
    {name:'source-rgba-f32',bytes:width*height*16},
    {name:'lanczos-horizontal-f32',bytes:512*height*16},
    {name:'lanczos-resized-f32',bytes:512*512*16},
    {name:'normalized-image-and-transport-copy',bytes:2*3145728},
    {name:'output-mapped-copy-and-transport-copy',bytes:2*5312512},
  ];
  const gpu=[...source.units.map(u=>({name:'resident-weight:'+u.name,bytes:u.expandedBytes})),
    {name:'image-input',bytes:3145728},{name:'learned-output',bytes:5312512},
    {name:'readback-staging',bytes:5312512},{name:'dispatch-uniform',bytes:36}];
  const total=rows=>rows.reduce((sum,r)=>sum+r.bytes,0);
  return{cpu,gpu,cpuBytes:total(cpu),gpuBytes:total(gpu),requiredBytes:total(cpu)+total(gpu),
    meaning:'explicit selected-phase backing bound plus separately observed browser/process baseline; not full-model or opaque-memory certification'};
}

function decodeHalf(h){
  const sign=h&0x8000?-1:1,e=(h>>>10)&31,m=h&1023;
  return e===31?(m?NaN:sign*Infinity):e===0?sign*m*2**-24:sign*(1+m/1024)*2**(e-15);
}
export function checkPatchOutput({source,inputPath,outputPath}){
  const inputBytes=fs.readFileSync(inputPath),outputBytes=fs.readFileSync(outputPath);
  if(inputBytes.length!==3145728||outputBytes.length!==5312512)throw Error('complete production-shape input/output required');
  const floats=b=>new Float32Array(b.buffer,b.byteOffset,b.byteLength/4);
  const input=floats(inputBytes),output=floats(outputBytes),fd=fs.openSync(source.path,'r');
  const values=new Map();
  try{for(const u of source.units){
    const raw=Buffer.alloc(u.size);if(fs.readSync(fd,raw,0,u.size,u.offset)!==u.size)throw Error('reference source truncated');
    if(createHash('sha256').update(raw).digest('hex')!==u.sha256)throw Error('reference source changed');
    values.set(u.name,u.dtype===1?Array.from({length:raw.length/2},(_,i)=>decodeHalf(raw.readUInt16LE(2*i))):Array.from(floats(raw)));
  }}finally{fs.closeSync(fd);}
  const [weight,bias,cls,pos]=PATCH_TENSOR_NAMES.map(n=>values.get(n));
  if(!weight||weight.length!==1024*588||bias?.length!==1024||cls?.length!==1024||pos?.length!==1297*1024)
    throw Error('actual canonical patch weight shapes required');
  const rows=[];
  // Two CLS coordinates plus 64 distributed output coordinates; preserve all
  // input/output bytes so an all-coordinate competing reference is replayable.
  for(let i=0;i<66;i++){
    const token=i<2?0:1+((i-2)*997)%1296,d=(i*467)%1024,index=token*1024+d;
    let expected=token===0?cls[d]:bias[d],absoluteSum=Math.abs(expected);
    if(token){const patch=token-1,y=Math.floor(patch/36)*14,x=(patch%36)*14;
      for(let c=0;c<3;c++)for(let py=0;py<14;py++)for(let px=0;px<14;px++){
        const product=input[c*512*512+(y+py)*512+x+px]*weight[d*588+c*196+py*14+px];
        expected+=product;absoluteSum+=Math.abs(product);
      }
    }
    expected+=pos[index];absoluteSum+=Math.abs(pos[index]);
    // Standard forward-error envelope for 590 float32 operations, including
    // legal fused/non-fused multiply-add. This is arithmetic-derived, not a
    // tolerance tuned to hide an observed failure.
    const tolerance=(590*2**-24/(1-590*2**-24))*absoluteSum+2**-24*Math.abs(expected);
    const actual=output[index],error=Math.abs(actual-expected);
    rows.push({token,d,index,expected,actual,error,tolerance,pass:Number.isFinite(actual)&&error<=tolerance});
  }
  let finiteCount=0,nonzeroCount=0;for(const value of output){if(Number.isFinite(value))finiteCount++;if(value!==0)nonzeroCount++;}
  const hash=b=>createHash('sha256').update(b).digest('hex');
  return{inputBytes:inputBytes.length,outputBytes:outputBytes.length,inputSha256:hash(inputBytes),outputSha256:hash(outputBytes),
    finiteCount,nonzeroCount,reference:{tested:rows.length,mismatches:rows.filter(r=>!r.pass).length,rows,meaning:'sampled independent scalar arithmetic; complete raw tensors retained for replay'}};
}
