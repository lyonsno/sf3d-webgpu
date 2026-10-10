import assert from 'node:assert/strict';
import {TwoStreamBackbone,createTwoStreamAttentionDutyPlan} from '../src/lib/two_stream.js';
import {createEmptyBuffer} from '../src/lib/gpu.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
// Deterministic row-independent dispatch emulation exercises exact copies,
// offsets, partial tails and complete outputs; it is not native GPU parity.
function run(rowsPerTile,fine=false){
  const device=fakeWeightDevice(),backbone=new TwoStreamBackbone(device),N=7,D=2;
  const input=createEmptyBuffer(device,N*D*4),values=new Float32Array(input.data);
  values.set(Array.from({length:N*D},(_,i)=>Math.fround(((i*7)%11-5)/13)));
  const encoder={copyBufferToBuffer(src,offset,dst,dstOffset,size){
    new Uint8Array(dst.data,dstOffset,size).set(new Uint8Array(src.data,offset,size));
  }};
  const calls=[];
  backbone._dispatchLinear=(enc,src,dst,weight,bias,rows,inDim,outDim)=>{
    calls.push({rows,inDim,outDim});
    const a=new Float32Array(src.data),b=new Float32Array(dst.data);
    for(let row=0;row<rows;row++)for(let col=0;col<outDim;col++){
      let sum=Math.fround((col%3)/7);
      for(let k=0;k<inDim;k++)sum=Math.fround(sum+Math.fround(a[row*inDim+k]*Math.fround(((col+1)*(k+2)%17)/100)));
      b[row*outDim+col]=sum;
    }
  };
  backbone._dispatchGEGLUActivation=(enc,src,dst,rows,inner)=>{
    const a=new Float32Array(src.data),b=new Float32Array(dst.data);
    for(let row=0;row<rows;row++)for(let col=0;col<inner;col++){
      const x=a[row*2*inner+col],gate=a[row*2*inner+inner+col];
      b[row*inner+col]=Math.fround(x*0.5*gate*(1+Math.tanh(Math.sqrt(2/Math.PI)*(gate+0.044715*gate**3))));
    }
  };
  const ff={geglu:{weight:{},bias:{}},proj:{weight:{},bias:{}}};
  let result;
  if(fine){
    const operation={kind:'fuse',ownerId:'block-0-fuse-out',N_z:N,D,z2NormBuf:input,weights:{ff}};
    const state={residentFFN:true,ffnRowsPerTile:rowsPerTile,activeOperation:operation};
    for(let rowStart=0,rangeIndex=0;rowStart<N;rowStart+=rowsPerTile,rangeIndex++){
      const rowCount=Math.min(rowsPerTile,N-rowStart);
      backbone._dispatchFineFuseResidentFFNRange(encoder,state,{
        block:0,ownerId:'block-0-fuse-out-ffn-projection',rangeIndex,
        rangeCount:Math.ceil(N/rowsPerTile),totalRows:N,rowStart,rowCount,rowEnd:rowStart+rowCount,
      });
    }
    assert.equal(operation.ffnProjection.nextRowStart,N);
    result=operation.ffnProjection.output;
  }else result=backbone._dispatchGEGLUFFN(encoder,input,ff,N,D,{rowsPerTile});
  return {bytes:new Uint8Array(result.data),buffers:device.buffers,calls,N,D};
}
const legacy=run(null),resident=run(2);
assert.deepEqual(resident.bytes,legacy.bytes,'every row and final partial tail preserve exact deterministic computation');
assert.ok(resident.buffers.every(b=>b.size<=2*8192*4),'resident FFN must not materialize the complete GEGLU expansion');
assert.equal(resident.buffers.length,6,'input, complete output and one reusable four-buffer scratch tile');
assert.deepEqual(resident.calls.filter(c=>c.outDim===8192).map(c=>c.rows),[2,2,2,1],'all7 rows are computed, not capped');
for(const tile of [1,3,7,20])assert.deepEqual(run(tile).bytes,legacy.bytes);
for(const tile of [1,2,3,7,20])assert.deepEqual(run(tile,true).bytes,legacy.bytes,'fine duties use complete real row kernels and placements');
const invalid=new TwoStreamBackbone(fakeWeightDevice());
for(const rowsPerTile of [0,-1,1.5,NaN])assert.throws(()=>invalid._dispatchGEGLUFFN({}, {}, {}, 7,2,{rowsPerTile}),/rowsPerTile/);
assert.equal(invalid.device.buffers.length,0,'malformed tile policy refuses before allocation');
const fullPlan=createTwoStreamAttentionDutyPlan(1297,{residentFFN:true,linearRowsPerDuty:128});
assert.equal(fullPlan.length,3350,'resident plan declares actual work, not empty legacy expansion/activation duties');
assert.equal(fullPlan.filter(d=>d.kind==='fuse-geglu-linear-range'||d.kind==='fuse-geglu-activate'||d.kind==='fuse-ffn-linear-range').length,0);
for(let block=0;block<4;block++){
  const rows=fullPlan.filter(d=>d.block===block&&d.kind==='fuse-resident-ffn-range');
  assert.equal(rows.length,216);
  assert.equal(rows.reduce((n,d)=>n+d.rowCount,0),27648);
  assert.equal(rows[0].rowStart,0);assert.equal(rows.at(-1).rowEnd,27648);
}
assert.equal(createTwoStreamAttentionDutyPlan(1297).length,4218,'default route remains unchanged');
assert.throws(()=>createTwoStreamAttentionDutyPlan(1297,{residentFFN:'yes'}),/residentFFN/);
console.log('row-resident FFN preserves complete deterministic rows with reusable bounded scratch');
