import assert from 'node:assert/strict';
import {TwoStreamBackbone} from '../src/lib/two_stream.js';
import {twoStreamPhaseDemand} from './resident_two_stream_acceptance.mjs';
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
const D=1024,rows=32;
const cases=[
  [{kind:'fuse-prepare',direction:'in'},3089,27648],
  [{kind:'fuse-prepare',direction:'out'},27648,3089],
  [{kind:'basic-self-prepare'},3089,3089],
  [{kind:'basic-cross-prepare'},3089,1297],
];
// Exercise actual constructor descriptors without allocating canonical CPU
// arrays in a fake backend. Local source-allocation witness, not Metal proof.
const allocated=(q,kv,reuse)=>{
  const sizes=[],device={createBuffer({size}){sizes.push(size);return {size,destroy(){}};}};
  const b=new TwoStreamBackbone(device);b._dispatchLinearNoBias=()=>{};
  const state=b._createAttentionState({}, {}, {}, {},q,kv,D,rows,reuse);
  assert.equal(state.attnOutBuf===state.qBuf,reuse);
  return sizes.reduce((n,size)=>n+size,0);
};
for(const [duty,q,kv] of cases){
  const base={name:'two-stream-duty',duty,attentionRowsPerDuty:rows,normX:false};
  const old=twoStreamPhaseDemand(base),next=twoStreamPhaseDemand({...base,reuseAttentionQueryStorage:true});
  const omitted=q*D*4;
  assert.equal(allocated(q,kv,false)-allocated(q,kv,true),omitted);
  assert.equal(next.workGpuBytes,old.workGpuBytes-omitted,'NEW must exclude only the separate output allocation actually removed');
  assert.equal(next.requiredBytes,old.requiredBytes-omitted);
  assert.equal(next.weightGpuBytes,old.weightGpuBytes);assert.equal(next.rangeCpuBytes,old.rangeCpuBytes);
  assert.deepEqual(twoStreamPhaseDemand({...base,reuseAttentionQueryStorage:false}),old,'legacy declarations unchanged');
}
assert.equal(twoStreamPhaseDemand({name:'two-stream-duty',duty:cases[0][0],attentionRowsPerDuty:32,normX:false,
  reuseAttentionQueryStorage:true}).requiredBytes,308559872);
for(const value of [null,1,'true',{},NaN])assert.throws(()=>twoStreamPhaseDemand({
  name:'two-stream-duty',duty:cases[0][0],reuseAttentionQueryStorage:value}),/query.*reuse|boolean/i);
for(const duty of [{kind:'attention-tile'},{kind:'basic-finish'},{kind:'final'}]){
  const p={name:'two-stream-duty',duty};assert.deepEqual(twoStreamPhaseDemand({...p,reuseAttentionQueryStorage:true}),twoStreamPhaseDemand(p));
}
console.log('PASS exact prospective query-output NEW matches actual constructor allocation delta; CPU/source/weights/default/other duties unchanged; no physical subtraction');
