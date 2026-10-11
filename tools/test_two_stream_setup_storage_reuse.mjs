import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {twoStreamPhaseDemand} from './resident_two_stream_acceptance.mjs';

// Exercise the actual setup body with tiny deterministic complete CPU operands.
// This is an ordering/ownership witness, not native shader conformance.
const source=fs.readFileSync(new URL('../src/lib/two_stream.js',import.meta.url),'utf8');
const body=source.slice(source.indexOf('  _dispatchForwardSetup(encoder, state) {'),
  source.indexOf('\n  _dispatchForwardBlockStage(',source.indexOf('  _dispatchForwardSetup(encoder, state) {')));
const CONFIG={dim:4,triplaneTokens:6,numLatents:2};
const usage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
function fixture(reuse){
  const allocations=[],operations=[];
  const make=size=>{const b={size,usage:140,data:new Float32Array(size/4)};allocations.push(b);return b;};
  const setup=vm.runInNewContext('('+body.replace('_dispatchForwardSetup(encoder, state)','function(encoder, state)')+')',
    {CONFIG,GPUBufferUsage:usage,createEmptyBuffer:(_device,size)=>make(size)});
  const embedding=make(96),image=make(32),latent=make(32);
  for(const [j,b]of [embedding,image,latent].entries())for(let i=0;i<b.data.length;i++)b.data[i]=Math.fround((i-j*3-7)/11);
  const weights={tokenizer_embeddings_buf:embedding,latentInit:latent,normTriplane:{},
    projTriplane:{weight:1.25,bias:-.3},normImage:{},projImage:{weight:-.7,bias:.2},
    normLatent:{},projLatent:{weight:.9,bias:.1}};
  const backbone={device:{},_residentAdapterOwner:{allocations:[{buffer:embedding,size:96}]},
    _residentWorkOwner:{allocations:[],buffers:new Set()},_diagnosticBuffers:{},
    _dispatchGroupNorm(enc,input,norm,D,N,groups){operations.push(['GN',D,N,groups]);
      const output=make(input.size);for(let i=0;i<input.data.length;i++)output.data[i]=Math.fround(input.data[i]*.5+.125);return output;},
    _dispatchTranspose(enc,input,output,rows,columns){assert.notEqual(input,output);
      operations.push(['transpose',rows,columns]);for(let r=0;r<rows;r++)for(let c=0;c<columns;c++)output.data[c*rows+r]=input.data[r*columns+c];},
    _dispatchLinear(enc,input,output,w,b,rows,D,out){assert.notEqual(input,output);
      operations.push(['linear',rows,D,out]);for(let i=0;i<input.data.length;i++)output.data[i]=Math.fround(input.data[i]*w+b);},
    _dispatchLayerNorm(enc,input,output,norm,rows,D){assert.notEqual(input,output);
      operations.push(['LN',rows,D]);for(let i=0;i<input.data.length;i++)output.data[i]=Math.fround(input.data[i]-.25);},
    _dispatchConcat(enc,a,b,output,na,nb){operations.push(['concat',na,nb]);output.data.set(a.data);output.data.set(b.data,na);}};
  const state={imageTokensBuf:image,N_img:2,N_latent:4,weights,residentFFN:true,
    reuseTokenizerSetupStorage:reuse,rematerializeTokenizerEmbedding:true,nextFineDutyIndex:0};
  return {setup,backbone,state,allocations,operations,embedding,image,latent};
}
const optimized=fixture(true),ordinary=fixture(false);
optimized.setup.call(optimized.backbone,{},optimized.state);
ordinary.setup.call(ordinary.backbone,{},ordinary.state);
assert.equal(optimized.allocations.length,ordinary.allocations.length-2,
  'setup must allocate two fewer complete triplane backings');
assert.deepEqual(optimized.state.currentTriplane.data,ordinary.state.currentTriplane.data);
assert.deepEqual(optimized.state.currentLatent.data,ordinary.state.currentLatent.data);
assert.deepEqual(optimized.operations,ordinary.operations,'every original operation retains full dimensions and order');
assert.deepEqual(optimized.operations[0],['GN',4,6,32]);
assert.equal(optimized.backbone._diagnosticBuffers.gnOutBuf,undefined,'overwritten GN cannot masquerade as preserved diagnostic output');
assert.equal(optimized.backbone._diagnosticBuffers.triPermBuf,undefined,'adapter-owned retired transpose cannot masquerade as retained output');
assert.equal(optimized.state.tokenizerSetupStorageReuse.reusedTriplaneBuffers,2);
assert.equal(optimized.state.tokenizerSetupStorageReuse.bytesPerBuffer,96);
for(const mutate of [x=>x.state.rematerializeTokenizerEmbedding=false,x=>x.state.residentFFN=false,
  x=>x.state.nextFineDutyIndex=1,x=>x.backbone._residentAdapterOwner=null,
  x=>x.backbone._residentAdapterOwner.allocations[0].buffer={},x=>x.backbone._residentAdapterOwner.allocations[0].size=92,
  x=>x.embedding.size=92,x=>x.embedding.usage=4,x=>x.backbone._residentWorkOwner=null,
  x=>x.state.reuseTokenizerSetupStorage='yes']){
  const x=fixture(true);mutate(x);const count=x.allocations.length,bytes=x.embedding.data.slice();
  assert.throws(()=>x.setup.call(x.backbone,{},x.state),/owned|setup|reuse|rematerial/i);
  assert.equal(x.allocations.length,count,'invalid custody refuses before encoding or new backing');
  assert.equal(x.operations.length,0);assert.deepEqual(x.embedding.data,bytes);
}
const demand={name:'two-stream-duty',duty:{kind:'setup'},reuseTokenizerSetupStorage:true,rematerializeTokenizerEmbedding:true};
assert.equal(twoStreamPhaseDemand(demand).workGpuBytes,151335168);
assert.equal(twoStreamPhaseDemand({...demand,reuseTokenizerSetupStorage:false}).workGpuBytes,377827584);
for(const mutate of [x=>x.rematerializeTokenizerEmbedding=false,x=>delete x.rematerializeTokenizerEmbedding,
  x=>x.reuseTokenizerSetupStorage='yes']){
  const x={...demand};mutate(x);assert.throws(()=>twoStreamPhaseDemand(x),/setup|reuse|rematerial/i);
}
console.log('PASS actual setup complete ordered operands, exact owned sequential backing reuse, default unchanged, invalid custody fail-before-allocation, truthful new demand; local schedule only');
