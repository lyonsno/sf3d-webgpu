import assert from 'node:assert/strict';
import {SF3DImageTokenizer} from '../src/lib/sf3d_backbone.js';
import {createEmptyBuffer} from '../src/lib/gpu.js';
import {fakeWeightDevice} from './fixtures/weight_resource_fixture.mjs';

const tokenizer=Object.create(SF3DImageTokenizer.prototype),events=[],output={tokensBuf:{},N:1297};
tokenizer._setupEncode=(encoder,image,camera,weights)=>{assert.equal(weights.patchEmbed,'patch');events.push('setup');return {};};
tokenizer._encodeBlock=(encoder,index,ctx,weights)=>{assert.equal(weights.blocks[index],`block-${index}`);events.push(`encode-${index}`);};
tokenizer._finalizeEncode=(encoder,ctx,weights)=>{assert.equal(weights.layernorm,'final');events.push('final');return output;};
const result=await tokenizer.encodeCooperative({imageBuf:{},cameraEmbedBuf:{},weights:{},numBlocks:24,chunkBlocks:1,
  async withChunkWeights({blockStart,blockEnd,isFirst,isLast},work){
    events.push(`load-${blockStart}`);
    const blocks=Array(24).fill(null);for(let i=blockStart;i<blockEnd;i++)blocks[i]=`block-${i}`;
    try{return await work({blocks,...(isFirst?{patchEmbed:'patch'}:{}),...(isLast?{layernorm:'final'}:{})});}
    finally{events.push(`retire-${blockStart}`);}
  },
  async driver(start,end,encode){encode({});events.push(`fence-${start}`);},
});
assert.equal(result,output);
assert.equal(events.filter(e=>e.startsWith('encode-')).length,24);
for(let i=0;i<24;i++){
  assert.ok(events.indexOf(`load-${i}`)<events.indexOf(`encode-${i}`));
  assert.ok(events.indexOf(`fence-${i}`)<events.indexOf(`retire-${i}`));
  if(i<23)assert.ok(events.indexOf(`retire-${i}`)<events.indexOf(`load-${i+1}`));
}
assert.equal(events.filter(e=>e==='setup').length,1);assert.equal(events.filter(e=>e==='final').length,1);
await assert.rejects(()=>tokenizer.encodeCooperative({numBlocks:24,chunkBlocks:1,weights:{},withChunkWeights:42,driver:async()=>{}}),/withChunkWeights/);
globalThis.GPUBufferUsage={STORAGE:128,COPY_SRC:4,COPY_DST:8};
const retiring=Object.create(SF3DImageTokenizer.prototype),device=fakeWeightDevice();retiring.device=device;
retiring._setupEncode=()=>({temporary:createEmptyBuffer(device,16),output:createEmptyBuffer(device,16)});
retiring._encodeBlock=()=>{};retiring._finalizeEncode=(encoder,ctx)=>({tokensBuf:ctx.output,N:1297});
const retained=await retiring.encodeCooperative({imageBuf:{},cameraEmbedBuf:{},weights:{},numBlocks:24,chunkBlocks:1,
  retireIntermediateBuffers:true,driver:async(start,end,encode)=>{encode({});}});
assert.equal(device.buffers[0].destroyed,1,'completed DINO work buffers retire rather than accumulating into the next model phase');
assert.equal(retained.tokensBuf.destroyed,0,'the complete encoder output survives for the backbone');
console.log('actual DINO cooperative loop loads/fences/retires all 24 blocks without changing setup/final order');
