import assert from 'node:assert/strict';
const subject=await import('./resident_two_stream_acceptance.mjs').catch(error=>{
  if(error.code==='ERR_MODULE_NOT_FOUND'&&error.url===new URL('./resident_two_stream_acceptance.mjs',import.meta.url).href)return {};
  throw error;
});
assert.equal(typeof subject.twoStreamPhaseDemand,'function','complete backbone must identify its actual new allocation cliffs');
assert.equal(subject.acceptResidentTwoStream({}).ok,false,'early missing evidence must produce a refusal rather than throw while writing terminal report');
const tri=27648*1024*4,latent=3089*1024*4;
assert.equal(subject.twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-resident-ffn-range',rangeIndex:0,rowCount:128}}).workGpuBytes,
  tri+128*(1024*2+4096*3)*4+4096,'complete FFN output plus reusable four-buffer scratch, not full GEGLU expansion');
assert.equal(subject.twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-resident-ffn-range',rangeIndex:1}}).workGpuBytes,0);
assert.ok(subject.twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-prepare',direction:'in'}}).workGpuBytes>4*tri);
assert.equal(subject.twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'final'}}).workGpuBytes,2*tri+4096);
assert.throws(()=>subject.twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-geglu-linear-range'}}),/resident/);
assert.throws(()=>subject.twoStreamPhaseDemand({name:'two-stream-final',tensors:[{size:-1,dtype:0}]}),/tensor/);
assert.throws(()=>subject.inspectTwoStreamOutputBytes(Buffer.alloc(4)),/complete/);
assert.equal(subject.twoStreamPhaseDemand({name:'two-stream-duty',duty:{kind:'fuse-prepare',direction:'in'},
  attentionRowsPerDuty:32,normX:false}).workGpuBytes,
  (3*3089+2*27648)*1024*4+16*32*27648*4+32*1024*4+1024*4+4096,
  'actual smaller query score scratch and canonical absent normalization must identify only new backing');
console.log('complete backbone cliff demand and incomplete output refusal');
