import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {weightFixture} from './fixtures/weight_resource_fixture.mjs';
import {parseFlatWeightHeader} from '../src/lib/flat_tensor_ranges.js';
import {loadDinoConversionHashes} from './native_dino_observation.mjs';
const fixture=weightFixture(),header=new DataView(fixture.bytes.buffer).getUint32(12,true);
const table=parseFlatWeightHeader(fixture.bytes.buffer.slice(0,header),fixture.bytes.length).tensors;
const canonical={sha256:'a'.repeat(64),headerSha256:'b'.repeat(64),byteLength:fixture.bytes.length};
const rows=[];let rawBytes=0,expandedBytes=0;
for(const phase of ['camera',...Array.from({length:24},(_,i)=>'dino-block-'+i)]){
  const b=Number(phase.slice('dino-block-'.length));
  const selected=[...table].filter(([name])=>phase==='camera'?name.startsWith('camera_embedder.linear.'):
    name.startsWith('image_tokenizer.model.encoder.layer.'+b+'.')||(b===0&&name.startsWith('image_tokenizer.model.embeddings.'))||
    (b===23&&name.startsWith('image_tokenizer.model.layernorm.')));
  for(const [name,t]of selected){
    const bytes=t.size*(t.dtype===1?2:1),digest=createHash('sha256').update(Buffer.alloc(bytes)).digest('hex');
    rawBytes+=t.size;expandedBytes+=bytes;
    for(const mode of ['mapped','streamed'])rows.push({mode,phase,name,rawSHA256:digest,convertedSHA256:digest,words:bytes/4,bytes});
  }
}
const replay={schema:'sf3d.canonical-dino-conversion-replay.v0',status:'passed',phase:'complete',
  source:{wholeSHA256FromNative:canonical.sha256,headerSHA256:canonical.headerSha256,byteLength:canonical.byteLength},
  scope:{phases:25,tensors:536,rawBytes,expandedBytes},rows};
assert.equal(loadDinoConversionHashes(replay,canonical,table).size,536);
for(const [label,mutate] of [
 ['words',r=>r.rows[0].words=String(r.rows[0].words)],
 ['phase',r=>r.rows.find(x=>x.phase==='dino-block-0').phase='xxxxxxxxxxx0'],
]){
 if(process.argv.includes('--phase-only')&&label!=='phase')continue;
 const damaged=JSON.parse(JSON.stringify(replay));mutate(damaged);
 assert.throws(()=>loadDinoConversionHashes(damaged,canonical,table),/malformed|canonical phase/,
   'retyped words and noncanonical phase identity must refuse: '+label);
}
for(const mutate of [r=>r.status='running',r=>r.source.headerSHA256='c'.repeat(64),r=>r.rows.pop(),
  r=>r.rows.push(r.rows[0]),r=>r.scope.tensors='536',r=>r.rows[1].mode='fallback',r=>r.rows[0].bytes=1,
  r=>r.rows[0].convertedSHA256='d'.repeat(64),r=>r.scope.rawBytes--,r=>r.rows[1].rawSHA256='e'.repeat(64)]){
  const damaged=JSON.parse(JSON.stringify(replay));mutate(damaged);assert.throws(()=>loadDinoConversionHashes(damaged,canonical,table));
}
const added=JSON.parse(JSON.stringify(replay));added.unrelated='compatible';added.rows[0].future='compatible';
assert.equal(loadDinoConversionHashes(added,canonical,table).size,536);
console.log('PASS complete536/1072 conversion source/mode/shape/hash/totals reference validation and additive compatibility; synthetic policy fixture');
