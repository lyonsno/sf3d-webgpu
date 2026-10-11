import assert from 'node:assert/strict';
import * as mat from '../src/lib/materialize_core.js';
import * as baker from '../src/lib/texture_baker.js';
assert.equal(typeof mat.createDilationScratch,'function','full texture tail needs counted typed dilation backing');
assert.equal(typeof baker.unwrapResidentUV,'function','resident UV must avoid retained quadratic duplicate intersection pairs');
const tetraVertices=new Float32Array([0,0,0,1,0,0,0,1,0,0,0,1]),tetraFaces=new Uint32Array([0,1,2,0,2,3,0,3,1,1,3,2]);
const ordinary=baker.unwrapUV(tetraVertices,tetraFaces,4,4),resident=baker.unwrapResidentUV(tetraVertices,tetraFaces,4,4);
for(const field of ['uvs','newVertices','newNormals','newFaces','faceAssignment'])assert.deepEqual(resident[field],ordinary[field]);
const size=5,pixels=size*size;
for(let pattern=0;pattern<32;pattern++){
  const mask=Uint8Array.from({length:pixels},(_,i)=>(pattern>>(i%5))&1);
  const original=Uint8Array.from({length:pixels*4},(_,i)=>(i*47)%256),a=original.slice(),b=original.slice();
  mat.dilateTexture(a,mask,size,4);
  const scratch=mat.createDilationScratch(pixels);
  assert.equal(Object.values(scratch).reduce((n,x)=>n+x.byteLength,0),pixels*8);
  mat.dilateTextureCounted(b,mask,size,4,scratch);
  assert.deepEqual(b,a,'complete snapshot-neighbor order and rounding match actual ordinary dilation');
}
let encodes=0;
const oldDocument=globalThis.document;
globalThis.document={createElement(){return {getContext(){return {createImageData(w,h){return {data:new Uint8ClampedArray(w*h*4)};},putImageData(){}};},toBlob(done){encodes++;done(new Blob([new Uint8Array([255,216,255,217])],{type:'image/jpeg'}));}};}};
try{
  const args=[new Float32Array([0,0,0,1,0,0,0,1,0]),new Float32Array([0,0,1,0,0,1,0,0,1]),new Uint32Array([0,1,2]),new Float32Array([0,0,1,0,0,1]),new Uint8Array(16),new Uint8Array(16),3,1,2,0.7,0.2];
  await assert.rejects(()=>baker.exportGLB(...args,{onBeforeCpuAllocation:async()=>{throw 0;},requireNormalTexture:true}),e=>e===0);
  assert.equal(encodes,0,'denied first transform allocation must precede every image encode');
  const phases=[];
  const glb=await baker.exportGLB(...args,{onBeforeCpuAllocation:async(name,bytes)=>{assert.ok(Number.isSafeInteger(bytes)&&bytes>0);phases.push([name,bytes]);},requireNormalTexture:true});
  assert.deepEqual(phases.map(p=>p[0]),['glb-transforms','glb-albedo-image-data','glb-albedo-encoded','glb-normal-image-data','glb-normal-encoded','glb-json','glb-complete-buffer']);
  assert.equal(phases[0][1],84);assert.equal(phases.at(-1)[1],glb.byteLength);
  const view=new DataView(glb);assert.equal(view.getUint32(8,true),glb.byteLength);
  const json=JSON.parse(new TextDecoder().decode(new Uint8Array(glb,20,view.getUint32(12,true))).trim());
  assert.equal(json.images.length,2);assert.equal(json.materials[0].normalTexture.index,1);
  assert.equal(json.materials[0].pbrMetallicRoughness.roughnessFactor,0.7);
  assert.equal(json.accessors[2].count,3);
}finally{globalThis.document=oldDocument;}
console.log('PASS counted full dilation equivalence and before-allocation actual GLB guard/normal/material contracts; synthetic JPEG only');
