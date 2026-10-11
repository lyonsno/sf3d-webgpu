import assert from 'node:assert/strict';
import {TriplaneDecoder} from '../src/lib/triplane_decoder.js';
import {captureGpuBufferAllocations} from '../src/lib/gpu.js';
globalThis.GPUBufferUsage={UNIFORM:1,COPY_DST:2};
let allocated,destroyed=0;
const device={createBuffer({size}){allocated={size,destroy(){destroyed++;},
  getMappedRange(){throw Error('actual uniform map failure');}};return allocated;}};
const decoder=new TriplaneDecoder(device),owned=[];
// The established cooperative bake retires captured per-range scratch. Its
// cached uniforms must not become that scratch and be reused after destruction.
const stableDevice={createBuffer({size}){return {size,
  getMappedRange(){return new ArrayBuffer(size);},unmap(){},destroy(){assert.fail('cached default uniform retired as range scratch');}};}};
const stable=new TriplaneDecoder(stableDevice);
const captured=captureGpuBufferAllocations(()=>stable._cachedUniform(new Uint32Array([1,2,3])));
assert.equal(captured.allocations.length,0,'default cached uniforms are not per-range scratch');
assert.equal(stable._cachedUniform(new Uint32Array([1,2,3])),captured.value);
decoder._residentQueryWorkOwner={allocations:owned};
assert.throws(()=>captureGpuBufferAllocations(()=>decoder._cachedUniform(new Uint32Array([1,2,3])),
  {ownedAllocations:owned}),/actual uniform map failure/);
assert.equal(owned.length,1,'decoder uniform must enter caller inventory before map failure');
assert.equal(owned[0].buffer,allocated);
assert.equal(destroyed,0,'caller owns recovery, not a lost or double-retired store');
assert.equal(decoder._uniformCache.size,0,'failed uniform initialization cannot enter cache');
console.log('PASS actual decoder uniform synchronous failure capture');
