import assert from 'node:assert/strict';
import { getDummyBias } from '../src/lib/shader_ops.js';

globalThis.GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4 };

function device() {
  return {
    createBuffer({ size, label }) {
      const bytes = new ArrayBuffer(size);
      return { label, getMappedRange: () => bytes, unmap() {} };
    },
  };
}

const first = device();
const second = device();
assert.equal(getDummyBias(first), getDummyBias(first));
assert.notEqual(getDummyBias(first), getDummyBias(second));
console.log('activation dummy bias stays cached on its owning GPUDevice');
