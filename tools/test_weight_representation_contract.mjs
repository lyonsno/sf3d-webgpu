import assert from 'node:assert/strict';
import { summarizeWeightRepresentation } from '../src/lib/weights.js';

const summary = summarizeWeightRepresentation(new Map([
  ['fp32.tensor', { dtype: 0, size: 16 }],
  ['fp16.tensor-a', { dtype: 1, size: 8 }],
  ['fp16.tensor-b', { dtype: 1, size: 4 }],
]));
assert.deepEqual(summary.sourceEncoding, {
  fp32: { tensorCount: 1, bytes: 16 },
  fp16: { tensorCount: 2, bytes: 12 },
});
assert.equal(summary.gpuStorage.format, 'fp32');
assert.equal(summary.gpuStorage.bytesPerElement, 4);
assert.equal(summary.lazyRawPayload.format, 'source-encoded');
assert.throws(() => summarizeWeightRepresentation(new Map([['bad', { dtype: 2, size: 8 }]])), /Unsupported tensor dtype 2/);
console.log('weight representation contract passed');
