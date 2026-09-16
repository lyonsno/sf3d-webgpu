#!/usr/bin/env node
/**
 * Weight-chunk compaction contract (src/lib/weights.js compactRetainedTensors).
 *
 * loadWeights streams weights.bin into ~2.1 GB of chunks. The eager builders
 * upload most tensors to GPU buffers, but the CLIP estimator reads its
 * tensors lazily through the raw accessors, which used to keep every chunk
 * alive for the producer's lifetime (2.25 GB JS heap in the live-flame
 * composition page). After the eager pass, only the lazily read families and
 * any tensor not consumed eagerly are copied out; the chunks are released.
 */
import assert from 'node:assert/strict';
import { compactRetainedTensors, extractBytesFromChunks } from '../src/lib/weights.js';

// Two chunks of 8 bytes; three tensors: A [0,6), B [6,10) (spans the chunk boundary), C [10,16).
const chunks = [new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]), new Uint8Array([8, 9, 10, 11, 12, 13, 14, 15])];
const chunked = { chunks, offsets: [0, 8], totalSize: 16 };
const tensors = new Map([
  ['dino.a', { dtype: 1, offset: 0, size: 6 }],
  ['image_estimator.b', { dtype: 1, offset: 6, size: 4 }],
  ['other.c', { dtype: 1, offset: 10, size: 6 }],
]);
assert.deepEqual([...extractBytesFromChunks(chunked, 6, 4)], [6, 7, 8, 9], 'cross-chunk extraction');

// dino.a was consumed eagerly (uploaded); image_estimator.* is a retained family; other.c was never consumed.
const consumed = new Set(['dino.a', 'image_estimator.b']);
const r = compactRetainedTensors(tensors, chunked, consumed, ['image_estimator.']);
assert.deepEqual([...r.retained.keys()].sort(), ['image_estimator.b', 'other.c']);
assert.deepEqual([...r.retained.get('image_estimator.b')], [6, 7, 8, 9]);
assert.deepEqual([...r.retained.get('other.c')], [10, 11, 12, 13, 14, 15]);
assert.equal(r.retainedBytes, 10); assert.equal(r.droppedBytes, 6);
// Retained copies own their memory: the chunks are released and no longer referenced.
assert.equal(chunked.chunks.length, 0, 'chunks released');
assert.equal(chunked.offsets.length, 0);
assert.equal(r.retained.get('image_estimator.b').byteOffset, 0);
assert.equal(r.retained.get('image_estimator.b').buffer.byteLength, 4, 'a standalone copy, not a view into a chunk');
// Reading a dropped tensor after compaction fails loud.
assert.throws(() => r.rawBytes('dino.a'), /dino.a was uploaded at load and its raw bytes were released/);
assert.deepEqual([...r.rawBytes('other.c')], [10, 11, 12, 13, 14, 15]);
console.log('ok  compaction retains lazily read families and unconsumed tensors; releases the rest');
console.log('\nWEIGHTS COMPACTION CONTRACT PASSED');
