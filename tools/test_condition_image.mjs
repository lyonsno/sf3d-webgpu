import assert from 'node:assert/strict';
import fs from 'node:fs';
import { resizeBlendNormalize } from '../src/lib/preprocess_core.js';
import { prepareForeground } from '../src/lib/foreground_prepare.js';
import { prepareConditionImage, validateConditionReply } from '../src/lib/condition_image.js';
const fixture = JSON.parse(fs.readFileSync(new URL('./fixtures/foreground-reference.json', import.meta.url)));
// Capture the existing worker's actual reply, without simulating its math.
let reply;
globalThis.self = { postMessage(value) { reply = value; } };
await import('../src/lib/preprocess_worker.js');
for (const c of fixture.cases) {
  const src = Float32Array.from(c.input, x => x / 255);
  const params = { srcW: c.width, srcH: c.height, size: c.size,
    bg: [0.5, 0.5, 0.5], imageMean: [0.485, 0.456, 0.406], imageStd: [0.229, 0.224, 0.225] };
  self.onmessage({ data: { ...params, srcBuffer: src.buffer, id: c.name } });
  assert.equal(reply.ok, true);
  assert.ok(reply.rgbaBuffer instanceof ArrayBuffer, 'worker must return the same framed pixels used by geometry for CLIP');
  const framed = prepareForeground(Uint8Array.from(c.input), c.width, c.height, c.size);
  assert.deepEqual(new Uint8Array(reply.rgbaBuffer), framed.data);
  const expected = resizeBlendNormalize(Float32Array.from(framed.data, x => x / 255), c.size, c.size, c.size,
    params.bg, params.imageMean, params.imageStd);
  assert.deepEqual(new Float32Array(reply.chwBuffer), expected);
  assert.deepEqual(validateConditionReply(reply, c.size), prepareConditionImage(src, c.width, c.height, c.size,
    params.bg, params.imageMean, params.imageStd));
}
const replyBase = { chwBuffer: new Float32Array(12).buffer, rgbaBuffer: new Uint8Array(16).buffer };
assert.throws(() => validateConditionReply({ ...replyBase, rgbaBuffer: new ArrayBuffer(15) }, 2), /matching RGBA/);
assert.throws(() => validateConditionReply({ ...replyBase, rgbaBuffer: undefined }, 2), /matching RGBA/);
assert.throws(() => validateConditionReply({ ...replyBase, chwBuffer: new Float32Array(12).fill(NaN).buffer }, 2), /non-finite/);
assert.throws(() => validateConditionReply({ ...replyBase, chwBuffer: new ArrayBuffer(4) }, 2), /length/);
delete globalThis.self;
console.log('shared condition image worker tests passed');
