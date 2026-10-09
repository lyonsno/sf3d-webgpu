import assert from 'node:assert/strict';
import fs from 'node:fs';
import { prepareForeground as prepare } from '../src/lib/foreground_prepare.js';
const fixture = JSON.parse(fs.readFileSync(new URL('./fixtures/foreground-reference.json', import.meta.url)));
for (const c of fixture.cases) {
  const input = Uint8Array.from(c.input);
  const result = prepare(input, c.width, c.height, c.size);
  assert.equal(result.data.length, c.size * c.size * 4);
  assert.deepEqual(result.crop, c.crop);
  assert.deepEqual([...result.data], c.output, `${c.name}: pixels must match upstream framing + Pillow bicubic RGBA`);
  assert.deepEqual([...input], c.input, 'source pixels remain caller-owned');
}
assert.throws(() => prepare(new Uint8Array(16), 2, 2), /Empty foreground mask/);
assert.throws(() => prepare(new Uint8Array([1, 2, 3, 255]), 1, 1), /crop is empty/);
assert.throws(() => prepare(new Uint8Array(15), 2, 2), /matching RGBA/);
console.log('foreground Pillow reference tests passed');
