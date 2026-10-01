import assert from 'node:assert/strict';
import { SF3DImageTokenizer } from '../src/lib/sf3d_backbone.js';

const tokenizer = Object.create(SF3DImageTokenizer.prototype);
const events = [];
const tokens = {};
tokenizer._setupEncode = () => ({ currentTokens: tokens, D: 1024, tokenW: 36, tokenH: 36 });
tokenizer._encodeBlock = (_, block) => events.push(`encode:${block + 1}`);
tokenizer._finalizeEncode = () => ({ final: true });
const errors = [];
const result = await tokenizer.encodeCooperative({
  numBlocks: 24, chunkBlocks: 5,
  driver: async (start, end, encode) => { encode({}); events.push(`submitted:${end}`); },
  onBlockTokens: async payload => {
    assert.equal(events.at(-1), `submitted:${payload.completedBlocks}`);
    assert.equal(payload.tokensBuf, tokens);
    assert.equal(payload.totalBlocks, 24);
    assert.equal(payload.width * payload.height, 1296);
    events.push(`observed:${payload.completedBlocks}`);
    if (payload.completedBlocks === 10) throw new Error('viewer unavailable');
  },
  onPreviewError: ({ stageId, error }) => { errors.push([stageId, error.message]); throw new Error('notification also failed'); },
});
assert.deepEqual(events.filter(e => e.startsWith('observed')), ['observed:5', 'observed:10', 'observed:15', 'observed:20', 'observed:24']);
assert.deepEqual(result, { final: true });
assert.deepEqual(errors, [['dino-block-10', 'viewer unavailable']]);
console.log('DINO observations occur after each submitted chunk, carry totals, and cannot abort inference');
