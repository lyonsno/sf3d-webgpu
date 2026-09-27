import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runOptionalPreview } from '../src/lib/preview_policy.js';

const failure = new Error('preview projection failed');
const seen = [];
const skipped = await runOptionalPreview('block-0-fuse-out', async () => { throw failure; }, event => seen.push(event));
assert.equal(skipped.status, 'skipped');
assert.equal(seen.length, 1);
assert.equal(seen[0].stageId, 'block-0-fuse-out');
assert.equal(seen[0].error, failure);

const complete = await runOptionalPreview('block-1-fuse-out', async () => 42, () => { throw new Error('should not notify'); });
assert.deepEqual(complete, { status: 'complete', value: 42 });

const notificationFailure = await runOptionalPreview('block-0-fuse-out', async () => { throw failure; }, () => { throw new Error('notifier failed'); });
assert.equal(notificationFailure.status, 'skipped');
assert.equal(notificationFailure.error, failure);
assert.match(notificationFailure.notificationError.message, /notifier failed/);
const inference = readFileSync(new URL('../src/lib/inference.js', import.meta.url), 'utf8');
assert.match(inference, /onStageComplete:[\s\S]*runOptionalPreview\(stageId/);
console.log('optional previews cannot replace the primary inference outcome');
