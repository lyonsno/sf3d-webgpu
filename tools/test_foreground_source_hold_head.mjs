// Non-GPU live conformance over the exact selected service on this M2 source.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {startForegroundService} from './foreground_service.mjs';
const repoRoot = path.resolve(process.argv[2] ?? '');
assert.ok(process.argv[2], 'explicit actual Kaminos source root required');
const revision = execFileSync('git', ['rev-parse', 'HEAD'], {cwd: repoRoot, encoding: 'utf8'}).trim();
const out = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'sf3d-source-head-conformance-')));
for (const sourceHoldHead of [true, false]) {
  const service = await startForegroundService({repoRoot, revision, outputDir: path.join(out, String(sourceHoldHead)), sourceHoldHead});
  try {
    const origin = service.receipt.origin;
    const preflight = await (await fetch(origin+'/api/sf3d-source-admission?requestId=live-head-conformance')).json();
    assert.equal(preflight.observation.machine, 'Mac14,9', 'this is live little-box source conformance, not portable synthetic policy');
    assert.equal(preflight.verdict, 'refused');
    for (const route of ['/lib/sf3d/weights.bin', '/lib/sf3d/%77eights.bin']) {
      const response = await fetch(origin+route, {method: 'HEAD'});
      assert.equal(response.status, sourceHoldHead ? 503 : 409,
        'explicit source HEAD must reach actual Kaminos refusal, never substitute selected-consumer409');
      assert.equal((await response.arrayBuffer()).byteLength, 0);
      if (sourceHoldHead) assert.equal(response.headers.get('x-sf3d-memory-authority'), 'circuit-breaker-only');
    }
    for (const [method, route] of [['GET', '/lib/sf3d/weights.bin'], ['HEAD', '/other.bin'],
      ['GET', '/other.glb'], ['GET', '/api/read?root=assets&path=other.bin']]) {
      assert.equal((await fetch(origin+route, {method})).status, 409, 'other model/data restrictions remain held');
    }
    assert.equal(service.receipt.sourceHoldHead, sourceHoldHead, 'receipt binds effective adapter policy');
  } finally {
    const closed = await service.close();
    assert.equal(closed.exitObserved, true);
  }
}
console.log('Live selected service delegates only explicit canonical weight HEAD to actual M2 source503; other model routes remain409');
