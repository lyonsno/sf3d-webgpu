import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const repo = path.resolve(new URL('..', import.meta.url).pathname);
const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sf3d-packed-linear-failure-'));
const reportPath = path.join(outputDir, 'report.json');
const result = spawnSync(process.execPath, [
  'tools/smoke_packed_fp16_linear.mjs',
  '--report', reportPath,
], {
  cwd: repo,
  encoding: 'utf8',
});

assert.notEqual(result.status, 0, 'missing producer identity must fail');
assert.equal(fs.existsSync(reportPath), true, `terminal report missing; stderr=${result.stderr}`);
const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
assert.equal(report.schema, 'sf3d.packed-fp16-linear-browser-smoke.v0');
assert.equal(report.ok, false);
assert.equal(report.terminal.primaryOutputWritten, false);
assert.match(report.failure.message, /kit-producer-revision|kit-tarball/);

console.log('packed fp16 linear failure-report contracts passed');
