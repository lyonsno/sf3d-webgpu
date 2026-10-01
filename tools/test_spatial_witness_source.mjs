import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'spatial-source-'));
const git = (...args) => execFileSync('git', args, { cwd: dir, encoding: 'utf8' });
const witness = new URL('./test_spatial_postprocessor_browser.mjs', import.meta.url).pathname;
try {
  git('init'); git('config', 'user.name', 'Local fixture'); git('config', 'user.email', 'fixture@example.invalid');
  fs.writeFileSync(path.join(dir, 'fixture.js'), 'export const version = 1;');
  git('add', 'fixture.js'); git('commit', '-m', 'Fixture');
  for (const staged of [false, true]) {
    fs.writeFileSync(path.join(dir, 'new-implementation.js'), 'export const version = 2;');
    if (staged) git('add', 'new-implementation.js');
    const out = fs.mkdtempSync(path.join(os.tmpdir(), 'spatial-source-result-'));
    try {
      spawnSync(process.execPath, [witness, '--source-preflight-only', '--output-dir', out, '--chrome', '/missing-independent-test-browser'], { cwd: dir });
      const report = JSON.parse(fs.readFileSync(path.join(out, 'report.json')));
      assert.equal(report.phase, 'source-preflight', `must reject ${staged ? 'staged' : 'untracked'} source before browser startup`);
      assert.match(report.error, /clean committed source/);
      assert.equal(report.ok, false);
    } finally { fs.rmSync(out, { recursive: true, force: true }); }
  }
} finally { fs.rmSync(dir, { recursive: true, force: true }); }
console.log('Spatial witness rejects untracked and staged implementation before execution');
