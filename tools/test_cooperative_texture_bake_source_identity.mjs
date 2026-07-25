#!/usr/bin/env node

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';

const smokePath = new URL('./smoke_cooperative_bake_ab.mjs', import.meta.url);
const repo = path.resolve(new URL('..', import.meta.url).pathname);
const source = fs.readFileSync(smokePath, 'utf8');

assert.match(source, /--expected-revision/, 'smoke must require a requested source revision');
assert.match(source, /sourceIdentity\.matchesRequestedRevision/, 'pass predicate must bind requested revision');
assert.match(source, /sourceIdentity\.clean/, 'pass predicate must bind clean source state');
assert.match(source, /dirtyDiffSha256/, 'explicit dirty evidence must carry an inspectable diff identity');
assert.match(source, /frameGapOverlaps/, 'frame attribution must use interval overlap');
assert.match(
  source,
  /frameGapAttribution:\s*result\.on\.frameGapAttribution/,
  'persisted cooperative arm must retain its overlap attribution',
);

function runPreflight(args) {
  const reportPath = path.join(
    os.tmpdir(),
    `sf3d-source-identity-${process.pid}-${Math.random().toString(16).slice(2)}.json`,
  );
  const result = spawnSync(
    process.execPath,
    [smokePath.pathname, '--report', reportPath, ...args],
    { cwd: repo, encoding: 'utf8' },
  );
  const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
  fs.rmSync(reportPath);
  return { result, report };
}

const missing = runPreflight([]);
assert.notEqual(missing.result.status, 0);
assert.equal(missing.report.failurePhase, 'source-identity');
assert.match(missing.report.error, /expected-revision/);

const unresolvable = runPreflight(['--expected-revision', 'definitely-not-a-real-revision']);
assert.notEqual(unresolvable.result.status, 0);
assert.equal(unresolvable.report.failurePhase, 'source-identity');
assert.equal(
  unresolvable.report.lastTrustworthyEvidence.source.requestedRevision,
  'definitely-not-a-real-revision',
);
assert.equal(unresolvable.report.lastTrustworthyEvidence.source.matchesRequestedRevision, false);
assert.match(unresolvable.report.error, /cannot resolve requested revision/);

const parentRevision = execFileSync('git', ['rev-parse', 'HEAD^'], {
  cwd: repo,
  encoding: 'utf8',
}).trim();
const mismatched = runPreflight(['--expected-revision', parentRevision]);
assert.notEqual(mismatched.result.status, 0);
assert.equal(mismatched.report.failurePhase, 'source-identity');
assert.equal(
  mismatched.report.lastTrustworthyEvidence.source.matchesRequestedRevision,
  false,
);

const dirtyMarker = path.join(repo, `.source-identity-test-${process.pid}`);
fs.writeFileSync(dirtyMarker, 'intentional dirty-source falsifier\n');
try {
  const dirty = runPreflight(['--expected-revision', 'HEAD']);
  assert.notEqual(dirty.result.status, 0);
  assert.equal(dirty.report.failurePhase, 'source-identity');
  assert.equal(dirty.report.lastTrustworthyEvidence.source.clean, false);
  assert.match(dirty.report.lastTrustworthyEvidence.source.dirtyDiffSha256, /^[a-f0-9]{64}$/);
} finally {
  fs.rmSync(dirtyMarker, { force: true });
}

console.log('cooperative texture-bake source identity contract passed');
