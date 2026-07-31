import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  buildM2ChildInvocation,
  parseM2StagedDiscriminantOptions,
} from './m2_staged_discriminant_contract.mjs';

const revision = 'd'.repeat(40);
const canonicalSha = 'e1f70de3407df24d571bf68f70fac2b59373bdd948075a2387f1834e4faff8b7';

assert.throws(
  () => parseM2StagedDiscriminantOptions(['--mode', 'setup-only']),
  /--journal is required/,
);
assert.throws(
  () => parseM2StagedDiscriminantOptions([
    '--journal', '/Users/noahlyons/.local/state/sf3d/journal.jsonl',
    '--mode', 'single-arm',
  ]),
  /--arm is required/,
);
assert.throws(
  () => parseM2StagedDiscriminantOptions([
    '--journal', '/Users/noahlyons/.local/state/sf3d/journal.jsonl',
    '--mode', 'setup-only',
    '--arm', 'monolithic',
  ]),
  /does not accept --arm/,
);
assert.throws(
  () => parseM2StagedDiscriminantOptions([
    '--journal', '/Users/noahlyons/.local/state/sf3d/journal.jsonl',
    '--mode', 'single-arm',
    '--arm', 'not-a-route',
  ]),
  /unknown arm/,
);
assert.throws(
  () => parseM2StagedDiscriminantOptions([
    '--journal', '/Users/noahlyons/.local/state/sf3d/journal.jsonl',
    '--mode', 'single-arm',
    '--arm', 'arena-plus-worker',
    '--timeout', '60',
  ]),
  /unknown option --timeout/,
  'the staged runner must not silently impose an execution timeout',
);

const setup = parseM2StagedDiscriminantOptions([
  '--journal', '/Users/noahlyons/.local/state/sf3d/setup.jsonl',
  '--mode', 'setup-only',
  '--expected-revision', revision,
]);
assert.equal(setup.mode, 'setup-only');
assert.equal(setup.arm, null);
assert.equal(setup.reportPath, '/Users/noahlyons/.local/state/sf3d/setup.report.json');

const single = parseM2StagedDiscriminantOptions([
  '--journal', '/Users/noahlyons/.local/state/sf3d/single.jsonl',
  '--mode', 'single-arm',
  '--arm', 'arena-plus-worker',
  '--expected-revision', revision,
  '--expected-output-sha', canonicalSha,
  '--batch', '4096',
]);
const invocation = buildM2ChildInvocation(single);
assert.deepEqual(invocation.args.slice(0, 2), ['tools/smoke_five_arm_ab.mjs', '--profile']);
assert.ok(invocation.args.includes('single-arm'));
assert.ok(invocation.args.includes('arena-plus-worker'));
assert.ok(invocation.args.includes(canonicalSha));
assert.equal(invocation.env.SF3D_PARENT_CHECKPOINT_FD, '3');
assert.equal(invocation.timeout, undefined);

const smokeSource = fs.readFileSync(new URL('./smoke_five_arm_ab.mjs', import.meta.url), 'utf8');
assert.match(smokeSource, /setup-only/);
assert.match(smokeSource, /single-arm/);
assert.match(smokeSource, /__sf3dParentCheckpoint/);
assert.match(smokeSource, /emitParentCheckpoint\('phase-before'/);
assert.match(smokeSource, /emitParentCheckpoint\('phase-after'/);

const wrapperSource = fs.readFileSync(new URL('./smoke_m2_staged_discriminant.mjs', import.meta.url), 'utf8');
assert.match(wrapperSource, /resource-heartbeat/);
assert.match(wrapperSource, /replayParentPhaseJournal/);
assert.match(wrapperSource, /stdio: \['ignore', 'pipe', 'pipe', 'pipe'\]/);
assert.doesNotMatch(wrapperSource, /setTimeout\([^)]*process\.kill/);

console.log('M2 staged discriminant contract passed');

