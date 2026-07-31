import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  buildGreenroomClaimArgs,
  buildGreenroomOwnershipUnknownArgs,
  buildGreenroomReleaseArgs,
  buildGreenroomRenewArgs,
  buildM2ChildInvocation,
  parseM2StagedDiscriminantOptions,
} from './m2_staged_discriminant_contract.mjs';

const revision = 'd'.repeat(40);
const canonicalSha = 'e1f70de3407df24d571bf68f70fac2b59373bdd948075a2387f1834e4faff8b7';
const greenroomArgs = [
  '--greenroom-root', '/Users/noahlyons/dev/gpu-greenroom',
  '--greenroom-queue-dir', '/Users/noahlyons/.local/state/gpu-greenroom',
  '--greenroom-owner', 'mini-wake-and-bake-pit-boss',
  '--greenroom-agent-id', 'mini-wake-and-bake-pit-boss',
];

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
  ...greenroomArgs,
]);
assert.equal(setup.mode, 'setup-only');
assert.equal(setup.arm, null);
assert.equal(setup.reportPath, '/Users/noahlyons/.local/state/sf3d/setup.report.json');
assert.equal(setup.childReportPath, '/Users/noahlyons/.local/state/sf3d/setup.report.child.json');

const single = parseM2StagedDiscriminantOptions([
  '--journal', '/Users/noahlyons/.local/state/sf3d/single.jsonl',
  '--mode', 'single-arm',
  '--arm', 'arena-plus-worker',
  '--expected-revision', revision,
  '--expected-output-sha', canonicalSha,
  '--batch', '4096',
  ...greenroomArgs,
]);
const invocation = buildM2ChildInvocation(single);
assert.deepEqual(invocation.args.slice(0, 2), ['tools/smoke_five_arm_ab.mjs', '--profile']);
assert.ok(invocation.args.includes('single-arm'));
assert.ok(invocation.args.includes('arena-plus-worker'));
assert.ok(invocation.args.includes(canonicalSha));
assert.equal(invocation.env.SF3D_PARENT_CHECKPOINT_FD, '3');

const claimArgs = buildGreenroomClaimArgs({
  options: { ...single, greenroomHandoffBumpId: 'handoff-7' },
  invocationId: 'sf3d-m2-test',
  repoRoot: '/repo/sf3d-webgpu',
  pid: 123,
  processGroup: 120,
  effectiveRoute: 'node tools/smoke_five_arm_ab.mjs --profile single-arm',
});
assert.deepEqual(claimArgs.slice(0, 4), ['lease', 'claim', '--lease-id', 'sf3d-m2-test']);
assert.ok(claimArgs.includes('--supports-checkpoints'));
assert.deepEqual(claimArgs.slice(-2), ['--handoff-bump-id', 'handoff-7']);
assert.deepEqual(buildGreenroomRenewArgs('lease-1'), [
  'lease', 'renew', 'lease-1', '--ttl-seconds', '300',
]);
assert.deepEqual(buildGreenroomReleaseArgs({
  leaseId: 'lease-1', releasedBy: 'mini', reason: 'complete',
}), ['lease', 'release', 'lease-1', '--released-by', 'mini', '--reason', 'complete']);
assert.deepEqual(buildGreenroomOwnershipUnknownArgs('lease-1'), [
  'lease', 'renew', 'lease-1', '--lifecycle-state', 'ownership_unknown', '--not-interruptible',
]);
assert.equal(invocation.timeout, undefined);

const smokeSource = fs.readFileSync(new URL('./smoke_five_arm_ab.mjs', import.meta.url), 'utf8');
assert.match(smokeSource, /setup-only/);
assert.match(smokeSource, /single-arm/);
assert.match(smokeSource, /__sf3dParentCheckpoint/);
assert.match(smokeSource, /emitParentCheckpoint\('phase-before'/);
assert.match(smokeSource, /emitParentCheckpoint\('phase-after'/);
assert.match(smokeSource, /writeJsonReportDurable\(REPORT_PATH/);
assert.doesNotMatch(smokeSource, /writeFileSync\(REPORT_PATH/);

const replaySource = fs.readFileSync(new URL('./replay_parent_phase_journal.mjs', import.meta.url), 'utf8');
assert.match(replaySource, /writeJsonReportDurable\(reportPath/);
assert.doesNotMatch(replaySource, /writeJsonReportAtomic/);

const wrapperSource = fs.readFileSync(new URL('./smoke_m2_staged_discriminant.mjs', import.meta.url), 'utf8');
assert.match(wrapperSource, /resource-heartbeat/);
assert.match(wrapperSource, /buildGreenroomClaimArgs/);
assert.match(wrapperSource, /buildGreenroomRenewArgs/);
assert.match(wrapperSource, /buildGreenroomReleaseArgs/);
assert.match(wrapperSource, /if \(terminalWritten\) return/);
assert.match(wrapperSource, /acceptCheckpoints = false/);
assert.match(wrapperSource, /replayParentPhaseJournal/);
assert.match(wrapperSource, /stdio: \['ignore', 'pipe', 'pipe', 'pipe'\]/);
assert.doesNotMatch(wrapperSource, /setTimeout\([^)]*process\.kill/);

console.log('M2 staged discriminant contract passed');
