import path from 'node:path';

const MODES = new Set(['setup-only', 'single-arm', 'prefix']);
const ARMS = new Set(['monolithic', 'arena-plus-worker']);
const PREFIXES = new Set(['weights', 'model', 'pipelines', 'arena', 'worker', 'arm-entry']);
const VALUE_OPTIONS = new Set([
  '--journal',
  '--report',
  '--mode',
  '--arm',
  '--prefix',
  '--expected-revision',
  '--expected-output-sha',
  '--image',
  '--batch',
  '--greenroom-root',
  '--greenroom-queue-dir',
  '--greenroom-owner',
  '--greenroom-agent-id',
  '--greenroom-handoff-bump-id',
]);

function parseValues(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    if (!VALUE_OPTIONS.has(flag)) throw new Error(`unknown option ${flag}`);
    if (values.has(flag)) throw new Error(`${flag} may be provided only once`);
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value`);
    values.set(flag, value);
    index += 1;
  }
  return values;
}

function reportPathFor(journalPath) {
  const extension = path.extname(journalPath);
  return extension
    ? `${journalPath.slice(0, -extension.length)}.report.json`
    : `${journalPath}.report.json`;
}

function childReportPathFor(reportPath) {
  return reportPath.endsWith('.json')
    ? `${reportPath.slice(0, -5)}.child.json`
    : `${reportPath}.child.json`;
}

export function parseM2StagedDiscriminantOptions(argv) {
  const values = parseValues(argv);
  const journal = values.get('--journal');
  if (!journal) throw new Error('--journal is required');
  const mode = values.get('--mode');
  if (!MODES.has(mode)) throw new Error('--mode must be setup-only, single-arm, or prefix');
  const arm = values.get('--arm') ?? null;
  const prefix = values.get('--prefix') ?? null;
  if (mode === 'single-arm' && !arm) throw new Error('--arm is required for single-arm');
  if (mode !== 'single-arm' && arm) throw new Error(`${mode} does not accept --arm`);
  if (arm && !ARMS.has(arm)) throw new Error(`unknown arm: ${arm}`);
  if (mode === 'prefix' && !prefix) throw new Error('--prefix is required for prefix mode');
  if (mode !== 'prefix' && prefix) throw new Error(`${mode} does not accept --prefix`);
  if (prefix && !PREFIXES.has(prefix)) throw new Error(`unknown prefix: ${prefix}`);

  const expectedRevision = values.get('--expected-revision');
  if (!expectedRevision || !/^[a-f0-9]{40}$/i.test(expectedRevision)) {
    throw new Error('--expected-revision must be an exact 40-character commit');
  }
  const expectedOutputSha = values.get('--expected-output-sha') ?? null;
  if (mode === 'single-arm' && (!expectedOutputSha || !/^[a-f0-9]{64}$/i.test(expectedOutputSha))) {
    throw new Error('--expected-output-sha is required for single-arm');
  }
  const batchText = values.get('--batch') ?? '4096';
  const batch = Number(batchText);
  if (!Number.isSafeInteger(batch) || batch <= 0 || String(batch) !== batchText) {
    throw new Error('--batch must be a positive safe integer');
  }

  const journalPath = path.resolve(journal);
  const reportPath = path.resolve(values.get('--report') ?? reportPathFor(journalPath));
  if (reportPath === journalPath) throw new Error('--report must not alias --journal');
  const greenroomRoot = values.get('--greenroom-root');
  const greenroomQueueDir = values.get('--greenroom-queue-dir');
  const greenroomOwner = values.get('--greenroom-owner');
  const greenroomAgentId = values.get('--greenroom-agent-id');
  if (!greenroomRoot) throw new Error('--greenroom-root is required');
  if (!greenroomQueueDir) throw new Error('--greenroom-queue-dir is required');
  if (!greenroomOwner) throw new Error('--greenroom-owner is required');
  if (!greenroomAgentId) throw new Error('--greenroom-agent-id is required');
  return Object.freeze({
    journalPath,
    reportPath,
    childReportPath: childReportPathFor(reportPath),
    mode,
    arm,
    prefix,
    expectedRevision: expectedRevision.toLowerCase(),
    expectedOutputSha: expectedOutputSha?.toLowerCase() ?? null,
    image: values.get('--image') ? path.resolve(values.get('--image')) : null,
    batch,
    greenroomRoot: path.resolve(greenroomRoot),
    greenroomQueueDir: path.resolve(greenroomQueueDir),
    greenroomOwner,
    greenroomAgentId,
    greenroomHandoffBumpId: values.get('--greenroom-handoff-bump-id') ?? null,
  });
}

export function buildM2ChildInvocation(options) {
  if (options.mode === 'prefix') {
    const args = [
      'tools/smoke_m2_cold_browser_prefix.mjs',
      '--prefix', options.prefix,
      '--expected-revision', options.expectedRevision,
      '--report', options.childReportPath,
      '--batch', String(options.batch),
    ];
    if (options.image) args.push('--image', options.image);
    return {
      command: process.execPath,
      args,
      env: { ...process.env, SF3D_PARENT_CHECKPOINT_FD: '3' },
    };
  }
  const args = [
    'tools/smoke_five_arm_ab.mjs',
    '--profile', options.mode,
    '--expected-revision', options.expectedRevision,
    '--report', options.childReportPath,
    '--batch', String(options.batch),
  ];
  if (options.arm) args.push('--single-arm', options.arm);
  if (options.expectedOutputSha) args.push('--expected-output-sha', options.expectedOutputSha);
  if (options.image) args.push('--image', options.image);
  return {
    command: process.execPath,
    args,
    env: { ...process.env, SF3D_PARENT_CHECKPOINT_FD: '3' },
  };
}

export function buildGreenroomClaimArgs({
  options,
  invocationId,
  repoRoot,
  pid,
  processGroup,
  effectiveRoute,
}) {
  const args = [
    'lease', 'claim', '--lease-id', invocationId,
    '--owner', options.greenroomOwner, '--agent-id', options.greenroomAgentId,
    '--repo-root', repoRoot, '--pid', String(pid),
    '--process-group', String(processGroup),
    '--effective-route', effectiveRoute, '--backend', 'webgpu-metal',
    '--device', 'chrome-webgpu:metal',
    '--profile', `sf3d-m2-${options.mode}-${options.prefix ?? options.arm ?? 'setup'}`,
    '--supports-checkpoints', '--ttl-seconds', '300',
  ];
  if (options.greenroomHandoffBumpId) {
    args.push('--handoff-bump-id', options.greenroomHandoffBumpId);
  }
  return args;
}

export function buildGreenroomRenewArgs(leaseId) {
  return ['lease', 'renew', leaseId, '--ttl-seconds', '300'];
}

export function buildGreenroomReleaseArgs({ leaseId, releasedBy, reason }) {
  return ['lease', 'release', leaseId, '--released-by', releasedBy, '--reason', reason];
}

export function buildGreenroomOwnershipUnknownArgs(leaseId) {
  return [
    'lease', 'renew', leaseId,
    '--lifecycle-state', 'ownership_unknown', '--not-interruptible',
  ];
}
