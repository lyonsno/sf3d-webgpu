import path from 'node:path';

const MODES = new Set(['setup-only', 'single-arm']);
const ARMS = new Set(['monolithic', 'arena-plus-worker']);
const VALUE_OPTIONS = new Set([
  '--journal',
  '--report',
  '--mode',
  '--arm',
  '--expected-revision',
  '--expected-output-sha',
  '--image',
  '--batch',
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

export function parseM2StagedDiscriminantOptions(argv) {
  const values = parseValues(argv);
  const journal = values.get('--journal');
  if (!journal) throw new Error('--journal is required');
  const mode = values.get('--mode');
  if (!MODES.has(mode)) throw new Error('--mode must be setup-only or single-arm');
  const arm = values.get('--arm') ?? null;
  if (mode === 'single-arm' && !arm) throw new Error('--arm is required for single-arm');
  if (mode === 'setup-only' && arm) throw new Error('setup-only does not accept --arm');
  if (arm && !ARMS.has(arm)) throw new Error(`unknown arm: ${arm}`);

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
  return Object.freeze({
    journalPath,
    reportPath: path.resolve(values.get('--report') ?? reportPathFor(journalPath)),
    mode,
    arm,
    expectedRevision: expectedRevision.toLowerCase(),
    expectedOutputSha: expectedOutputSha?.toLowerCase() ?? null,
    image: values.get('--image') ? path.resolve(values.get('--image')) : null,
    batch,
  });
}

export function buildM2ChildInvocation(options) {
  const args = [
    'tools/smoke_five_arm_ab.mjs',
    '--profile', options.mode,
    '--expected-revision', options.expectedRevision,
    '--report', options.reportPath,
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

