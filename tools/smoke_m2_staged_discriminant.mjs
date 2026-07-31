#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import {
  buildGreenroomClaimArgs,
  buildGreenroomOwnershipUnknownArgs,
  buildGreenroomReleaseArgs,
  buildGreenroomRenewArgs,
  buildM2ChildInvocation,
  parseM2StagedDiscriminantOptions,
} from './m2_staged_discriminant_contract.mjs';
import {
  createParentPhaseJournal,
  replayParentPhaseJournal,
  resolveDurableArtifactPath,
  writeJsonReportDurable,
} from './parent_phase_journal.mjs';

const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const ROUTE_ID = 'sf3d.image-to-mesh.webgpu-local.v0';
const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const options = parseM2StagedDiscriminantOptions(process.argv.slice(2));
const journalPath = resolveDurableArtifactPath(options.journalPath);
const reportPath = resolveDurableArtifactPath(options.reportPath);
const childReportPath = resolveDurableArtifactPath(options.childReportPath);
if (new Set([journalPath, reportPath, childReportPath]).size !== 3) {
  throw new Error('journal, parent report, and child report paths must be distinct');
}
for (const outputPath of [journalPath, reportPath, childReportPath]) {
  if (fs.existsSync(outputPath)) throw new Error(`stale output already exists: ${outputPath}`);
}

const invocationId = `sf3d-m2-${Date.now()}-${process.pid}`;
const invocation = buildM2ChildInvocation({ ...options, childReportPath });
const journal = createParentPhaseJournal({
  journalPath,
  invocationId,
  requested: {
    routeId: ROUTE_ID,
    revision: options.expectedRevision,
    mode: options.mode,
    arm: options.arm,
    expectedOutputSha: options.expectedOutputSha,
    image: options.image,
    batch: options.batch,
    parentReportPath: reportPath,
    childReportPath,
    greenroom: {
      root: options.greenroomRoot,
      queueDir: options.greenroomQueueDir,
      owner: options.greenroomOwner,
      agentId: options.greenroomAgentId,
      handoffBumpId: options.greenroomHandoffBumpId,
    },
  },
});

let child = null;
let lease = null;
let terminalWritten = false;
let checkpointBuffer = '';
let acceptCheckpoints = true;
let observedBrowserPid = null;
let shutdownSignal = null;

const greenroomBinary = path.join(options.greenroomRoot, '.venv/bin/gpu-greenroom');
function greenroom(args) {
  const output = execFileSync(
    greenroomBinary,
    ['--queue-dir', options.greenroomQueueDir, ...args],
    { cwd: options.greenroomRoot, encoding: 'utf8' },
  );
  return JSON.parse(output);
}

function processGroup(pid) {
  const value = execFileSync('ps', ['-o', 'pgid=', '-p', String(pid)], { encoding: 'utf8' }).trim();
  return Number(value);
}

function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function sourceIdentity() {
  const revision = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: REPO, encoding: 'utf8' }).trim();
  const dirty = execFileSync('git', ['status', '--short'], { cwd: REPO, encoding: 'utf8' }).trim();
  const packageLock = JSON.parse(fs.readFileSync(path.join(REPO, 'package-lock.json'), 'utf8'));
  const kit = packageLock.packages?.['node_modules/@kaminos/webgpu-inference-kit'] ?? null;
  const greenroomRevision = execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: options.greenroomRoot,
    encoding: 'utf8',
  }).trim();
  return {
    routeId: ROUTE_ID,
    revision,
    revisionMatchesRequested: revision === options.expectedRevision,
    clean: !dirty,
    dirtyPaths: dirty ? dirty.split('\n') : [],
    mode: options.mode,
    arm: options.arm,
    browserPath: fs.realpathSync(CHROME_PATH),
    kit: kit ? { version: kit.version, resolved: kit.resolved, integrity: kit.integrity } : null,
    greenroom: {
      root: fs.realpathSync(options.greenroomRoot),
      queueDir: fs.realpathSync(options.greenroomQueueDir),
      revision: greenroomRevision,
      binary: fs.realpathSync(greenroomBinary),
      binarySha256: sha256File(greenroomBinary),
    },
    host: {
      hostname: os.hostname(), platform: os.platform(), release: os.release(),
      architecture: os.arch(), totalMemoryBytes: os.totalmem(),
      logicalCpuCount: os.cpus().length, cpuModel: os.cpus()[0]?.model ?? null,
    },
  };
}

function claimLease() {
  const route = [invocation.command, ...invocation.args].join(' ');
  const args = buildGreenroomClaimArgs({
    options,
    invocationId,
    repoRoot: REPO,
    pid: process.pid,
    processGroup: processGroup(process.pid),
    effectiveRoute: route,
  });
  const claimed = greenroom(args);
  if (claimed.lifecycle_state !== 'active' || claimed.lease_id !== invocationId) {
    throw new Error(`Greenroom did not return the active requested lease: ${JSON.stringify(claimed)}`);
  }
  return claimed;
}

function renewLease() {
  const renewed = greenroom(buildGreenroomRenewArgs(lease.lease_id));
  journal.append('greenroom-lease-renewed', renewed);
  return renewed;
}

function requestShutdown(signal, detail = {}) {
  if (shutdownSignal) return;
  shutdownSignal = signal;
  acceptCheckpoints = false;
  stopIntervals();
  journal.append('shutdown-requested', { signal, childPid: child?.pid ?? null, ...detail });
  if (child && !child.killed) child.kill('SIGTERM');
}

function releaseLease(reason) {
  if (!lease || lease.lifecycle_state !== 'active') return null;
  const released = greenroom(buildGreenroomReleaseArgs({
    leaseId: lease.lease_id,
    releasedBy: options.greenroomOwner,
    reason,
  }));
  lease = released;
  journal.append('greenroom-lease-released', released);
  return released;
}

function markLeaseOwnershipUnknown(reason) {
  if (!lease?.lease_id) return null;
  const unknown = greenroom(buildGreenroomOwnershipUnknownArgs(lease.lease_id));
  lease = unknown;
  journal.append('greenroom-lease-ownership-unknown', { ...unknown, reason });
  return unknown;
}

function processRssBytes(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return null;
  try {
    const kib = Number(execFileSync('ps', ['-o', 'rss=', '-p', String(pid)], { encoding: 'utf8' }).trim());
    return Number.isFinite(kib) ? kib * 1024 : null;
  } catch { return null; }
}

function writeReplayReport() {
  const replay = replayParentPhaseJournal(journalPath);
  writeJsonReportDurable(reportPath, replay);
  return replay;
}

function appendTerminal(status, payload = {}) {
  if (terminalWritten) return;
  journal.append('terminal', { status, ...payload });
  terminalWritten = true;
  acceptCheckpoints = false;
  journal.close();
  writeReplayReport();
}

function consumeCheckpointData(chunk) {
  if (!acceptCheckpoints) return;
  checkpointBuffer += chunk.toString();
  while (checkpointBuffer.includes('\n')) {
    const newline = checkpointBuffer.indexOf('\n');
    const line = checkpointBuffer.slice(0, newline);
    checkpointBuffer = checkpointBuffer.slice(newline + 1);
    if (!line || !acceptCheckpoints) continue;
    try {
      const checkpoint = JSON.parse(line);
      if (Number.isSafeInteger(checkpoint.browserPid)) observedBrowserPid = checkpoint.browserPid;
      const type = checkpoint.event === 'phase-after' && checkpoint.trustworthy
        ? 'phase-checkpoint' : 'child-checkpoint';
      journal.append(type, type === 'phase-checkpoint'
        ? { ...checkpoint, boundary: checkpoint.boundary ?? checkpoint.phase }
        : checkpoint);
    } catch (error) {
      if (acceptCheckpoints) journal.append('checkpoint-parse-failure', { line, error: error.message });
    }
  }
}

let heartbeat = null;
let renewal = null;
function stopIntervals() {
  if (heartbeat) clearInterval(heartbeat);
  if (renewal) clearInterval(renewal);
}

function finalizeAfterChild(code, signal, childError = null) {
  if (terminalWritten) return;
  stopIntervals();
  acceptCheckpoints = false;
  let childReport = null;
  try { childReport = JSON.parse(fs.readFileSync(childReportPath, 'utf8')); } catch {}
  let releaseError = null;
  try { releaseLease(shutdownSignal ? `staged run interrupted by ${shutdownSignal}` : 'staged run exited'); }
  catch (error) {
    releaseError = error.message;
    try { markLeaseOwnershipUnknown(`lease release failed: ${releaseError}`); }
    catch (unknownError) { releaseError += `; ownership-unknown transition failed: ${unknownError.message}`; }
  }
  const interrupted = shutdownSignal !== null;
  const succeeded = !interrupted && !childError && code === 0 && childReport?.ok === true && !releaseError;
  appendTerminal(interrupted ? 'interrupted' : (succeeded ? 'succeeded' : 'failed'), {
    signal: shutdownSignal,
    childExitCode: code,
    childSignal: signal,
    childError,
    childReportStatus: childReport?.status ?? null,
    childReportOk: childReport?.ok ?? null,
    childReportPath,
    greenroomReleaseError: releaseError,
    greenroomLease: lease,
    lastChildReport: childReport,
  });
  process.exitCode = interrupted ? 128 : (succeeded ? 0 : 1);
}

try {
  const effective = sourceIdentity();
  journal.append('effective-identity', effective);
  if (!effective.revisionMatchesRequested) throw new Error(`effective revision ${effective.revision} does not match requested ${options.expectedRevision}`);
  if (!effective.clean) throw new Error(`source worktree is dirty: ${effective.dirtyPaths.join(', ')}`);
  if (effective.kit?.version !== '0.1.42') throw new Error(`effective inference-kit must be 0.1.42, got ${effective.kit?.version ?? 'missing'}`);

  lease = claimLease();
  journal.append('greenroom-lease-claimed', lease);
  journal.append('heavy-work-start', { phase: 'child-launch', command: invocation.command, args: invocation.args });
  child = spawn(invocation.command, invocation.args, {
    cwd: REPO, env: invocation.env, stdio: ['ignore', 'pipe', 'pipe', 'pipe'],
  });
  child.stdout.pipe(process.stdout);
  child.stderr.pipe(process.stderr);
  child.stdio[3].on('data', consumeCheckpointData);

  heartbeat = setInterval(() => {
    if (!acceptCheckpoints) return;
    journal.append('resource-heartbeat', {
      childPid: child.pid, childRssBytes: processRssBytes(child.pid),
      browserPid: observedBrowserPid, browserRssBytes: processRssBytes(observedBrowserPid),
      hostFreeMemoryBytes: os.freemem(), hostTotalMemoryBytes: os.totalmem(),
      hostLoadAverage: os.loadavg(), greenroomLeaseId: lease.lease_id,
    });
  }, 5000);
  heartbeat.unref();
  renewal = setInterval(() => {
    if (!acceptCheckpoints) return;
    try {
      lease = renewLease();
    } catch (error) {
      requestShutdown('GREENROOM_LEASE_RENEWAL_FAILED', { error: error.message });
    }
  }, 60000);
  renewal.unref();

  process.once('SIGINT', () => requestShutdown('SIGINT'));
  process.once('SIGTERM', () => requestShutdown('SIGTERM'));
  child.once('error', error => finalizeAfterChild(null, null, error.message));
  child.once('close', (code, signal) => finalizeAfterChild(code, signal));
} catch (error) {
  let releaseError = null;
  try { releaseLease('staged preflight failed'); }
  catch (releaseFailure) {
    releaseError = releaseFailure.message;
    try { markLeaseOwnershipUnknown(`preflight release failed: ${releaseError}`); }
    catch (unknownError) { releaseError += `; ownership-unknown transition failed: ${unknownError.message}`; }
  }
  appendTerminal('failed', { failurePhase: 'preflight', error: error.message, greenroomReleaseError: releaseError, greenroomLease: lease });
  console.error(`M2 STAGED DISCRIMINANT FAILED [preflight]: ${error.message}`);
  process.exitCode = 1;
}
