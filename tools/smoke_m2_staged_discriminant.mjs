#!/usr/bin/env node
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import {
  buildM2ChildInvocation,
  parseM2StagedDiscriminantOptions,
} from './m2_staged_discriminant_contract.mjs';
import {
  createParentPhaseJournal,
  replayParentPhaseJournal,
} from './parent_phase_journal.mjs';
import { writeJsonReportAtomic } from './json_report_atomic.mjs';

const REPO = path.resolve(new URL('..', import.meta.url).pathname);
const ROUTE_ID = 'sf3d.image-to-mesh.webgpu-local.v0';
const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const options = parseM2StagedDiscriminantOptions(process.argv.slice(2));
const invocationId = `sf3d-m2-${Date.now()}-${process.pid}`;
const invocation = buildM2ChildInvocation(options);
const journal = createParentPhaseJournal({
  journalPath: options.journalPath,
  invocationId,
  requested: {
    routeId: ROUTE_ID,
    revision: options.expectedRevision,
    mode: options.mode,
    arm: options.arm,
    expectedOutputSha: options.expectedOutputSha,
    image: options.image,
    batch: options.batch,
    reportPath: options.reportPath,
  },
});

let child = null;
let terminalWritten = false;
let checkpointBuffer = '';

function sourceIdentity() {
  const revision = execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: REPO,
    encoding: 'utf8',
  }).trim();
  const dirty = execFileSync('git', ['status', '--short'], {
    cwd: REPO,
    encoding: 'utf8',
  }).trim();
  const packageLock = JSON.parse(fs.readFileSync(path.join(REPO, 'package-lock.json'), 'utf8'));
  const kit = packageLock.packages?.['node_modules/@kaminos/webgpu-inference-kit'] ?? null;
  return {
    routeId: ROUTE_ID,
    revision,
    revisionMatchesRequested: revision === options.expectedRevision,
    clean: !dirty,
    dirtyPaths: dirty ? dirty.split('\n') : [],
    mode: options.mode,
    arm: options.arm,
    browserPath: fs.realpathSync(CHROME_PATH),
    kit: kit ? {
      version: kit.version,
      resolved: kit.resolved,
      integrity: kit.integrity,
    } : null,
    host: {
      hostname: os.hostname(),
      platform: os.platform(),
      release: os.release(),
      architecture: os.arch(),
      totalMemoryBytes: os.totalmem(),
      logicalCpuCount: os.cpus().length,
      cpuModel: os.cpus()[0]?.model ?? null,
    },
  };
}

function processRssBytes(pid) {
  try {
    const kib = Number(execFileSync('ps', ['-o', 'rss=', '-p', String(pid)], {
      encoding: 'utf8',
    }).trim());
    return Number.isFinite(kib) ? kib * 1024 : null;
  } catch {
    return null;
  }
}

function writeReplayReport() {
  const replay = replayParentPhaseJournal(options.journalPath);
  writeJsonReportAtomic(options.reportPath, replay);
  return replay;
}

function appendTerminal(status, payload = {}) {
  if (terminalWritten) return;
  journal.append('terminal', { status, ...payload });
  terminalWritten = true;
  journal.close();
  writeReplayReport();
}

function consumeCheckpointData(chunk) {
  checkpointBuffer += chunk.toString();
  while (checkpointBuffer.includes('\n')) {
    const newline = checkpointBuffer.indexOf('\n');
    const line = checkpointBuffer.slice(0, newline);
    checkpointBuffer = checkpointBuffer.slice(newline + 1);
    if (!line) continue;
    try {
      const checkpoint = JSON.parse(line);
      const type = checkpoint.event === 'phase-after' && checkpoint.trustworthy
        ? 'phase-checkpoint'
        : 'child-checkpoint';
      journal.append(type, checkpoint.event === 'phase-after' && checkpoint.trustworthy
        ? { ...checkpoint, boundary: checkpoint.boundary ?? checkpoint.phase }
        : checkpoint);
    } catch (error) {
      journal.append('checkpoint-parse-failure', { line, error: error.message });
    }
  }
}

try {
  const effective = sourceIdentity();
  journal.append('effective-identity', effective);
  if (!effective.revisionMatchesRequested) {
    throw new Error(`effective revision ${effective.revision} does not match requested ${options.expectedRevision}`);
  }
  if (!effective.clean) throw new Error(`source worktree is dirty: ${effective.dirtyPaths.join(', ')}`);
  if (effective.kit?.version !== '0.1.42') {
    throw new Error(`effective inference-kit must be 0.1.42, got ${effective.kit?.version ?? 'missing'}`);
  }

  journal.append('heavy-work-start', {
    phase: 'child-launch',
    command: invocation.command,
    args: invocation.args,
  });
  child = spawn(invocation.command, invocation.args, {
    cwd: REPO,
    env: invocation.env,
    stdio: ['ignore', 'pipe', 'pipe', 'pipe'],
  });
  child.stdout.pipe(process.stdout);
  child.stderr.pipe(process.stderr);
  child.stdio[3].on('data', consumeCheckpointData);

  const heartbeat = setInterval(() => {
    journal.append('resource-heartbeat', {
      childPid: child.pid,
      childRssBytes: processRssBytes(child.pid),
      hostFreeMemoryBytes: os.freemem(),
      hostTotalMemoryBytes: os.totalmem(),
      hostLoadAverage: os.loadavg(),
    });
  }, 5000);
  heartbeat.unref();

  const onSignal = (signal) => {
    clearInterval(heartbeat);
    appendTerminal('interrupted', { signal, childPid: child?.pid ?? null });
    if (child && !child.killed) child.kill('SIGTERM');
    process.exitCode = 128;
  };
  process.once('SIGINT', () => onSignal('SIGINT'));
  process.once('SIGTERM', () => onSignal('SIGTERM'));

  child.on('error', (error) => {
    clearInterval(heartbeat);
    appendTerminal('failed', { failurePhase: 'child-launch', error: error.message });
    process.exitCode = 1;
  });
  child.on('close', (code, signal) => {
    clearInterval(heartbeat);
    let childReport = null;
    try {
      childReport = JSON.parse(fs.readFileSync(options.reportPath, 'utf8'));
    } catch {}
    const succeeded = code === 0 && childReport?.ok === true;
    appendTerminal(succeeded ? 'succeeded' : 'failed', {
      childExitCode: code,
      childSignal: signal,
      childReportStatus: childReport?.status ?? null,
      childReportOk: childReport?.ok ?? null,
      lastChildReport: childReport,
    });
    process.exitCode = succeeded ? 0 : 1;
  });
} catch (error) {
  appendTerminal('failed', { failurePhase: 'preflight', error: error.message });
  console.error(`M2 STAGED DISCRIMINANT FAILED [preflight]: ${error.message}`);
  process.exitCode = 1;
}

