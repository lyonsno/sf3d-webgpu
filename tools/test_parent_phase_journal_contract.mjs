import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import {
  createParentPhaseJournal,
  replayParentPhaseJournal,
} from './parent_phase_journal.mjs';

const repo = path.resolve(new URL('..', import.meta.url).pathname);
const testRoot = path.join(
  os.homedir(),
  '.local/state/sf3d/test-parent-journal-contract',
);
const journalPath = path.join(testRoot, 'episode.jsonl');

fs.rmSync(testRoot, { recursive: true, force: true });
fs.mkdirSync(testRoot, { recursive: true });

try {
  assert.throws(
    () => createParentPhaseJournal({
      journalPath: '/private/tmp/sf3d-unsafe.jsonl',
      invocationId: 'volatile-path',
      requested: { mode: 'setup-only' },
    }),
    /volatile/i,
    'production journals must reject /tmp and /private/tmp',
  );

  const journal = createParentPhaseJournal({
    journalPath,
    invocationId: 'contract-episode',
    requested: {
      revision: 'a'.repeat(40),
      mode: 'single-arm',
      arm: 'arena-plus-worker',
      routeId: 'sf3d.image-to-mesh.webgpu-local.v0',
    },
  });

  assert.throws(
    () => journal.append('heavy-work-start', { phase: 'browser-launch' }),
    /effective identity/i,
    'heavy work cannot begin before requested/effective identity is durable',
  );

  journal.append('effective-identity', {
    revision: 'a'.repeat(40),
    mode: 'single-arm',
    arm: 'arena-plus-worker',
    routeId: 'sf3d.image-to-mesh.webgpu-local.v0',
    browserPath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  });
  journal.append('heavy-work-start', { phase: 'browser-launch' });
  journal.append('phase-checkpoint', {
    phase: 'runtime-setup',
    boundary: 'browser-ready',
    trustworthy: true,
  });
  journal.append('resource-heartbeat', {
    childPid: 123,
    hostFreeMemoryBytes: 456,
  });
  journal.close();

  const interrupted = replayParentPhaseJournal(journalPath);
  assert.equal(interrupted.integrityOk, true);
  assert.equal(interrupted.status, 'interrupted');
  assert.equal(interrupted.lastTrustworthyBoundary, 'browser-ready');
  assert.equal(interrupted.events.length, 5);
  assert.deepEqual(
    interrupted.events.map(event => event.sequence),
    [0, 1, 2, 3, 4],
  );

  assert.throws(
    () => createParentPhaseJournal({
      journalPath,
      invocationId: 'stale-overwrite',
      requested: { mode: 'setup-only' },
    }),
    /already exists/i,
    'a stale prior journal must never be overwritten',
  );

  const abruptPath = path.join(testRoot, 'abrupt.jsonl');
  const child = spawnSync(
    process.execPath,
    [
      '--input-type=module',
      '--eval',
      `import { createParentPhaseJournal } from ${JSON.stringify(new URL('./parent_phase_journal.mjs', import.meta.url).href)};
       const journal = createParentPhaseJournal({ journalPath: ${JSON.stringify(abruptPath)}, invocationId: 'abrupt', requested: { mode: 'setup-only' } });
       journal.append('effective-identity', { revision: '${'b'.repeat(40)}', mode: 'setup-only', arm: null, routeId: 'sf3d.image-to-mesh.webgpu-local.v0', browserPath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
       journal.append('heavy-work-start', { phase: 'browser-launch' });
       journal.append('phase-checkpoint', { phase: 'runtime-setup', boundary: 'weights-loading', trustworthy: true });
       process.exit(91);`,
    ],
    { cwd: repo, encoding: 'utf8' },
  );
  assert.equal(child.status, 91, child.stderr);
  const abrupt = replayParentPhaseJournal(abruptPath);
  assert.equal(abrupt.integrityOk, true);
  assert.equal(abrupt.status, 'interrupted');
  assert.equal(abrupt.lastTrustworthyBoundary, 'weights-loading');

  const corruptedLines = fs.readFileSync(abruptPath, 'utf8').trimEnd().split('\n');
  const corrupted = JSON.parse(corruptedLines[1]);
  corrupted.payload.revision = 'c'.repeat(40);
  corruptedLines[1] = JSON.stringify(corrupted);
  const corruptPath = path.join(testRoot, 'corrupt.jsonl');
  fs.writeFileSync(corruptPath, `${corruptedLines.join('\n')}\n`);
  assert.throws(
    () => replayParentPhaseJournal(corruptPath),
    /hash|integrity/i,
    'journal replay must reject modified checkpoints',
  );

  console.log('parent phase journal contract passed');
} finally {
  fs.rmSync(testRoot, { recursive: true, force: true });
}
