#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  createParentPhaseJournal,
  replayParentPhaseJournal,
  resolveDurableArtifactPath,
} from './parent_phase_journal.mjs';

const parent = path.join(os.homedir(), '.local/state/sf3d');
fs.mkdirSync(parent, { recursive: true });
const root = fs.mkdtempSync(path.join(parent, 'test-phase-journal-'));
const journalPath = path.join(root, 'episode.jsonl');

try {
  assert.throws(() => resolveDurableArtifactPath('/private/tmp/volatile.jsonl'), /volatile/i);
  const journal = createParentPhaseJournal({
    journalPath,
    invocationId: 'parent-journal-contract',
    requested: { routeId: 'sf3d.image-to-mesh.webgpu-local.v0', mode: 'setup-only' },
  });
  journal.append('effective-identity', { revision: 'a'.repeat(40), package: '0.1.52' });
  journal.append('phase-entered', { phase: 'dinov2-tokenizer' });
  journal.append('phase-completed', { phase: 'dinov2-tokenizer', tokenShape: { count: 9216, width: 1024 } });
  journal.append('terminal', { status: 'succeeded' });
  journal.close();

  const replay = replayParentPhaseJournal(journalPath);
  assert.equal(replay.integrityOk, true);
  assert.equal(replay.status, 'succeeded');
  assert.equal(replay.lastEnteredPhase, 'dinov2-tokenizer');
  assert.equal(replay.lastCompletedPhase, 'dinov2-tokenizer');
  assert.equal(replay.events.length, 5);

  const interruptedPath = path.join(root, 'interrupted.jsonl');
  const interruptedWriter = createParentPhaseJournal({
    journalPath: interruptedPath,
    invocationId: 'interrupted-contract',
    requested: { routeId: 'sf3d.image-to-mesh.webgpu-local.v0' },
  });
  interruptedWriter.append('effective-identity', { revision: 'b'.repeat(40) });
  interruptedWriter.append('phase-entered', { phase: 'dinov2-tokenizer' });
  interruptedWriter.close();
  const partialPath = path.join(root, 'partial.jsonl');
  fs.copyFileSync(interruptedPath, partialPath);
  fs.appendFileSync(partialPath, '{"schema":"sf3d.parent-phase-journal-event.v0"');
  const partialReplay = replayParentPhaseJournal(partialPath);
  assert.equal(partialReplay.integrityOk, true);
  assert.equal(partialReplay.status, 'interrupted');
  assert.equal(partialReplay.hasPartialTail, true);
  assert.equal(partialReplay.lastEnteredPhase, 'dinov2-tokenizer');
  assert.equal(partialReplay.lastCompletedPhase, null);

  const blankPath = path.join(root, 'blank-line.jsonl');
  const intactRows = fs.readFileSync(journalPath, 'utf8').trimEnd().split('\n');
  intactRows.splice(2, 0, '');
  fs.writeFileSync(blankPath, `${intactRows.join('\n')}\n`);
  assert.throws(() => replayParentPhaseJournal(blankPath), /blank|integrity/i);

  const tamperedPath = path.join(root, 'tampered.jsonl');
  const rows = fs.readFileSync(journalPath, 'utf8').trimEnd().split('\n');
  const changed = JSON.parse(rows[2]);
  changed.payload.phase = 'not-dinov2';
  rows[2] = JSON.stringify(changed);
  fs.writeFileSync(tamperedPath, `${rows.join('\n')}\n`);
  assert.throws(() => replayParentPhaseJournal(tamperedPath), /hash|integrity/i);

  console.log('parent phase journal contract passed');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
