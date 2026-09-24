#!/usr/bin/env node
import path from 'node:path';
import { replayParentPhaseJournal } from './parent_phase_journal.mjs';

const args = process.argv.slice(2);
const index = args.indexOf('--journal');
if (index < 0 || index !== args.lastIndexOf('--journal')) throw new Error('--journal must be provided once');
const value = args[index + 1];
if (!value || value.startsWith('--') || args.some((arg, i) => i !== index && i !== index + 1)) {
  throw new Error('usage: node tools/replay_parent_phase_journal.mjs --journal PATH');
}
const replay = replayParentPhaseJournal(path.resolve(value));
process.stdout.write(`${JSON.stringify(replay, null, 2)}\n`);
