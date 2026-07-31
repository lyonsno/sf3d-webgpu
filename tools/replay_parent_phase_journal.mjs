#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { replayParentPhaseJournal } from './parent_phase_journal.mjs';
import { writeJsonReportAtomic } from './json_report_atomic.mjs';

function option(flag) {
  const indexes = process.argv
    .map((value, index) => (value === flag ? index : -1))
    .filter(index => index >= 0);
  if (indexes.length > 1) throw new Error(`${flag} may be provided only once`);
  if (!indexes.length) return null;
  const value = process.argv[indexes[0] + 1];
  if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value`);
  return value;
}

const known = new Set(['--journal', '--report']);
for (const argument of process.argv.slice(2)) {
  if (argument.startsWith('--') && !known.has(argument)) throw new Error(`unknown option ${argument}`);
}

const journalArgument = option('--journal');
if (!journalArgument) throw new Error('--journal is required');
const journalPath = path.resolve(journalArgument);
const replay = replayParentPhaseJournal(journalPath);
const reportArgument = option('--report');
if (reportArgument) {
  const reportPath = path.resolve(reportArgument);
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  writeJsonReportAtomic(reportPath, replay);
}
process.stdout.write(`${JSON.stringify(replay, null, 2)}\n`);

