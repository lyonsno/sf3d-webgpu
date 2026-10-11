import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import * as writers from './json_report_atomic.mjs';

const write = writers.writeJsonReportAtomicByPhase ?? writers.writeJsonReportAtomic;
const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'sf3d-phase-report-'));
const filename = path.join(directory, 'report.json');
try {
  const phases = Array.from({length: 64}, (_, index) => ({
    phase: 'actual-phase-' + index,
    process: {lastObservation: {transport: {stdout: 'retained raw observation '.repeat(32)}}},
    verdict: 'admitted', additiveUnknown: {index, unicode: '🪑\n\"\\'},
  }));
  const report = {schema: 'test', requested: {route: 'actual', cap: 2147483648},
    phaseObservations: phases, memorySafety: {reason: 'process-footprint-budget'},
    undefinedValue: undefined, tail: [null, -0, NaN, undefined]};
  const expected = JSON.stringify(report, null, 2);
  const largestIndependentValue = Math.max(...Object.entries(report)
    .filter(([key]) => key !== 'phaseObservations')
    .map(([key, value]) => JSON.stringify({[key]: value}, null, 2).length),
    ...phases.map((value, i) => JSON.stringify({[i]: value}, null, 2).length));
  const originalStringify = JSON.stringify;
  let largestSerialization = 0;
  try {
    JSON.stringify = (...args) => {
      const value = originalStringify(...args);
      largestSerialization = Math.max(largestSerialization, value?.length ?? 0);
      return value;
    };
    write(filename, report);
  } finally { JSON.stringify = originalStringify; }
  assert.equal(fs.readFileSync(filename, 'utf8'), expected,
    'all fields, complete phase history, routes and safety failure must survive byte-for-byte');
  assert(largestSerialization <= largestIndependentValue,
    `writer materialized growing history: ${largestSerialization} > ${largestIndependentValue}`);
  assert.equal(typeof writers.writeJsonReportAtomicByPhase, 'function');
  console.log('ok full phase history remains exact without a history-sized serialization');

  for (const phaseObservations of [[], [null, undefined, NaN, 'scalar', {toJSON(key) {return {key};}}]]) {
    const value = {before: 1, phaseObservations, after: {nested: true}};
    write(filename, value);
    assert.equal(fs.readFileSync(filename, 'utf8'), JSON.stringify(value, null, 2));
  }
  console.log('ok empty, scalar and index-sensitive toJSON entries retain JSON semantics');

  const published = fs.readFileSync(filename, 'utf8');
  const circular = {}; circular.self = circular;
  assert.throws(() => write(filename, {phaseObservations: [{raw: circular}]}), /circular/i);
  assert.equal(fs.readFileSync(filename, 'utf8'), published);
  assert.deepEqual(fs.readdirSync(directory), ['report.json']);
  console.log('ok serialization failure cannot replace the last complete report');

  const originalWrite = fs.writeSync;
  try {
    let calls = 0;
    fs.writeSync = (...args) => {
      if (++calls > 1) throw Error('injected disk failure');
      return originalWrite(...args);
    };
    assert.throws(() => write(filename, {phaseObservations: phases}), /injected disk failure/);
  } finally {fs.writeSync = originalWrite;}
  assert.equal(fs.readFileSync(filename, 'utf8'), published);
  assert.deepEqual(fs.readdirSync(directory), ['report.json']);
  console.log('ok partial write failure preserves published evidence and removes owned temporary');

  try {
    fs.writeSync = (fd, bytes, offset, length) => originalWrite(fd, bytes, offset, Math.min(length, 3));
    write(filename, report);
  } finally {fs.writeSync = originalWrite;}
  assert.equal(fs.readFileSync(filename, 'utf8'), expected);
  console.log('ok short UTF-8 writes cannot silently truncate evidence');

  try {
    fs.writeSync = () => 0;
    assert.throws(() => write(filename, report), /invalid progress/);
  } finally {fs.writeSync = originalWrite;}
  assert.equal(fs.readFileSync(filename, 'utf8'), expected);
  assert.deepEqual(fs.readdirSync(directory), ['report.json']);
  const customArray = []; customArray.toJSON = () => [];
  assert.throws(() => write(filename, {phaseObservations: customArray}), /phase data array/);
  assert.throws(() => write(filename, {phaseObservations: null}), /phase data array/);
  console.log('ok unwriteable or unsupported data cannot silently become a complete report');

  const originalRename = fs.renameSync;
  try {
    fs.renameSync = () => {throw Error('injected publication failure');};
    assert.throws(() => write(filename, report), /injected publication failure/);
  } finally {fs.renameSync = originalRename;}
  assert.equal(fs.readFileSync(filename, 'utf8'), expected);
  assert.deepEqual(fs.readdirSync(directory), ['report.json']);
  console.log('ok failed atomic replacement leaves complete previous report intact');
} finally {fs.rmSync(directory, {recursive: true, force: true});}
