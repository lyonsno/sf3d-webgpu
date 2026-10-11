import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {startProcessMemory} from './process_memory_guard.mjs';

const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'sf3d-current-snapshot-'));
let monitor;
try {
  let sequence = 0;
  const row = () => ({runId: 'current', rootPid: 42, atUnixMs: Date.now(), status: 'observed',
    processes: [42, 100 + sequence++].map(pid => ({pid, processStartAbstime: pid,
      physicalFootprintBytes: 50, kernelLifetimePeakPhysicalFootprintBytes: 60})),
    sampledAggregatePhysicalFootprintBytes: 100});
  monitor = await startProcessMemory({rootPid: 42, runId: 'current', rawPath: path.join(directory, 'raw.jsonl'),
    periodMs: 1000, probe: async () => row(), maxFootprintBytes: 150,
    onUnsafe: () => assert.fail('valid sample must not stop')});
  const originalClone = globalThis.structuredClone;
  let clonedLifetimeHistory = false, current;
  try {
    globalThis.structuredClone = value => {
      clonedLifetimeHistory ||= Object.hasOwn(value, 'processes') || Object.hasOwn(value, 'unavailableProcessObservations');
      return originalClone(value);
    };
    current = await monitor.sample({fresh: true, includeLifetimeHistory: false});
  } finally {globalThis.structuredClone = originalClone;}
  assert.equal(clonedLifetimeHistory, false, 'live admission must not first clone discarded lifetime history');
  assert.equal(Object.hasOwn(current, 'processes'), false);
  assert.equal(current.coverage, 'sampled-owned-process-tree');
  assert.equal(current.lastObservation.runId, 'current');
  assert.equal(current.lastObservation.sampledAggregatePhysicalFootprintBytes, 100);
  assert.equal(current.freshness.observationIndex, current.sampleCount);
  assert.equal(current.freshness.probeAtUnixMs, current.lastObservation.atUnixMs);
  current.lastObservation.runId = 'caller mutation';
  const complete = await monitor.sample();
  assert.equal(complete.lastObservation.runId, 'current');
  assert.equal(Object.keys(complete.processes).length, sequence + 1);
  await assert.rejects(monitor.sample({includeLifetimeHistory: 'false'}), /boolean/);
  const terminal = await monitor.stop(); monitor = null;
  assert.equal(Object.keys(terminal.processes).length, sequence + 1);
  assert.equal((await fs.readFile(terminal.rawPath, 'utf8')).trim().split('\n').length, terminal.sampleCount);
  console.log('ok fresh admission omits no current authority; complete raw and terminal history stay uncapped');

  let samples = 0, stopped = false;
  monitor = await startProcessMemory({rootPid: 42, runId: 'current', rawPath: path.join(directory, 'unsafe.jsonl'),
    probe: async () => {const value = row(); if (samples++) {
      value.processes[0].physicalFootprintBytes = 150; value.sampledAggregatePhysicalFootprintBytes = 200;
    } return value;}, maxFootprintBytes: 150, onUnsafe: () => {stopped = true;}});
  await assert.rejects(monitor.sample({fresh: true, includeLifetimeHistory: false}), /budget exceeded/);
  assert.equal(stopped, true);
  assert.equal((await monitor.stop()).status, 'budget-refused'); monitor = null;
  console.log('ok current-only snapshots cannot bypass periodic or fresh process-budget intervention');
} finally {if (monitor) await monitor.stop(); await fs.rm(directory, {recursive: true, force: true});}
