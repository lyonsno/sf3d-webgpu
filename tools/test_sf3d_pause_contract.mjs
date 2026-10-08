#!/usr/bin/env node
import assert from 'node:assert/strict';
import { createWebGpuInferenceControl } from '@kaminos/webgpu-inference-kit/core';
import { withForegroundScope } from '../src/lib/foreground_scope.js';
import { finishProducerRunWithEvidence, runControlledPipeline } from '../src/lib/sf3d_producer.js';

const phases = [];
const control = createWebGpuInferenceControl({
  queue: { onSubmittedWorkDone: async () => {} },
  withForeground: async (phase, work) => {
    phases.push(phase);
    return work();
  },
});
assert.equal((await control.pause()).status, 'paused');
const work = [];
const pending = withForegroundScope({
  inferenceControl: control,
  withForeground: async (phase, duty) => {
    phases.push(phase);
    return duty();
  },
}, 'uv-unwrap', () => { work.push('executed'); return 17; });
await Promise.resolve();
assert.deepEqual(work, [], 'CPU/worker leaf must not start while inference is parked');
assert.equal((await control.resume()).status, 'running');
assert.equal(await pending, 17);
assert.deepEqual(work, ['executed']);
assert.deepEqual(phases, ['inference-paused', 'uv-unwrap']);
await control.close();

const stop = new AbortController();
const stoppedControl = createWebGpuInferenceControl({
  queue: { onSubmittedWorkDone: async () => {} },
  signal: stop.signal,
});
await stoppedControl.pause();
let started = false;
const stoppedDuty = withForegroundScope({ inferenceControl: stoppedControl },
  'marching-tet', () => { started = true; });
stop.abort(new Error('operator stop'));
await assert.rejects(stoppedDuty, /operator stop/);
assert.equal(started, false, 'Stop while parked must not enter the queued leaf');
assert.equal((await stoppedControl.close()).status, 'cancelled');

let active = null;
let enter;
let release;
const entered = new Promise(resolve => { enter = resolve; });
const gate = new Promise(resolve => { release = resolve; });
const admitted = [];
const run = runControlledPipeline({
  queue: { onSubmittedWorkDone: async () => {} },
  options: { marker: 'same-invocation' },
  withForeground: async (_phase, duty) => duty(),
  onControl: control => { active = control; },
  execute: async options => {
    assert.equal(options.marker, 'same-invocation');
    enter();
    await gate;
    return options.inferenceControl.runDuty(() => { admitted.push('model duty'); return 23; });
  },
});
await entered;
assert.equal((await active.pause()).status, 'paused');
release();
await Promise.resolve();
assert.deepEqual(admitted, [], 'the pending model duty stays parked');
assert.equal((await active.resume()).status, 'running');
assert.equal(await run, 23, 'resume finishes the original invocation');
assert.deepEqual(admitted, ['model duty']);
assert.equal(active, null, 'invocation control closes before foreground finish');

const finalStop = new AbortController();
let terminalStopError;
await assert.rejects(runControlledPipeline({
  queue: { onSubmittedWorkDone: async () => {} },
  signal: finalStop.signal,
  options: {},
  execute: ({ inferenceControl }) => inferenceControl.runDuty(() => {
    finalStop.abort(new Error('Stop during final model duty'));
    return 31;
  }),
}), error => {
  terminalStopError = error;
  return error.name === 'AbortError' && /Stop during final model duty/.test(error.message);
},
'Stop at the last duty must not return a successful GLB/result');
await assert.rejects(finishProducerRunWithEvidence({
  prepared: { release: async () => ({ completed: true }) },
  pipelineFailed: true,
  pipelineError: terminalStopError,
  runId: 'stopped-run',
  lastProgress: 'final duty',
  startedAtMs: performance.now(),
  deviceInjected: true,
  commit: 'test',
}), error => error.name === 'AbortError'
  && error.sf3dRun?.runId === 'stopped-run'
  && error.sf3dRun?.inferenceCompleted === false
  && error.sf3dRun?.lastProgress === 'final duty',
'the stopped run must preserve producer failure evidence');
console.log('SF3D PAUSE CONTRACT PASSED');
