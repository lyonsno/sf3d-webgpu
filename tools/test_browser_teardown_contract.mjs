import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import {
  closeOwnedBrowser,
  probeOwnedProcess,
  stopOwnedChildProcess,
  waitForOwnedProcessExit,
} from './browser_teardown.mjs';

function fakeClock() {
  let value = 0;
  return () => value += 5;
}

function error(code, message) {
  return Object.assign(new Error(message), { code });
}

assert.deepEqual(probeOwnedProcess(11, { killFn: () => {} }), { state: 'present', error: null });
assert.deepEqual(probeOwnedProcess(12, { killFn: () => { throw error('ESRCH', 'gone'); } }), { state: 'absent', error: null });
assert.deepEqual(probeOwnedProcess(13, { killFn: () => { throw error('EPERM', 'denied'); } }), { state: 'unknown', error: 'EPERM: denied' });
assert.equal(probeOwnedProcess(null).state, 'unavailable');

const immediateTimer = callback => { queueMicrotask(callback); return 1; };
const neverExited = new EventEmitter();
neverExited.pid = 14;
neverExited.exitCode = null;
neverExited.signalCode = null;
const waited = await waitForOwnedProcessExit(neverExited, {
  graceMs: 1,
  probeProcessFn: () => ({ state: 'present', error: null }),
  setTimeoutFn: immediateTimer,
  clearTimeoutFn: () => {},
});
assert.equal(waited.exited, false);
assert.equal(waited.outcome, 'grace-expired');

const noChild = await stopOwnedChildProcess(null, { now: fakeClock() });
assert.equal(noChild.ok, true);
assert.equal(noChild.outcome, 'not-started');

const missingIdentity = await stopOwnedChildProcess({ pid: null }, { now: fakeClock() });
assert.equal(missingIdentity.ok, false);
assert.equal(missingIdentity.outcome, 'identity-unavailable');

const vite = new EventEmitter();
vite.pid = 15;
vite.exitCode = null;
vite.signalCode = null;
const viteSignals = [];
vite.kill = signal => { viteSignals.push(signal); return true; };
let waits = 0;
const viteStopped = await stopOwnedChildProcess(vite, {
  probeProcessFn: () => ({ state: 'present', error: null }),
  waitForExitFn: async () => ({
    exited: ++waits === 2,
    outcome: waits === 1 ? 'grace-expired' : 'exit-event',
    exitCode: null,
    signalCode: waits === 2 ? 'SIGKILL' : null,
    finalProbe: waits === 2 ? { state: 'absent', error: null } : { state: 'present', error: null },
  }),
  now: fakeClock(),
});
assert.equal(viteStopped.ok, true);
assert.equal(viteStopped.outcome, 'exited-after-sigkill');
assert.deepEqual(viteSignals, ['SIGTERM', 'SIGKILL']);

const signalFailure = new EventEmitter();
signalFailure.pid = 16;
signalFailure.exitCode = null;
signalFailure.signalCode = null;
signalFailure.kill = () => { throw new Error('signal rejected'); };
const viteSignalFailed = await stopOwnedChildProcess(signalFailure, {
  probeProcessFn: () => ({ state: 'unknown', error: 'EPERM: denied' }),
  now: fakeClock(),
});
assert.equal(viteSignalFailed.ok, false);
assert.equal(viteSignalFailed.outcome, 'sigterm-failed');
assert.equal(viteSignalFailed.finalProbe.state, 'unknown');

const stubborn = new EventEmitter();
stubborn.pid = 17;
stubborn.exitCode = null;
stubborn.signalCode = null;
stubborn.kill = () => true;
const stubbornResult = await stopOwnedChildProcess(stubborn, {
  probeProcessFn: () => ({ state: 'present', error: null }),
  waitForExitFn: async () => ({
    exited: false,
    outcome: 'grace-expired',
    exitCode: null,
    signalCode: null,
    finalProbe: { state: 'present', error: null },
  }),
  now: fakeClock(),
});
assert.equal(stubbornResult.ok, false);
assert.equal(stubbornResult.outcome, 'sigkill-grace-expired');

const absentBrowser = await closeOwnedBrowser(null, { now: fakeClock() });
assert.equal(absentBrowser.ok, true);
assert.equal(absentBrowser.outcome, 'not-started');

const identityMissing = await closeOwnedBrowser({
  process: () => null,
  close: async () => {},
  disconnect: () => {},
}, { now: fakeClock() });
assert.equal(identityMissing.ok, false);
assert.equal(identityMissing.outcome, 'process-identity-unavailable');

let deadDisconnected = 0;
let deadCloseCalls = 0;
const alreadyExited = await closeOwnedBrowser({
  process: () => ({ pid: 18 }),
  disconnect: () => { deadDisconnected += 1; },
  close: async () => { deadCloseCalls += 1; },
}, {
  probeProcessFn: () => ({ state: 'absent', error: null }),
  now: fakeClock(),
});
assert.equal(alreadyExited.ok, true);
assert.equal(alreadyExited.outcome, 'already-exited-disconnected');
assert.equal(deadDisconnected, 1);
assert.equal(deadCloseCalls, 0);

const browserChild = new EventEmitter();
browserChild.pid = 19;
browserChild.exitCode = null;
browserChild.signalCode = null;
const browserSignals = [];
browserChild.kill = signal => { browserSignals.push(signal); return true; };
const hungBrowser = await closeOwnedBrowser({
  process: () => browserChild,
  disconnect: () => {},
  close: () => new Promise(() => {}),
}, {
  probeProcessFn: () => ({ state: 'unknown', error: 'EPERM: denied' }),
  waitForExitFn: async () => ({
    exited: true,
    outcome: 'exit-event',
    exitCode: null,
    signalCode: 'SIGTERM',
    finalProbe: { state: 'absent', error: null },
  }),
  setTimeoutFn: immediateTimer,
  clearTimeoutFn: () => {},
  now: fakeClock(),
});
assert.equal(hungBrowser.ok, true);
assert.equal(hungBrowser.outcome, 'grace-expired-process-exited');
assert.equal(hungBrowser.initialProbe.state, 'unknown');
assert.deepEqual(browserSignals, ['SIGTERM']);

const unkillableBrowserChild = new EventEmitter();
unkillableBrowserChild.pid = 20;
unkillableBrowserChild.exitCode = null;
unkillableBrowserChild.signalCode = null;
unkillableBrowserChild.kill = () => { throw new Error('signal rejected'); };
const unkillableBrowser = await closeOwnedBrowser({
  process: () => unkillableBrowserChild,
  disconnect: () => {},
  close: async () => { throw new Error('transport closed'); },
}, {
  probeProcessFn: () => ({ state: 'unknown', error: 'EPERM: denied' }),
  now: fakeClock(),
});
assert.equal(unkillableBrowser.ok, false);
assert.equal(unkillableBrowser.outcome, 'close-failed-teardown-failed');
assert.equal(unkillableBrowser.processTeardown.ok, false);

console.log('owned process teardown contract passed');
