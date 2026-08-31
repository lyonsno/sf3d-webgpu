import assert from 'node:assert/strict';
import { closeOwnedBrowser, DEFAULT_TEARDOWN_GRACE_MS } from './browser_teardown.mjs';

function fakeClock() {
  let value = 0;
  return () => value += 5;
}

const absent = await closeOwnedBrowser(null, { now: fakeClock() });
assert.equal(absent.ok, true);
assert.equal(absent.outcome, 'not-started');

let deadDisconnected = 0;
let deadCloseCalls = 0;
const alreadyExited = await closeOwnedBrowser({
  process: () => ({ pid: 101 }),
  disconnect: () => { deadDisconnected += 1; },
  close: async () => { deadCloseCalls += 1; },
}, { processExistsFn: () => false, now: fakeClock() });
assert.equal(alreadyExited.ok, true);
assert.equal(alreadyExited.outcome, 'already-exited-disconnected');
assert.equal(deadDisconnected, 1);
assert.equal(deadCloseCalls, 0);

const deadDisconnectFailed = await closeOwnedBrowser({
  process: () => ({ pid: 105 }),
  disconnect: () => { throw new Error('disconnect rejected'); },
  close: () => assert.fail('already-exited browser must not issue CDP close'),
}, { processExistsFn: () => false, now: fakeClock() });
assert.equal(deadDisconnectFailed.ok, false);
assert.equal(deadDisconnectFailed.outcome, 'already-exited-disconnect-failed');
assert.equal(deadDisconnectFailed.disconnectError, 'disconnect rejected');

let normalCloseCalls = 0;
const normal = await closeOwnedBrowser({
  process: () => ({ pid: 102 }),
  disconnect: () => assert.fail('normal close must not disconnect'),
  close: async () => { normalCloseCalls += 1; },
}, { processExistsFn: () => true, now: fakeClock() });
assert.equal(normal.ok, true);
assert.equal(normal.outcome, 'closed');
assert.equal(normalCloseCalls, 1);

let hungDisconnected = 0;
const hungSignals = [];
const hung = await closeOwnedBrowser({
  process: () => ({ pid: 103, kill: signal => { hungSignals.push(signal); return true; } }),
  disconnect: () => { hungDisconnected += 1; },
  close: () => new Promise(() => {}),
}, {
  processExistsFn: () => true,
  setTimeoutFn: callback => { queueMicrotask(callback); return 1; },
  clearTimeoutFn: () => {},
  now: fakeClock(),
});
assert.equal(hung.ok, true);
assert.equal(hung.outcome, 'grace-expired-recovered');
assert.equal(hung.graceMs, DEFAULT_TEARDOWN_GRACE_MS);
assert.equal(hungDisconnected, 1);
assert.deepEqual(hungSignals, ['SIGTERM']);

const rejected = await closeOwnedBrowser({
  process: () => ({ pid: 104, kill: () => { throw new Error('signal rejected'); } }),
  disconnect: () => {},
  close: async () => { throw new Error('transport closed'); },
}, { processExistsFn: () => true, now: fakeClock() });
assert.equal(rejected.ok, false);
assert.equal(rejected.outcome, 'close-failed-termination-failed');
assert.equal(rejected.closeError, 'transport closed');
assert.equal(rejected.terminationError, 'signal rejected');

console.log('owned browser teardown contract passed');
