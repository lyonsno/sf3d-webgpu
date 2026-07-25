/**
 * Worker failure-lifecycle contract for callWorker (worker_call.js).
 *
 * Fail-first: proves that EVERY worker failure path rejects loudly and none
 * hangs or silently returns bad output. Uses a fake EventTarget-based worker so
 * no real Web Worker / browser is needed.
 *
 * Run: node tools/test_worker_call_lifecycle.mjs
 */
import assert from 'node:assert/strict';
import { callWorker } from '../src/lib/worker_call.js';

// Minimal fake Worker: EventTarget with postMessage + a scripted reply behavior.
function makeFakeWorker(behavior) {
  const listeners = { message: [], error: [], messageerror: [] };
  return {
    addEventListener: (t, fn) => listeners[t].push(fn),
    removeEventListener: (t, fn) => { listeners[t] = listeners[t].filter(f => f !== fn); },
    _emit: (type, data) => { for (const fn of [...listeners[type]]) fn(data); },
    _listenerCount: () => listeners.message.length + listeners.error.length + listeners.messageerror.length,
    postMessage(msg) { behavior(this, msg); },
  };
}

const okResult = (data) => data.value;

// 1. Success — resolves with mapped value, listeners cleaned up.
{
  const w = makeFakeWorker((self, msg) => self._emit('message', { data: { id: msg.id, ok: true, value: 42 } }));
  const v = await callWorker(w, { id: 'a' }, [], { onResult: okResult });
  assert.equal(v, 42);
  assert.equal(w._listenerCount(), 0, 'listeners must be removed after success');
  console.log('ok  success resolves + cleans up listeners');
}

// 2. Posted error { ok:false } — rejects with the error, no hang.
{
  const w = makeFakeWorker((self, msg) => self._emit('message', { data: { id: msg.id, ok: false, error: 'boom' } }));
  await assert.rejects(() => callWorker(w, { id: 'b' }, [], { onResult: okResult }), /worker failed: boom/);
  assert.equal(w._listenerCount(), 0);
  console.log('ok  posted error rejects');
}

// 3. Malformed output (onResult throws) — rejects as invalid, does NOT resolve.
{
  const w = makeFakeWorker((self, msg) => self._emit('message', { data: { id: msg.id, ok: true, value: 'wrong' } }));
  await assert.rejects(
    () => callWorker(w, { id: 'c' }, [], { onResult: (d) => { if (d.value !== 42) throw new Error('bad shape'); return d.value; } }),
    /worker output invalid: bad shape/);
  console.log('ok  malformed output rejects (no silent bad result)');
}

// 4. Top-level worker exception (onerror) — rejects, never hangs.
{
  const w = makeFakeWorker((self) => self._emit('error', { message: 'top-level throw' }));
  await assert.rejects(() => callWorker(w, { id: 'd' }, [], { onResult: okResult }), /worker error: top-level throw/);
  assert.equal(w._listenerCount(), 0);
  console.log('ok  onerror rejects');
}

// 5. messageerror (undeserializable reply) — rejects.
{
  const w = makeFakeWorker((self) => self._emit('messageerror', {}));
  await assert.rejects(() => callWorker(w, { id: 'e' }, [], { onResult: okResult }), /messageerror/);
  console.log('ok  messageerror rejects');
}

// 6. Wedged worker (never replies) — rejects on timeout, never hangs.
{
  const w = makeFakeWorker(() => { /* silence */ });
  const t0 = Date.now();
  await assert.rejects(() => callWorker(w, { id: 'f' }, [], { onResult: okResult, timeoutMs: 60 }), /timed out after 60ms/);
  assert.ok(Date.now() - t0 >= 55, 'must wait the timeout before rejecting');
  assert.equal(w._listenerCount(), 0, 'timeout must clean up listeners');
  console.log('ok  wedged worker times out (no infinite hang)');
}

// 7. postMessage throws (e.g. DataCloneError) — rejects synchronously-safe.
{
  const w = makeFakeWorker(() => { throw new Error('DataCloneError'); });
  await assert.rejects(() => callWorker(w, { id: 'g' }, [], { onResult: okResult }), /postMessage failed: DataCloneError/);
  console.log('ok  postMessage throw rejects');
}

// 8. Stray reply for a different id is ignored (no cross-talk), then real reply resolves.
{
  const w = makeFakeWorker((self, msg) => {
    self._emit('message', { data: { id: 'OTHER', ok: true, value: 999 } }); // not ours
    self._emit('message', { data: { id: msg.id, ok: true, value: 7 } });    // ours
  });
  const v = await callWorker(w, { id: 'h' }, [], { onResult: okResult });
  assert.equal(v, 7, 'must ignore stray ids and take our own reply');
  console.log('ok  ignores stray-id replies');
}

// 9. Late/duplicate reply after settle does not double-settle or throw.
{
  let self2 = null;
  const w = makeFakeWorker((self, msg) => { self2 = self; self._emit('message', { data: { id: msg.id, ok: true, value: 1 } }); });
  const v = await callWorker(w, { id: 'i' }, [], { onResult: okResult });
  assert.equal(v, 1);
  // Emit a duplicate after settle — listeners removed, so this is a no-op.
  assert.doesNotThrow(() => self2._emit('message', { data: { id: 'i', ok: true, value: 2 } }));
  console.log('ok  duplicate post-settle reply is inert');
}

console.log('\nALL WORKER LIFECYCLE CHECKS PASSED');
