import assert from 'node:assert/strict';
import { submitPreviewAndRetire } from '../src/lib/preview_submission.js';

function fixture({ submitError = null, callbackError = null, finishError = null,
  secondFenceError = null } = {}) {
  const events = [];
  let pendingCallbackWork = false;
  let fences = 0;
  const device = { queue: {
    submit() {
      events.push('submit');
      if (submitError) throw submitError;
    },
    async onSubmittedWorkDone() {
      events.push('fence');
      fences++;
      pendingCallbackWork = false;
      if (fences === 2 && secondFenceError) throw secondFenceError;
    },
  } };
  const encoder = { finish() {
    events.push('finish');
    if (finishError) throw finishError;
    return {};
  } };
  const allocations = [{ buffer: { destroy() {
    assert.equal(pendingCallbackWork, false);
    events.push('destroy');
  } } }];
  const consume = async () => {
    events.push('callback');
    pendingCallbackWork = true;
    if (callbackError) throw callbackError;
  };
  return { device, encoder, allocations, consume, events };
}

const normal = fixture();
await submitPreviewAndRetire(normal.device, normal.encoder, normal.allocations, normal.consume);
assert.deepEqual(normal.events, ['finish', 'submit', 'fence', 'callback', 'fence', 'destroy']);

for (const fault of ['submitError', 'callbackError', 'finishError']) {
  const error = new Error(fault);
  const failed = fixture({ [fault]: error });
  await assert.rejects(
    submitPreviewAndRetire(failed.device, failed.encoder, failed.allocations, failed.consume),
    error,
  );
  assert.equal(failed.events.at(-1), 'destroy', `${fault} must retire preview allocations`);
}
const originalFailure = new Error('callback failed');
const settlementFailure = new Error('second fence failed');
const competing = fixture({ callbackError: originalFailure, secondFenceError: settlementFailure });
await assert.rejects(
  submitPreviewAndRetire(competing.device, competing.encoder, competing.allocations, competing.consume),
  error => error === originalFailure && error.previewSettlementError === settlementFailure,
);
assert.equal(competing.events.at(-1), 'destroy');
console.log('preview allocation retirement survives submit, callback, and finish failure');
