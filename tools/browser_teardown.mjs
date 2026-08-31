const DEFAULT_TEARDOWN_GRACE_MS = 10_000;

function processExists(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function disconnect(browser) {
  try {
    browser.disconnect();
    return { disconnected: true, disconnectError: null };
  } catch (error) {
    return { disconnected: false, disconnectError: error.message };
  }
}

export async function closeOwnedBrowser(browser, {
  graceMs = DEFAULT_TEARDOWN_GRACE_MS,
  processExistsFn = processExists,
  setTimeoutFn = setTimeout,
  clearTimeoutFn = clearTimeout,
  now = () => performance.now(),
} = {}) {
  const startedAt = now();
  if (!browser) {
    return {
      ok: true,
      outcome: 'not-started',
      browserPid: null,
      graceMs,
      elapsedMs: now() - startedAt,
    };
  }

  let browserProcess = null;
  try { browserProcess = browser.process?.() ?? null; } catch {}
  const browserPid = Number.isSafeInteger(browserProcess?.pid) ? browserProcess.pid : null;
  if (browserPid !== null && !processExistsFn(browserPid)) {
    const disconnectResult = disconnect(browser);
    return {
      ok: disconnectResult.disconnected,
      outcome: disconnectResult.disconnected
        ? 'already-exited-disconnected'
        : 'already-exited-disconnect-failed',
      browserPid,
      graceMs,
      ...disconnectResult,
      closeError: null,
      terminationRequested: false,
      terminationError: null,
      elapsedMs: now() - startedAt,
    };
  }

  let timeoutId = null;
  const closeAttempt = Promise.resolve()
    .then(() => browser.close())
    .then(
      () => ({ outcome: 'closed', closeError: null }),
      error => ({ outcome: 'close-failed', closeError: error.message }),
    );
  const deadline = new Promise(resolve => {
    timeoutId = setTimeoutFn(
      () => resolve({ outcome: 'grace-expired', closeError: null }),
      graceMs,
    );
  });
  const first = await Promise.race([closeAttempt, deadline]);
  if (timeoutId !== null) clearTimeoutFn(timeoutId);
  if (first.outcome === 'closed') {
    return {
      ok: true,
      outcome: 'closed',
      browserPid,
      graceMs,
      disconnected: false,
      disconnectError: null,
      closeError: null,
      terminationRequested: false,
      terminationError: null,
      elapsedMs: now() - startedAt,
    };
  }

  const disconnectResult = disconnect(browser);
  let terminationRequested = false;
  let terminationError = null;
  if (browserProcess && browserPid !== null && processExistsFn(browserPid)) {
    try {
      terminationRequested = browserProcess.kill('SIGTERM');
      if (!terminationRequested) terminationError = 'owned browser process rejected SIGTERM';
    } catch (error) {
      terminationError = error.message;
    }
  }
  const recovered = disconnectResult.disconnected && terminationError === null;
  return {
    ok: recovered,
    outcome: recovered
      ? `${first.outcome}-recovered`
      : `${first.outcome}-termination-failed`,
    browserPid,
    graceMs,
    ...disconnectResult,
    closeError: first.closeError,
    terminationRequested,
    terminationSignal: terminationRequested ? 'SIGTERM' : null,
    terminationError,
    elapsedMs: now() - startedAt,
  };
}

export { DEFAULT_TEARDOWN_GRACE_MS };
