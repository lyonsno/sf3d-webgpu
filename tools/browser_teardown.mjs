const DEFAULT_TEARDOWN_GRACE_MS = 10_000;
const DEFAULT_FORCE_GRACE_MS = 10_000;

export function probeOwnedProcess(pid, { killFn = process.kill } = {}) {
  if (!Number.isSafeInteger(pid) || pid <= 0) {
    return { state: 'unavailable', error: 'owned process PID is unavailable' };
  }
  try {
    killFn(pid, 0);
    return { state: 'present', error: null };
  } catch (error) {
    if (error.code === 'ESRCH') return { state: 'absent', error: null };
    return {
      state: 'unknown',
      error: `${error.code ? `${error.code}: ` : ''}${error.message}`,
    };
  }
}

function childExited(child) {
  return child?.exitCode != null || child?.signalCode != null;
}

export async function waitForOwnedProcessExit(child, {
  graceMs,
  probeProcessFn = probeOwnedProcess,
  setTimeoutFn = setTimeout,
  clearTimeoutFn = clearTimeout,
} = {}) {
  const pid = Number.isSafeInteger(child?.pid) ? child.pid : null;
  if (pid === null) {
    return {
      exited: false,
      outcome: 'identity-unavailable',
      exitCode: child?.exitCode ?? null,
      signalCode: child?.signalCode ?? null,
      finalProbe: probeProcessFn(pid),
    };
  }
  const initialProbe = probeProcessFn(pid);
  if (childExited(child) || initialProbe.state === 'absent') {
    return {
      exited: true,
      outcome: childExited(child) ? 'child-exit-state' : 'process-absent',
      exitCode: child.exitCode ?? null,
      signalCode: child.signalCode ?? null,
      initialProbe,
      finalProbe: initialProbe,
    };
  }

  return await new Promise(resolve => {
    let settled = false;
    let timer = null;
    const finish = result => {
      if (settled) return;
      settled = true;
      if (timer !== null) clearTimeoutFn(timer);
      child.removeListener?.('exit', onExit);
      resolve(result);
    };
    const onExit = (exitCode, signalCode) => finish({
      exited: true,
      outcome: 'exit-event',
      exitCode: exitCode ?? null,
      signalCode: signalCode ?? null,
      initialProbe,
      finalProbe: { state: 'absent', error: null },
    });
    child.once?.('exit', onExit);
    const postListenerProbe = probeProcessFn(pid);
    if (childExited(child) || postListenerProbe.state === 'absent') {
      finish({
        exited: true,
        outcome: childExited(child) ? 'child-exit-state' : 'process-absent',
        exitCode: child.exitCode ?? null,
        signalCode: child.signalCode ?? null,
        initialProbe,
        finalProbe: postListenerProbe,
      });
      return;
    }
    timer = setTimeoutFn(() => {
      const finalProbe = probeProcessFn(pid);
      const exited = childExited(child) || finalProbe.state === 'absent';
      finish({
        exited,
        outcome: exited
          ? (childExited(child) ? 'child-exit-state-at-deadline' : 'process-absent-at-deadline')
          : 'grace-expired',
        exitCode: child.exitCode ?? null,
        signalCode: child.signalCode ?? null,
        initialProbe,
        finalProbe,
      });
    }, graceMs);
  });
}

export async function stopOwnedChildProcess(child, {
  graceMs = DEFAULT_TEARDOWN_GRACE_MS,
  forceGraceMs = DEFAULT_FORCE_GRACE_MS,
  probeProcessFn = probeOwnedProcess,
  waitForExitFn = waitForOwnedProcessExit,
  setTimeoutFn = setTimeout,
  clearTimeoutFn = clearTimeout,
  now = () => performance.now(),
} = {}) {
  const startedAt = now();
  if (!child) {
    return {
      ok: true,
      outcome: 'not-started',
      pid: null,
      graceMs,
      forceGraceMs,
      signals: [],
      elapsedMs: now() - startedAt,
    };
  }
  const pid = Number.isSafeInteger(child.pid) ? child.pid : null;
  if (pid === null) {
    return {
      ok: false,
      outcome: 'identity-unavailable',
      pid: null,
      graceMs,
      forceGraceMs,
      signals: [],
      initialProbe: probeProcessFn(pid),
      error: 'owned child process PID is unavailable',
      elapsedMs: now() - startedAt,
    };
  }
  const initialProbe = probeProcessFn(pid);
  if (childExited(child) || initialProbe.state === 'absent') {
    return {
      ok: true,
      outcome: childExited(child) ? 'already-exited' : 'already-absent',
      pid,
      graceMs,
      forceGraceMs,
      signals: [],
      initialProbe,
      finalProbe: initialProbe,
      exitCode: child.exitCode ?? null,
      signalCode: child.signalCode ?? null,
      elapsedMs: now() - startedAt,
    };
  }

  const signals = [];
  const requestSignal = signal => {
    try {
      const accepted = child.kill(signal);
      signals.push({ signal, accepted, error: accepted ? null : 'signal rejected' });
      return { accepted, error: accepted ? null : 'signal rejected' };
    } catch (error) {
      signals.push({ signal, accepted: false, error: error.message });
      return { accepted: false, error: error.message };
    }
  };

  const gracefulSignal = requestSignal('SIGTERM');
  if (!gracefulSignal.accepted) {
    const finalProbe = probeProcessFn(pid);
    const exited = childExited(child) || finalProbe.state === 'absent';
    return {
      ok: exited,
      outcome: exited ? 'exited-before-sigterm' : 'sigterm-failed',
      pid,
      graceMs,
      forceGraceMs,
      signals,
      initialProbe,
      finalProbe,
      error: exited ? null : gracefulSignal.error,
      elapsedMs: now() - startedAt,
    };
  }

  const gracefulWait = await waitForExitFn(child, {
    graceMs,
    probeProcessFn,
    setTimeoutFn,
    clearTimeoutFn,
  });
  if (gracefulWait.exited) {
    return {
      ok: true,
      outcome: 'exited-after-sigterm',
      pid,
      graceMs,
      forceGraceMs,
      signals,
      initialProbe,
      gracefulWait,
      finalProbe: gracefulWait.finalProbe,
      exitCode: gracefulWait.exitCode,
      signalCode: gracefulWait.signalCode,
      elapsedMs: now() - startedAt,
    };
  }

  const forceSignal = requestSignal('SIGKILL');
  if (!forceSignal.accepted) {
    const finalProbe = probeProcessFn(pid);
    const exited = childExited(child) || finalProbe.state === 'absent';
    return {
      ok: exited,
      outcome: exited ? 'exited-before-sigkill' : 'sigkill-failed',
      pid,
      graceMs,
      forceGraceMs,
      signals,
      initialProbe,
      gracefulWait,
      finalProbe,
      error: exited ? null : forceSignal.error,
      elapsedMs: now() - startedAt,
    };
  }

  const forceWait = await waitForExitFn(child, {
    graceMs: forceGraceMs,
    probeProcessFn,
    setTimeoutFn,
    clearTimeoutFn,
  });
  return {
    ok: forceWait.exited,
    outcome: forceWait.exited ? 'exited-after-sigkill' : 'sigkill-grace-expired',
    pid,
    graceMs,
    forceGraceMs,
    signals,
    initialProbe,
    gracefulWait,
    forceWait,
    finalProbe: forceWait.finalProbe,
    exitCode: forceWait.exitCode,
    signalCode: forceWait.signalCode,
    error: forceWait.exited ? null : (forceWait.finalProbe?.error ?? 'owned process did not exit'),
    elapsedMs: now() - startedAt,
  };
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
  forceGraceMs = DEFAULT_FORCE_GRACE_MS,
  probeProcessFn = probeOwnedProcess,
  waitForExitFn = waitForOwnedProcessExit,
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
      forceGraceMs,
      elapsedMs: now() - startedAt,
    };
  }

  let browserProcess = null;
  let processIdentityError = null;
  try { browserProcess = browser.process?.() ?? null; }
  catch (error) { processIdentityError = error.message; }
  const browserPid = Number.isSafeInteger(browserProcess?.pid) ? browserProcess.pid : null;
  if (browserPid === null) {
    return {
      ok: false,
      outcome: 'process-identity-unavailable',
      browserPid: null,
      graceMs,
      forceGraceMs,
      processIdentityError: processIdentityError ?? 'Puppeteer returned no owned browser process',
      elapsedMs: now() - startedAt,
    };
  }

  const initialProbe = probeProcessFn(browserPid);
  if (initialProbe.state === 'absent' || childExited(browserProcess)) {
    const disconnectResult = disconnect(browser);
    return {
      ok: disconnectResult.disconnected,
      outcome: disconnectResult.disconnected
        ? 'already-exited-disconnected'
        : 'already-exited-disconnect-failed',
      browserPid,
      graceMs,
      forceGraceMs,
      processIdentityError,
      initialProbe,
      ...disconnectResult,
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
  const closeResult = await Promise.race([closeAttempt, deadline]);
  if (timeoutId !== null) clearTimeoutFn(timeoutId);
  const disconnectResult = closeResult.outcome === 'closed'
    ? { disconnected: false, disconnectError: null }
    : disconnect(browser);
  const processTeardown = await stopOwnedChildProcess(browserProcess, {
    graceMs,
    forceGraceMs,
    probeProcessFn,
    waitForExitFn,
    setTimeoutFn,
    clearTimeoutFn,
    now,
  });
  const recovered = processTeardown.ok
    && (closeResult.outcome === 'closed' || disconnectResult.disconnected);
  return {
    ok: recovered,
    outcome: recovered
      ? `${closeResult.outcome}-process-exited`
      : `${closeResult.outcome}-teardown-failed`,
    browserPid,
    graceMs,
    forceGraceMs,
    processIdentityError,
    initialProbe,
    ...disconnectResult,
    closeError: closeResult.closeError,
    processTeardown,
    elapsedMs: now() - startedAt,
  };
}

export { DEFAULT_TEARDOWN_GRACE_MS, DEFAULT_FORCE_GRACE_MS };
