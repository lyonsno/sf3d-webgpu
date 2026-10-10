// Exact ChildProcess custody only; never resolve a browser from focus or signal a PID glob.
export async function stopOwnedBrowser(child, {graceMs = 10000} = {}) {
  if (!child || !Number.isSafeInteger(child.pid) || child.pid < 1) throw new Error('exact owned browser child is required');
  if (!Number.isFinite(graceMs) || graceMs <= 0) throw new Error('positive cleanup grace required');
  const report = {ownedPid:child.pid, signal:null, exitObserved:false};
  if (child.exitCode !== null || child.signalCode !== null) return {...report, status:'already-exited', exitObserved:true, code:child.exitCode, exitSignal:child.signalCode};
  let timer, onExit;
  const exited = new Promise((resolve, reject) => {
    onExit = (code, signal) => resolve({...report, status:'stopped', exitObserved:true, code, exitSignal:signal});
    child.once('exit', onExit);
    timer = setTimeout(() => reject(new Error(`owned browser ${child.pid} has not exited after SIGTERM`)), graceMs);
  });
  try {
    if (!child.kill('SIGTERM')) throw new Error('owned browser SIGTERM was not accepted');
    report.signal = 'SIGTERM'; return await exited;
  } finally { clearTimeout(timer); child.removeListener('exit',onExit); }
}

export function memoryStopAction({child, report, persist}) {
  return async safety => {
    report.memorySafety = {...safety, action:'stop-exact-owned-browser', ownedPid:child()?.pid ?? null};
    try { await persist(); } catch (error) { report.memorySafety.reportError = error.message; }
    const owned = child();
    report.memorySafety.stop = owned ? await stopOwnedBrowser(owned) : {status:'launch-prevented', exitObserved:true};
    await persist();
  };
}
