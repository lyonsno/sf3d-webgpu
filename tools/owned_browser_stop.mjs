// Exact ChildProcess custody only; never resolve a browser from focus or signal a PID glob.
export async function stopOwnedBrowser(child, {graceMs = 10000} = {}) {
  if (!child || !Number.isSafeInteger(child.pid) || child.pid < 1) throw new Error('exact owned browser child is required');
  if (!Number.isFinite(graceMs) || graceMs <= 0) throw new Error('positive cleanup grace required');
  const report = {ownedPid:child.pid, signal:null, signals:[], exitObserved:false};
  if (child.exitCode !== null || child.signalCode !== null) return {...report, status:'already-exited', exitObserved:true, code:child.exitCode, exitSignal:child.signalCode};
  let timer, onExit;
  const exited = new Promise((resolve, reject) => {
    onExit = (code, signal) => resolve({...report, status:'stopped', exitObserved:true, code, exitSignal:signal});
    child.once('exit', onExit);
    timer = setTimeout(() => {
      try {
        report.signal='SIGKILL';report.signals.push('SIGKILL');
        if(!child.kill('SIGKILL'))throw new Error('owned browser SIGKILL was not accepted');
        timer=setTimeout(()=>reject(new Error(`owned browser ${child.pid} has not exited after SIGTERM/SIGKILL`)),graceMs);
      }catch(error){reject(error);}
    }, graceMs);
  });
  try {
    report.signal = 'SIGTERM';report.signals.push('SIGTERM');
    if (!child.kill('SIGTERM')) throw new Error('owned browser SIGTERM was not accepted');
    return await exited;
  } finally { clearTimeout(timer); child.removeListener('exit',onExit); }
}

export async function ownedBrowserArguments(puppeteer,{profile,headless=true}) {
  if(typeof headless!=='boolean')throw Error('explicit boolean browser visibility required');
  const args=await puppeteer.defaultArgs({headless,userDataDir:profile,
    args:['--remote-debugging-port=0','--enable-unsafe-webgpu','--use-angle=metal','--no-first-run','--use-mock-keychain','--password-store=basic']});
  if(!Array.isArray(args)||args.some(value=>typeof value!=='string')||
    (headless?!args.includes('--headless=new'):args.some(value=>value.startsWith('--headless')))||!args.includes('--user-data-dir='+profile))
    throw Error('resolved isolated headless browser arguments required before spawn');
  return args;
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
