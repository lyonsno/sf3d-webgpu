export function acceptLoaderMemoryWitness(report) {
  const errors = [], need = (condition, text) => { if (!condition) errors.push(text); };
  need(report?.status === 'passed' && !report?.error, 'successful terminal path');
  need(report?.route === 'sf3d-loader-native-refusal-synthetic-weights-no-inference.v0', 'exact diagnostic route');
  need(report?.source?.clean === true && report.source.revision === report.requested?.revision, 'clean exact requested source');
  need(report?.backend?.isFallbackAdapter === false && /apple/i.test(report.backend.vendor), 'actual nonfallback Apple adapter');
  for (const key of ['sourceRefusal','conversionRefusal','gpuRefusal']) {
    const row = report?.cases?.[key];
    need(row?.error?.name === 'SF3DMemoryBudgetError', key+' actual budget refusal');
    need(row?.budget?.cpu?.liveBytes === 0 && row?.budget?.gpu?.liveBytes === 0, key+' partial cleanup');
    need(row?.budget?.cpu?.maxBytes === row?.requested?.cpuBytes && row?.budget?.gpu?.maxBytes === row?.requested?.gpuBytes, key+' effective allowance');
    need(row?.budget?.cpu?.physicalMemoryMeasured === false && row?.budget?.gpu?.physicalMemoryMeasured === false, key+' narrow scope');
  }
  need(report?.cases?.sourceRefusal?.fetchCount === 0, 'source refusal before fetch');
  need(report?.cases?.conversionRefusal?.budget?.refusal?.label === 'fp16-conversion' && report?.cases?.conversionRefusal?.budget?.gpu?.peakLiveBytes === 0, 'conversion refused before GPU creation');
  need(report?.cases?.gpuRefusal?.budget?.gpu?.peakLiveBytes === 8, 'exact native allocation prefix before refusal');
  need(JSON.stringify(report?.deviceControl) === '[7,11,13,17]', 'same device remains usable after cleanup');
  need(report?.validationError === null, 'native validation scope');
  need(report?.memorySafety?.stop?.exitObserved === true && report.memorySafety.stop.ownedPid === report.ownedBrowserPid, 'exact owned browser exit');
  need(report?.processRefusal?.status === 'budget-refused' && report.processRefusal?.safety?.reason === 'process-footprint-budget', 'actual process threshold refusal');
  need(report?.processRefusal?.lastObservation?.effectiveRoute === 'darwin-libproc-proc_pid_rusage/RUSAGE_INFO_V4', 'actual Darwin observer');
  const summary=report?.processRefusal, observation=summary?.lastObservation;
  need(summary?.coverage === 'sampled-owned-process-tree' && observation?.status === 'observed', 'complete current process coverage');
  need(summary?.runId === report?.runId+'-refusal' && observation?.runId === summary?.runId &&
    summary?.rootPid === report?.processObservation?.rootPid && observation?.rootPid === summary?.rootPid, 'current owned observation identity');
  need(Array.isArray(observation?.processes) && observation.processes.some(p=>p.pid===summary?.rootPid) &&
    observation.processes.some(p=>p.pid===report?.ownedBrowserPid), 'observed owner and browser before intervention');
  need(summary?.safety?.actionStatus === 'returned' && summary?.safety?.observedBytes === observation?.sampledAggregatePhysicalFootprintBytes, 'completed action on measured threshold crossing');
  need(report?.processRefusal?.safety?.observedBytes > report.processRefusal?.safety?.maxFootprintBytes, 'observed native threshold crossing');
  need(report?.cleanup?.browser?.exitObserved === true && report.cleanup.browser.ownedPid === report.ownedBrowserPid && report?.cleanup?.server === 'closed', 'owned resources closed');
  return {ok:errors.length === 0, errors};
}
