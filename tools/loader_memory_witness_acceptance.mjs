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
  if(report.requested?.sharedAllowance){
    const shared=report.sharedAllowance;
    need(shared?.requested?.cpuBytes===64&&shared?.requested?.gpuBytes===64,'exact tiny shared diagnostic allowance');
    need(shared?.distinctOwnedDevices===true,'two distinct actual owned devices');
    need(shared?.backend?.isFallbackAdapter===false&&/apple/i.test(shared.backend.vendor),'actual shared nonfallback Apple adapter');
    need(shared?.backends?.length===2&&shared.backends.every(b=>b?.isFallbackAdapter===false&&/apple/i.test(b.vendor)),'both independent adapter routes verified');
    need(JSON.stringify(shared?.outputs)==='[[14,22,26,34],[14,22,26,34]]','both guarded compute/readbacks complete');
    need(JSON.stringify(shared?.validationErrors)==='[null,null]','both shared-device validation scopes');
    for(const [scope,held,requested]of [['cpu',32,40],['gpu',48,32]]){
      const refusal=shared?.refusals?.[scope];
      need(refusal?.name==='SF3DMemoryBudgetError'&&refusal.memoryBudget?.liveBytes===held&&refusal.memoryBudget?.requestedBytes===requested&&refusal.memoryBudget?.maxBytes===64,'actual shared '+scope+' contention refusal');
      need(shared?.held?.[scope]?.liveBytes===held,'held shared '+scope+' backing');
      const rows=[shared?.parent,...(shared?.children??[])];
      need(rows.length===3&&rows.every(row=>row?.[scope]?.maxBytes===64&&row?.[scope]?.liveBytes===0&&row?.[scope]?.physicalMemoryMeasured===false),'shared '+scope+' effective allowance and retirement scope');
    }
  }
  errors.push(...acceptOwnedProcessWitness(report));
  return {ok:errors.length === 0, errors};
}

function acceptOwnedProcessWitness(report){
  const errors=[],need=(condition,text)=>{if(!condition)errors.push(text);};
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
  return errors;
}

export function acceptStagedTensorWitness(report){
  const errors=[],need=(condition,text)=>{if(!condition)errors.push(text);};
  need(report?.status==='passed'&&!report.error,'successful terminal path');
  const learned=report?.route==='sf3d-canonical-patch-embedding.v0';
  need(learned||report?.route==='sf3d-canonical-tensor-ranges-no-inference.v0','exact canonical-unit/patch-phase route');
  need(report?.source?.clean===true&&report.source.revision===report.requested?.revision,'clean exact requested source');
  need(report?.backend?.isFallbackAdapter===false&&/apple/i.test(report.backend.vendor),'actual nonfallback Apple adapter');
  const source=report?.canonicalSource,unit=learned?report?.patchPhase:report?.tensorUnit;
  need(/^[a-f0-9]{64}$/.test(source?.sha256??'')&&source.sha256===report.requested?.weightsSha256,'exact observed canonical artifact digest');
  need(JSON.stringify(source?.units?.map(x=>x.name))===JSON.stringify(report.requested?.tensorNames),'exact requested tensor selection');
  need(unit?.loadingReport?.mode==='tensor-ranges'&&unit.loadingReport.sourceETag===source?.etag&&unit.loadingReport.expectedWeightBytes===source?.byteLength,'effective identified range source');
  need(unit?.loadingReport?.ranges?.length===source?.units?.length+2,'all and only identified source units plus header');
  if(learned){
    const admission=report?.phaseAdmission,host=admission?.host;
    need(report.inputArtifact?.sha256===report.requested?.inputSha256,'exact observed phase input');
    need(admission?.authority==='reversible-selected-phase-only'&&admission?.verdict==='admitted'&&host?.source==='live-macos'&&host.hostname===report.source?.hostname,'fresh effective host admission');
    need(host?.model==='Mac14,9'&&host?.processor==='Apple M2 Pro','actual calibrated M2 Pro identity');
    need(Number.isSafeInteger(admission?.requiredBytes)&&admission.requiredBytes>0&&admission.requiredBytes<=host?.hostFreeBytes,'phase backing demand fits observed free bytes');
    need(JSON.stringify(unit?.shape)==='[1297,1024]'&&unit?.inputBytes===3145728&&unit?.outputBytes===5312512,'complete production-shape patch projection');
    need(unit?.finiteCount===1328128&&unit?.nonzeroCount>0,'finite nonblank learned output');
    need(/^[a-f0-9]{64}$/.test(unit?.outputSha256??'')&&/^[a-f0-9]{64}$/.test(unit?.inputSha256??''),'preserved complete input/output digests');
    need(unit?.reference?.tested>=66&&unit.reference.mismatches===0,'independent sampled arithmetic reference');
    if(report.requested.foreground){
      const fg=report.foreground;
      need(fg?.source?.revision===report.requested.foreground.revision&&fg.source.trackedClean===true,'exact tracked foreground source');
      need(report.browserArguments?.every(value=>!value.startsWith('--headless'))&&fg?.visibility==='headed-independent-browser','visible independent foreground route');
      need(fg?.sameDevice===true&&unit?.deviceOwnership==='pre-bound-caller-host','actual host device shared with learned operation');
      need(fg?.before?.active===true&&fg.after?.active===true&&!fg.after?.error,'ordinary foreground remains active');
      need(fg?.before?.renderer==='ordinary-volume'&&fg.after?.renderer==='ordinary-volume','actual ordinary renderer, not alternate/synthetic rendering');
      need(fg?.before?.grid===report.requested.foreground.grid&&fg.after?.grid===report.requested.foreground.grid,'effective requested grid variant');
      need(fg?.after?.frameCount>fg?.before?.frameCount&&fg?.after?.simStepCount>fg?.before?.simStepCount,'ordinary simulation and presentation advance through learned work');
      need(fg?.after?.submissions>fg?.before?.submissions,'native host queue submits through learned work');
      need(typeof report.evidencePaths?.foregroundBefore==='string'&&typeof report.evidencePaths?.foregroundAfter==='string','retained actual foreground frames');
      need(fg?.terminalBudget?.root?.cpu?.liveBytes===0&&fg.terminalBudget.root.gpu.liveBytes===0,'shared host allowance retired');
      need(fg?.terminalBudget?.children?.length>0&&fg.terminalBudget.children.every(row=>row.budget?.cpu?.liveBytes===0&&row.budget?.gpu?.liveBytes===0),'every guarded host device retired');
      need(fg?.textureEvents?.length>0&&fg.textureEvents.every(row=>row.bytes>0&&row.effective?.format===row.descriptor?.format),'native texture reservations and effective identity retained');
    }
  }else{
    need(Array.isArray(unit?.readbacks)&&unit.readbacks.length===source?.units?.length&&unit.readbacks.length>0,'complete native readbacks');
    for(const row of source?.units??[]){
      const actual=unit?.readbacks?.find(x=>x.name===row.name);
      need(actual?.bytes===row.expandedBytes&&Array.isArray(actual?.f32Words)&&actual.f32Words.length>0&&JSON.stringify(actual.f32Words)===JSON.stringify(row.expectedF32Words),'exact independent F32 reference: '+row.name);
    }
  }
  need(unit?.budget?.cpu?.liveBytes===0&&unit?.budget?.gpu?.liveBytes===0,'native unit resources retired');
  need(unit?.budget?.cpu?.maxBytes===report.requested?.cpuBytes&&unit?.budget?.gpu?.maxBytes===report.requested?.gpuBytes,'effective caller allowances');
  need(unit?.budget?.cpu?.physicalMemoryMeasured===false&&unit?.budget?.gpu?.physicalMemoryMeasured===false,'logical scope only');
  need(report?.validationError===null,'native validation scope');
  errors.push(...acceptOwnedProcessWitness(report));return {ok:errors.length===0,errors};
}
