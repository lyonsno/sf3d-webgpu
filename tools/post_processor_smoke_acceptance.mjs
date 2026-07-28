export function evaluatePostProcessorSmokeAcceptance({
  armSelection,
  outputIdentical,
  cooperativeComplete,
  progressHonest,
  cadenceObserved,
}) {
  const paired = armSelection === 'all' || armSelection === 'pair';
  return Object.freeze({
    paired,
    ok: paired
      && outputIdentical === true
      && cooperativeComplete === true
      && progressHonest === true
      && cadenceObserved === true,
  });
}
