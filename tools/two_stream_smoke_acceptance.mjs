export function evaluateTwoStreamSmokeAcceptance(options) {
  const {
    armSelection,
    outputIdentical,
    outputCanonical,
    cooperativeComplete,
    progressHonest,
  } = options;
  return Object.freeze({
    paired: armSelection === 'both',
    ok: armSelection === 'both'
      && outputIdentical === true
      && outputCanonical
      && cooperativeComplete
      && progressHonest,
  });
}
