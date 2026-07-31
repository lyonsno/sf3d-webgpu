/**
 * Pure assembly of the partial-arm failure record — the F2 remedy from
 * cranial-depth-enema's exact-commit review of 7393ab06.
 *
 * The A/B runs all arms inside one browser `page.evaluate`. Before this fix, if
 * a later arm (the bounded fifth arm) threw, Node received no prior-arm results
 * and the failure report fell back to the browser-setup checkpoint — unable to
 * demonstrate bounded-prefix drain-on-failure and needlessly discarding the
 * already completed strict/control evidence.
 *
 * The browser now keeps an ordered completed-arm ledger and a failure record.
 * This helper turns those raw browser-owned values into the durable failure
 * evidence the review requires: it NAMES the failing arm, the last completed
 * prior arm, and the failing arm's post-drain settlement state. Keeping it pure
 * makes the failure path deterministically testable without a browser.
 *
 * @param {object} partial
 * @param {Array<{name:string, cooperativeStatus?:string|null,
 *   completionPolicy?:string|null, issuedGpuDutyCount?:number|null,
 *   retiredGpuDutyCount?:number|null, inFlightGpuDutyCount?:number|null}>} partial.ledger
 *   completed arms in completion order (each already fully settled)
 * @param {{message:string, completedArms?:string[], lastCompletedArm?:string|null,
 *   drainAfterFailure?:object|null}|null} partial.failure
 *   the browser-side failure record for the arm that threw
 * @returns {object} durable partial-arm failure evidence
 */
export function assemblePartialArmFailure({ ledger, failure }) {
  const completed = Array.isArray(ledger) ? ledger : [];
  const completedArmNames = completed.map((entry) => entry.name);
  const lastCompletedArm = completedArmNames.length
    ? completedArmNames[completedArmNames.length - 1]
    : null;

  // The failing arm is the browser-recorded one; if the browser failure record
  // is missing (e.g. the throw escaped before it was written) we still surface
  // the completed ledger rather than losing it.
  const failingArm = failure?.failingArm
    ?? failure?.armName
    ?? deriveFailingArm(failure, completedArmNames);

  return {
    phase: 'browser-arms-partial',
    // Completed-arm evidence is preserved, in order, with each arm's settlement.
    completedArms: completed,
    completedArmNames,
    lastCompletedArm,
    // The arm that failed and its post-drain settlement (nonzero-exit witness).
    failingArm,
    failureMessage: failure?.message ?? null,
    drainAfterFailure: failure?.drainAfterFailure ?? null,
    // A well-formed partial failure names the failing arm and preserves at least
    // the drain settlement OR a nonempty completed ledger; if neither is present
    // the browser lost evidence and the caller should treat it as incomplete.
    evidencePreserved: Boolean(
      failure
      && (failure.drainAfterFailure != null || completed.length > 0),
    ),
  };
}

function deriveFailingArm(failure, completedArmNames) {
  if (!failure) return null;
  // If the browser recorded which arms completed, the failing arm is the first
  // not in that set is unknowable here; fall back to null rather than guess.
  if (Array.isArray(failure.completedArms) && failure.completedArms.length) {
    return null;
  }
  return null;
}
