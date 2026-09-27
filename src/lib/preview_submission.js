export async function submitPreviewAndRetire(device, encoder, allocations, consume) {
  let result;
  let primaryError = null;
  try {
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    result = await consume();
  } catch (error) {
    primaryError = error;
  }
  let settlementError = null;
  try {
    await device.queue.onSubmittedWorkDone();
  } catch (error) {
    settlementError = error;
  }
  const retirementErrors = [];
  for (const { buffer } of allocations) {
    try { buffer.destroy(); } catch (error) { retirementErrors.push(error); }
  }
  if (primaryError && typeof primaryError === 'object') {
    try {
      if (settlementError) primaryError.previewSettlementError = settlementError;
      if (retirementErrors.length) primaryError.previewRetirementErrors = retirementErrors;
    } catch {
      // Preserve the original failure even if the thrown object is immutable.
    }
  }
  if (primaryError) throw primaryError;
  if (settlementError) throw settlementError;
  if (retirementErrors.length) {
    throw new AggregateError(retirementErrors, 'preview buffer retirement failed');
  }
  return result;
}
