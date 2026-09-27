export async function runOptionalPreview(stageId, run, onError = null) {
  try {
    return { status: 'complete', value: await run() };
  } catch (error) {
    let notificationError = null;
    try { await onError?.({ stageId, error }); }
    catch (notifyError) { notificationError = notifyError; }
    return { status: 'skipped', error, notificationError };
  }
}
