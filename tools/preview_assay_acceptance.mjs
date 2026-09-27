export function acceptPreviewAssay(report) {
  const { result, requested } = report;
  if (!Number.isSafeInteger(result?.final?.vertices) || result.final.vertices <= 0
      || !Number.isSafeInteger(result?.final?.faces) || result.final.faces <= 0) {
    throw new Error('final mesh is empty or invalid');
  }
  if (result.browserErrors?.length) throw new Error('browser errors were reported');
  if (requested.partial) {
    if (!result.partialPreviews?.some(state => state.mesh?.faces > 0)) {
      throw new Error('all partial previews are missing or empty');
    }
  } else if (requested.intermediateStages?.length) {
    for (const stageId of requested.intermediateStages) {
      const state = result.intermediatePreviews?.find(state => state.stageId === stageId && state.mesh?.faces > 0);
      if (!state) {
        throw new Error(`missing intermediate preview ${stageId}`);
      }
      if (!Number.isFinite(state.availableMs) || !Number.isFinite(result.inferenceMs)) {
        throw new Error(`missing intermediate preview timing for ${stageId}`);
      }
      if (state.availableMs >= result.inferenceMs) {
        throw new Error(`intermediate preview ${stageId} arrived after final inference`);
      }
    }
  } else if (requested.intermediateStage) {
    if (!result.intermediatePreview?.mesh?.faces) {
      throw new Error('intermediate preview is missing or empty');
    }
    if (!Number.isFinite(result.intermediatePreview.availableMs)
        || !Number.isFinite(result.inferenceMs)) {
      throw new Error('missing intermediate preview timing');
    }
    if (result.intermediatePreview.availableMs >= result.inferenceMs) {
      throw new Error('intermediate preview arrived after final inference');
    }
  } else if (!result.previewMesh?.faces) {
    throw new Error('final triplane preview is missing or empty');
  }
  return true;
}
