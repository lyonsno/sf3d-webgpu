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
      if (!result.intermediatePreviews?.some(state => state.stageId === stageId && state.mesh?.faces > 0)) {
        throw new Error(`missing intermediate preview ${stageId}`);
      }
    }
  } else if (requested.intermediateStage) {
    if (!result.intermediatePreview?.mesh?.faces) {
      throw new Error('intermediate preview is missing or empty');
    }
  } else if (!result.previewMesh?.faces) {
    throw new Error('final triplane preview is missing or empty');
  }
  return true;
}
