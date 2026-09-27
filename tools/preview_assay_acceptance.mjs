export function acceptPreviewAssay(report) {
  const { result, requested } = report;
  if (!Number.isSafeInteger(result?.final?.vertices) || result.final.vertices <= 0
      || !Number.isSafeInteger(result?.final?.faces) || result.final.faces <= 0) {
    throw new Error('final mesh is empty or invalid');
  }
  if (result.browserErrors?.length) throw new Error('browser errors were reported');
  if (requested.reducedFactor != null) {
    if (!result.reducedPreview?.mesh?.faces) throw new Error('reduced preview is missing or empty');
  } else if (requested.partial) {
    if (!result.partialPreviews?.some(state => state.mesh?.faces > 0)) {
      throw new Error('all partial previews are missing or empty');
    }
  } else if (!result.previewMesh?.faces) {
    throw new Error('final triplane preview is missing or empty');
  }
  return true;
}
