/**
 * materialize_worker.js — Web Worker running the texture materialization
 * (albedo + normal + dilation) off the main thread. Cranial's assay's ~752ms
 * CPU tail. Uses the SAME materialize_core math, so output is byte-identical.
 *
 * Receives the decoded feature/normal payloads + TBN/mask (transferred zero-copy)
 * and returns the albedo + normal Uint8Arrays (transferred back).
 */
import { materializeTextures } from './materialize_core.js';

self.onmessage = (e) => {
  const { featuresBuf, normalsBuf, occupiedBuf, tbnBuf, maskBuf, resolution, numOccupied, id } = e.data;
  try {
    const { albedo, normalMap } = materializeTextures({
      featuresCPU: new Float32Array(featuresBuf),
      normalsCPU: new Float32Array(normalsBuf),
      occupiedIndices: new Uint32Array(occupiedBuf),
      tbnData: new Float32Array(tbnBuf),
      mask: new Uint8Array(maskBuf),
      resolution,
      numOccupied,
    });
    self.postMessage(
      { id, ok: true, albedo: albedo.buffer, normalMap: normalMap.buffer },
      [albedo.buffer, normalMap.buffer],
    );
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.stack || err) });
  }
};
