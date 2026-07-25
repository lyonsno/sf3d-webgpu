/**
 * uv_unwrap_worker.js — Web Worker running the CPU UV-unwrap off the main thread.
 *
 * uv-unwrap is the second-largest CPU foreground gap (~216ms) profiling found,
 * and unwrapUV is already pure (typed arrays in/out, no DOM/GPU), so we import it
 * directly from texture_baker.js. gpu.js's GPUBufferUsage refs are inside
 * functions (not module top-level), so importing here does not crash the worker;
 * we never call the GPU functions.
 *
 * Output (newVertices/newNormals/newFaces/uvs/faceAssignment) is byte-identical
 * to the main-thread path — same unwrapUV code.
 */
import { unwrapUV } from './texture_baker.js';

self.onmessage = (e) => {
  const { vertices, faces, numVertices, numFaces, id } = e.data;
  try {
    const r = unwrapUV(new Float32Array(vertices), new Uint32Array(faces), numVertices, numFaces);
    // Transfer all output buffers back (zero-copy).
    const transfers = [
      r.newVertices.buffer, r.newNormals.buffer, r.newFaces.buffer,
      r.uvs.buffer, r.faceAssignment.buffer,
    ];
    self.postMessage({
      id, ok: true,
      newVertices: r.newVertices.buffer,
      newNormals: r.newNormals.buffer,
      newFaces: r.newFaces.buffer,
      uvs: r.uvs.buffer,
      faceAssignment: r.faceAssignment.buffer,
      newNumVertices: r.newNumVertices,
      newNumFaces: r.newNumFaces,
    }, transfers);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.stack || err) });
  }
};
