/**
 * full_pipeline.js — single-image → textured GLB, the complete
 * sf3d.image-to-mesh.webgpu-local.v0 route as one callable.
 *
 * This is the exact step 1–6 sequence main.js runs (inference → CLIP materials →
 * UV unwrap → rasterize → texture bake → GLB export), extracted so the
 * acceptance capsule exercises the SAME route the app does, for both the control
 * (kit 0.1.36) and candidate (kit 0.1.38) arms, without duplicating the sequence
 * inside a browser-eval string.
 *
 * `options` is forwarded to runInference (e.g. { cooperativeDino, dinoSchedulingMode,
 * dinoChunkBlocks, captureDinoPayload }), so the capsule selects the DINO
 * execution mode per arm while the rest of the route is identical.
 */

import { runInference } from './inference.js';
import { unwrapUV, rasterizeUV, bakeTexture, exportGLB } from './texture_baker.js';
import { estimateMaterials } from './clip_estimator.js';
import { makeCooperativeTextureBake } from './cooperative_texture_bake.js';

const COND_SIZE = 512;
const TEX_RESOLUTION = 1024;

/**
 * Run UV unwrap on the main thread, or on a Web Worker when one is supplied.
 * Byte-identical output either way (same unwrapUV code). Vertices/faces are
 * transferred zero-copy to the worker; outputs transferred back.
 */
async function runUvUnwrap(vertices, faces, numVertices, numFaces, worker) {
  if (!worker) return unwrapUV(vertices, faces, numVertices, numFaces);
  return await new Promise((resolve, reject) => {
    const id = Math.random().toString(36).slice(2);
    const onMessage = (e) => {
      if (e.data.id !== id) return;
      worker.removeEventListener('message', onMessage);
      if (!e.data.ok) { reject(new Error(`uv-unwrap worker failed: ${e.data.error}`)); return; }
      resolve({
        uvs: new Float32Array(e.data.uvs),
        newVertices: new Float32Array(e.data.newVertices),
        newNormals: new Float32Array(e.data.newNormals),
        newFaces: new Uint32Array(e.data.newFaces),
        faceAssignment: new Uint8Array(e.data.faceAssignment),
        newNumVertices: e.data.newNumVertices,
        newNumFaces: e.data.newNumFaces,
      });
    };
    worker.addEventListener('message', onMessage);
    // Transfer sliced copies (one copy) so the caller's arrays stay intact for
    // any downstream use while the worker gets zero-copy ownership.
    const vBuf = vertices.slice().buffer, fBuf = faces.slice().buffer;
    worker.postMessage({ vertices: vBuf, faces: fBuf, numVertices, numFaces, id }, [vBuf, fBuf]);
  });
}

export async function runFullPipelineToGlb(device, pipelines, weights, inputImage, options = {}, onProgress) {
  const report = (msg) => { if (onProgress) onProgress(msg); };
  const t0 = performance.now();

  // Absolute-timestamp stage spans on the same performance.now() clock as any
  // caller-side rAF probe, so each foreground frame gap can be attributed to the
  // stage executing during it. Inference substages (dinov2/two-stream/triplane/
  // marching) come back from runInference; the outer steps are marked here.
  const spans = [];
  const mark = (name, start, end) => spans.push({ name, start, end });
  const timed = async (name, fn) => { const s = performance.now(); const r = await fn(); mark(name, s, performance.now()); return r; };

  // Step 1: inference → untextured mesh + triplane data (+ optional DINO payload)
  const meshResult = await runInference(
    device, pipelines, weights, inputImage, report,
    { ...options, recordStageSpans: spans });

  // Step 2: CLIP material estimation (exact PyTorch preprocessing order)
  const clipStart = performance.now();
  const clipCanvas = document.createElement('canvas');
  clipCanvas.width = COND_SIZE;
  clipCanvas.height = COND_SIZE;
  const clipCtx = clipCanvas.getContext('2d');
  clipCtx.drawImage(inputImage, 0, 0, COND_SIZE, COND_SIZE);
  const clipRaw = clipCtx.getImageData(0, 0, COND_SIZE, COND_SIZE).data;
  const clipPixels = new Float32Array(COND_SIZE * COND_SIZE * 4);
  for (let i = 0; i < clipRaw.length; i++) clipPixels[i] = clipRaw[i] / 255.0;
  for (let i = 0; i < COND_SIZE * COND_SIZE; i++) {
    const a = clipPixels[i * 4 + 3];
    clipPixels[i * 4]     = (clipPixels[i * 4] * a + 0.5 * (1 - a)) * a;
    clipPixels[i * 4 + 1] = (clipPixels[i * 4 + 1] * a + 0.5 * (1 - a)) * a;
    clipPixels[i * 4 + 2] = (clipPixels[i * 4 + 2] * a + 0.5 * (1 - a)) * a;
  }
  const { roughness, metallic } = await estimateMaterials(device, clipPixels, COND_SIZE, COND_SIZE, weights);
  mark('clip-material-estimate', clipStart, performance.now());

  // Step 3: UV unwrap (CPU). Optionally offloaded to a Web Worker
  // (options.uvUnwrapWorker) — the second-largest CPU foreground gap (~216ms);
  // byte-identical output (same unwrapUV code).
  const uvResult = await timed('uv-unwrap', () => runUvUnwrap(
    meshResult.vertices, meshResult.faces, meshResult.numVertices, meshResult.numFaces,
    options.uvUnwrapWorker));

  // Step 4: rasterize UV → per-texel 3D positions (CPU, synchronous)
  const rasterResult = await timed('uv-rasterize', async () => rasterizeUV(
    uvResult.uvs, uvResult.newVertices, uvResult.newFaces,
    uvResult.newNumFaces, TEX_RESOLUTION, uvResult.faceAssignment));

  // Step 5: texture bake (GPU triplane query). Optionally cooperative — batch
  // the per-texel decode into yieldable GPU duties (options.cooperativeBake).
  let bakeReport = null;
  const bakeOptions = {};
  if (options.cooperativeBake) {
    const cooperativeBatch = makeCooperativeTextureBake(device, {
      batchTexels: options.bakeBatchTexels || 16384,
      schedulingMode: options.bakeSchedulingMode === 'disabled' ? 'disabled' : 'cooperative',
      onProgress: (p) => { if (p.percent != null) report(`Texture bake ${p.completedItems}/${p.totalItems} (${p.percent.toFixed(0)}%)`); },
    });
    bakeOptions.cooperativeBatch = async (numOccupied, makeBatch) => { bakeReport = await cooperativeBatch(numOccupied, makeBatch); };
  }
  const bakeResult = await timed('texture-bake', () => bakeTexture(
    device, meshResult._triplaneDecoder, meshResult._triplanesBuf,
    meshResult._decoderWeights, rasterResult.positions3D, rasterResult.mask,
    rasterResult.tbnData, TEX_RESOLUTION, bakeOptions));

  // Step 6: GLB export
  const glb = await timed('glb-export', () => exportGLB(
    uvResult.newVertices, uvResult.newNormals, uvResult.newFaces, uvResult.uvs,
    bakeResult.albedo, bakeResult.normalMap,
    uvResult.newNumVertices, uvResult.newNumFaces, TEX_RESOLUTION,
    roughness, metallic));

  return {
    stageSpans: spans,
    glb,                                   // ArrayBuffer
    numVertices: meshResult.numVertices,
    numFaces: meshResult.numFaces,
    uvNumVertices: uvResult.newNumVertices,
    uvNumFaces: uvResult.newNumFaces,
    roughness,
    metallic,
    cooperativeReports: { ...(meshResult._cooperativeReports || {}), ...(bakeReport ? { 'texture-bake': bakeReport } : {}) },
    dinoPayload: meshResult._dinoPayload || null,   // { shape, length, tokens } or null
    sdf: meshResult._sdf,
    isosurfaceThreshold: meshResult._isosurfaceThreshold,
    stageTimings: meshResult._stageTimings || {},
    totalMs: performance.now() - t0,
  };
}
