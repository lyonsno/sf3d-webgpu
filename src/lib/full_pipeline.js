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

const COND_SIZE = 512;
const TEX_RESOLUTION = 1024;

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

  // Step 3: UV unwrap (CPU, synchronous)
  const uvResult = await timed('uv-unwrap', async () => unwrapUV(
    meshResult.vertices, meshResult.faces, meshResult.numVertices, meshResult.numFaces));

  // Step 4: rasterize UV → per-texel 3D positions (CPU, synchronous)
  const rasterResult = await timed('uv-rasterize', async () => rasterizeUV(
    uvResult.uvs, uvResult.newVertices, uvResult.newFaces,
    uvResult.newNumFaces, TEX_RESOLUTION, uvResult.faceAssignment));

  // Step 5: texture bake (GPU triplane query)
  const bakeResult = await timed('texture-bake', () => bakeTexture(
    device, meshResult._triplaneDecoder, meshResult._triplanesBuf,
    meshResult._decoderWeights, rasterResult.positions3D, rasterResult.mask,
    rasterResult.tbnData, TEX_RESOLUTION));

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
    cooperativeReports: meshResult._cooperativeReports || {},
    dinoPayload: meshResult._dinoPayload || null,   // { shape, length, tokens } or null
    sdf: meshResult._sdf,
    isosurfaceThreshold: meshResult._isosurfaceThreshold,
    stageTimings: meshResult._stageTimings || {},
    totalMs: performance.now() - t0,
  };
}
