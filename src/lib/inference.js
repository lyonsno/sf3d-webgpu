/**
 * inference.js — SF3D WebGPU inference pipeline.
 *
 * Full forward pass:
 *   1. Image preprocessing (normalize, resize to 512×512)
 *   2. Camera embedding (fixed default view)
 *   3. DINOv2 image tokenization (with AdaNorm modulation)
 *   4. Triplane tokenization (learned embeddings)
 *   5. Two-stream backbone (interleave transformer)
 *   6. PixelShuffle post-processing
 *   7. Triplane query + decoder MLP
 *   8. Marching tetrahedra mesh extraction
 *
 * Steps 1, 7, 8 run on CPU. Steps 2-6 run entirely on GPU.
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';
import {
  dispatchConv2d, dispatchConv1x1, dispatchActivation,
  dispatchPixelShuffle,
} from './shader_ops.js';

// SF3D model configuration
const CONFIG = {
  condImageSize: 512,
  patchSize: 14,
  hiddenDim: 1024,
  numHeads: 16,
  headDim: 64,
  numEncoderLayers: 24,
  triplanePlaneSize: 96,
  triplaneChannels: 1024,
  numLatents: 1792,
  numBackboneBlocks: 4,
  numBasicBlocks: 3,
  isosurfaceResolution: 160,
  isosurfaceThreshold: 10.0,
  radius: 0.87,
  defaultFovDeg: 40.0,
  defaultDistance: 1.6,
  // ImageNet normalization for DINOv2
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  // Background color
  bgColor: [0.5, 0.5, 0.5],
};

/**
 * Preprocess an image for SF3D input.
 * Returns Float32Array in CHW format, normalized with ImageNet stats.
 */
export function preprocessImage(imageData, width, height) {
  const size = CONFIG.condImageSize;
  // Create a canvas to resize
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  // Draw image centered and scaled
  ctx.fillStyle = `rgb(${CONFIG.bgColor.map(c => Math.round(c * 255)).join(',')})`;
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(imageData, 0, 0, size, size);

  const pixels = ctx.getImageData(0, 0, size, size);
  const data = pixels.data;

  // Convert to CHW float, normalize with ImageNet stats
  const chw = new Float32Array(3 * size * size);
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const pixIdx = (y * size + x) * 4;
        const val = data[pixIdx + c] / 255.0;
        chw[c * size * size + y * size + x] = (val - CONFIG.imageMean[c]) / CONFIG.imageStd[c];
      }
    }
  }

  return chw;
}

/**
 * Compute default camera embeddings.
 * SF3D uses a fixed camera: c2w at distance 1.6, fov 40°.
 */
export function computeCameraEmbedding() {
  // Default c2w matrix (4×4 flattened)
  const c2w = new Float32Array([
    0, 0, 1, CONFIG.defaultDistance,
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
  ]);

  // Intrinsic matrix (3×3 normalized)
  const fov = CONFIG.defaultFovDeg * Math.PI / 180;
  const focal = 0.5 / Math.tan(fov / 2);
  const intrinsicNormed = new Float32Array([
    focal, 0, 0.5,
    0, focal, 0.5,
    0, 0, 1,
  ]);

  // Camera embedder input: concat(c2w.flatten(), intrinsic_normed.flatten()) = 16 + 9 = 25
  const embedInput = new Float32Array(25);
  embedInput.set(c2w, 0);
  embedInput.set(intrinsicNormed, 16);

  return embedInput;
}

/**
 * Run the full SF3D inference pipeline.
 */
export async function runInference(device, weights, imageElement, onProgress) {
  const report = (msg) => { if (onProgress) onProgress(msg); console.log(msg); };

  // 1. Preprocess image
  report('Preprocessing image...');
  const imageData = preprocessImage(imageElement,
    imageElement.naturalWidth || imageElement.width,
    imageElement.naturalHeight || imageElement.height);
  const imageBuf = createStorageBuffer(device, imageData);

  // 2. Compute camera embedding
  report('Computing camera embedding...');
  const cameraInput = computeCameraEmbedding();
  // camera_embed = linear(camera_input) → [768]
  // This is a tiny operation, do it on CPU
  const cameraInputBuf = createStorageBuffer(device, cameraInput);

  // TODO: Dispatch camera embedding linear
  // TODO: Dispatch DINOv2 with modulated attention
  // TODO: Dispatch triplane tokenization
  // TODO: Dispatch two-stream backbone
  // TODO: Dispatch post-processor
  // TODO: Triplane query + decoder (CPU for now)
  // TODO: Marching tetrahedra (CPU)

  report('Pipeline dispatch not yet implemented — scaffolding complete');

  return null;
}
