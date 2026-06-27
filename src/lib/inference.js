/**
 * inference.js — SF3D WebGPU inference pipeline.
 *
 * Full forward pass:
 *   1. Image preprocessing (normalize, resize to 512×512)
 *   2. Camera embedding (linear projection)
 *   3. DINOv2 image tokenization (with AdaNorm modulation)
 *   4. Two-stream backbone (interleave transformer)
 *   5. PixelShuffle post-processing
 *   6. Triplane query + decoder MLP
 *   7. Marching tetrahedra mesh extraction (CPU)
 *
 * Steps 1 run on CPU. Steps 2-6 run on GPU. Step 7 runs on CPU.
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';
import { SF3DImageTokenizer } from './sf3d_backbone.js';
import { TwoStreamBackbone } from './two_stream.js';
import { dispatchPostProcessor } from './post_processor.js';
import { TriplaneDecoder } from './triplane_decoder.js';
import { loadTetData, marchingTetrahedra, scaleTensor } from './marching_tet.js';

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
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = `rgb(${CONFIG.bgColor.map(c => Math.round(c * 255)).join(',')})`;
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(imageData, 0, 0, size, size);

  const pixels = ctx.getImageData(0, 0, size, size);
  const data = pixels.data;

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
 * Compute default camera embeddings input.
 * SF3D uses a fixed camera: c2w at distance 1.6, fov 40°.
 * Returns [25] float array: concat(c2w_4x4, intrinsic_normed_3x3).
 */
export function computeCameraInput() {
  const c2w = new Float32Array([
    0, 0, 1, CONFIG.defaultDistance,
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
  ]);

  const fov = CONFIG.defaultFovDeg * Math.PI / 180;
  const focal = 0.5 / Math.tan(fov / 2);
  const intrinsicNormed = new Float32Array([
    focal, 0, 0.5,
    0, focal, 0.5,
    0, 0, 1,
  ]);

  const embedInput = new Float32Array(25);
  embedInput.set(c2w, 0);
  embedInput.set(intrinsicNormed, 16);
  return embedInput;
}

/**
 * Initialize all GPU pipeline modules.
 */
export function initPipelines(device) {
  const imageTokenizer = new SF3DImageTokenizer(device);
  imageTokenizer.init();

  const twoStream = new TwoStreamBackbone(device);
  twoStream.init();

  const triplaneDecoder = new TriplaneDecoder(device);
  triplaneDecoder.init();

  return { imageTokenizer, twoStream, triplaneDecoder };
}

/**
 * Run the full SF3D inference pipeline.
 *
 * @param {GPUDevice} device
 * @param {Object} pipelines - from initPipelines()
 * @param {Object} weights - from loadWeights()
 * @param {HTMLImageElement|HTMLCanvasElement} imageElement
 * @param {Function} onProgress - progress callback
 * @returns {{ vertices: Float32Array, faces: Uint32Array, numVertices: number, numFaces: number }}
 */
export async function runInference(device, pipelines, weights, imageElement, onProgress) {
  const report = (msg) => { if (onProgress) onProgress(msg); console.log(msg); };

  // 1. Preprocess image (CPU)
  report('Preprocessing image...');
  const imageData = preprocessImage(imageElement,
    imageElement.naturalWidth || imageElement.width,
    imageElement.naturalHeight || imageElement.height);
  const imageBuf = createStorageBuffer(device, imageData);

  // 2. Camera embedding (GPU)
  report('Computing camera embedding...');
  const cameraInput = computeCameraInput();
  const cameraInputBuf = createStorageBuffer(device, cameraInput);

  // Dispatch camera embedding: Linear(25 → 768)
  const encoder1 = device.createCommandEncoder();
  const cameraEmbedBuf = createEmptyBuffer(device, 768 * 4);
  _dispatchLinear(device, encoder1, pipelines, cameraInputBuf, cameraEmbedBuf,
    weights.cameraEmbedder.weight, weights.cameraEmbedder.bias, 1, 25, 768);
  device.queue.submit([encoder1.finish()]);

  // 3. DINOv2 image tokenization (GPU)
  report('Running DINOv2 backbone...');
  const encoder2 = device.createCommandEncoder();
  const dinov2Result = pipelines.imageTokenizer.encode(
    encoder2, imageBuf, cameraEmbedBuf, weights.imageTokenizer);
  device.queue.submit([encoder2.finish()]);

  // 4. Two-stream backbone (GPU)
  report('Running two-stream backbone...');
  const encoder3 = device.createCommandEncoder();

  // Backbone needs tokenizer embeddings on the weights object
  weights.backbone.tokenizer_embeddings_buf = weights.tokenizer.embeddings;

  const backboneResult = pipelines.twoStream.forward(
    encoder3, dinov2Result.tokensBuf, dinov2Result.N, weights.backbone);
  device.queue.submit([encoder3.finish()]);

  // 5. PixelShuffle post-processing (GPU)
  report('Running post-processor...');
  const encoder4 = device.createCommandEncoder();
  const triplaneResult = dispatchPostProcessor(
    device, encoder4, backboneResult.buffer, weights.postProcessor);
  device.queue.submit([encoder4.finish()]);

  // 6. Triplane query + decoder (GPU)
  report('Querying triplane and decoding...');

  // Load tet grid data
  const tetData = await loadTetData('tets/');
  report(`Loaded tet grid: ${tetData.numVertices} vertices, ${tetData.numTets} tets`);

  // Scale grid vertices from [0, 1] to bbox
  const bbox = [-CONFIG.radius, CONFIG.radius];
  const gridPositions = scaleTensor(tetData.gridVertices, [0, 1], bbox);
  const gridPosBuf = createStorageBuffer(device, gridPositions);

  // First pass: density + vertex_offset for mesh extraction
  const encoder5 = device.createCommandEncoder();
  const decoded = pipelines.triplaneDecoder.decode(
    encoder5, gridPosBuf, triplaneResult.buffer, tetData.numVertices,
    weights.decoder, ['density', 'vertex_offset']);
  device.queue.submit([encoder5.finish()]);

  // Read back density and vertex_offset to CPU
  report('Reading back SDF values...');
  const densityCPU = await readBuffer(device, decoded.density, tetData.numVertices * 4);
  const sdf = new Float32Array(densityCPU);

  // Subtract threshold: sdf = density - threshold
  for (let i = 0; i < sdf.length; i++) {
    sdf[i] -= CONFIG.isosurfaceThreshold;
  }

  const vertexOffsetCPU = await readBuffer(device, decoded.vertex_offset, tetData.numVertices * 3 * 4);
  const vertexOffsets = new Float32Array(vertexOffsetCPU);

  // 7. Marching tetrahedra (CPU)
  report('Extracting mesh...');
  // Grid vertices need to be in model space with deformation applied
  // The grid is in [0, 1], scale to bbox for the marching tet
  const mesh = marchingTetrahedra(
    gridPositions, sdf, tetData.indices, vertexOffsets, CONFIG.isosurfaceResolution);

  report(`Mesh extracted: ${mesh.numVertices} vertices, ${mesh.numFaces} faces`);

  // Scale mesh vertices from [0, 1] range to bbox
  const meshVertices = scaleTensor(mesh.vertices, [0, 1], bbox);

  return {
    vertices: meshVertices,
    faces: mesh.faces,
    numVertices: mesh.numVertices,
    numFaces: mesh.numFaces,
  };
}

// --- Helper: dispatch a single linear layer (for camera embedding) ---
function _dispatchLinear(device, encoder, pipelines, input, output, weight, bias, rows, inDim, outDim) {
  // Reuse the imageTokenizer's linear pipeline
  pipelines.imageTokenizer._dispatchLinear(encoder, input, output, weight, bias, rows, inDim, outDim);
}
