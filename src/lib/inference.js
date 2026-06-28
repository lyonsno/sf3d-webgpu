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
  let imageBuf;
  if (typeof imageElement === 'string' && imageElement.endsWith('.bin')) {
    // Test mode: load PyTorch-preprocessed image directly
    const resp = await fetch(imageElement);
    const buf = await resp.arrayBuffer();
    imageBuf = createStorageBuffer(device, new Float32Array(buf));
    console.log(`Loaded pre-processed image: ${new Float32Array(buf).length} floats`);
  } else {
    const imageData = preprocessImage(imageElement,
      imageElement.naturalWidth || imageElement.width,
      imageElement.naturalHeight || imageElement.height);
    imageBuf = createStorageBuffer(device, imageData);
  }

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
  let dinov2Result;
  // Check for bypass: load PyTorch DINOv2 output directly
  try {
    const bypassResp = await fetch('test_dinov2_output.bin');
    if (bypassResp.ok) {
      const bypassBuf = await bypassResp.arrayBuffer();
      const bypassData = new Float32Array(bypassBuf);
      const N = bypassData.length / 1024;
      console.log(`BYPASS: Loaded PyTorch DINOv2 output: ${N} tokens, [0,:5]=[${Array.from(bypassData.slice(0,5)).map(v=>v.toFixed(4)).join(',')}]`);
      dinov2Result = {
        tokensBuf: createStorageBuffer(device, bypassData),
        N: Math.floor(N),
        tokenH: 36,
        tokenW: 36,
      };
    } else {
      throw new Error('no bypass file');
    }
  } catch {
    const encoder2 = device.createCommandEncoder();
    dinov2Result = pipelines.imageTokenizer.encode(
      encoder2, imageBuf, cameraEmbedBuf, weights.imageTokenizer);
    device.queue.submit([encoder2.finish()]);
  }

  // Diagnostic: DINOv2 output
  {
    const d2Sample = await readBuffer(device, dinov2Result.tokensBuf, dinov2Result.N * 1024 * 4);
    let min = Infinity, max = -Infinity, nan = 0, zero = 0;
    for (let i = 0; i < d2Sample.length; i++) {
      if (isNaN(d2Sample[i])) { nan++; continue; }
      if (d2Sample[i] === 0) zero++;
      if (d2Sample[i] < min) min = d2Sample[i];
      if (d2Sample[i] > max) max = d2Sample[i];
    }
    console.log(`DINOv2 output diagnostic (first 1024 f32): min=${min}, max=${max}, NaN=${nan}, zero=${zero}/${d2Sample.length}`);
  }

  // Diagnostic: camera embedding
  {
    const camSample = await readBuffer(device, cameraEmbedBuf, 768 * 4);
    let min = Infinity, max = -Infinity, nan = 0, zero = 0;
    for (let i = 0; i < camSample.length; i++) {
      if (isNaN(camSample[i])) { nan++; continue; }
      if (camSample[i] === 0) zero++;
      if (camSample[i] < min) min = camSample[i];
      if (camSample[i] > max) max = camSample[i];
    }
    console.log(`Camera embedding diagnostic: min=${min}, max=${max}, NaN=${nan}, zero=${zero}/${camSample.length}`);
  }

  // 4. Two-stream backbone (GPU)
  report('Running two-stream backbone...');

  // Rearrange tokenizer embeddings from [3, 1024, 96, 96] to [1024, 27648]
  // PyTorch does: rearrange("Np Ct Hp Wp -> Ct (Np Hp Wp)")
  // Source [p,c,h,w] at p*C*H*W + c*H*W + h*W + w
  // Dest [c, p*H*W + h*W + w] at c*3*H*W + p*H*W + h*W + w
  if (!weights.backbone._rearrangedEmbeddings) {
    const C = 1024, Np = 3, H = 96, W = 96;
    const total = Np * C * H * W;
    const rearrangeEncoder = device.createCommandEncoder();

    if (!pipelines._rearrangePipeline) {
      pipelines._rearrangePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: device.createShaderModule({
            code: `
              struct P { Np: u32, C: u32, H: u32, W: u32, numWgX: u32 }
              @group(0) @binding(0) var<uniform> p: P;
              @group(0) @binding(1) var<storage, read> src: array<f32>;
              @group(0) @binding(2) var<storage, read_write> dst: array<f32>;
              @compute @workgroup_size(256)
              fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
                let idx = (wgid.x + wgid.y * p.numWgX) * 256u + lid.x;
                let total = p.Np * p.C * p.H * p.W;
                if (idx >= total) { return; }
                // idx iterates over destination [C, Np*H*W]
                let c = idx / (p.Np * p.H * p.W);
                let s = idx % (p.Np * p.H * p.W);
                let plane = s / (p.H * p.W);
                let hw = s % (p.H * p.W);
                // source layout: [Np, C, H, W]
                let srcIdx = plane * p.C * p.H * p.W + c * p.H * p.W + hw;
                dst[idx] = src[srcIdx];
              }
            `,
          }),
          entryPoint: 'main',
        },
      });
    }

    const totalWG = Math.ceil(total / 256);
    const wgX = Math.min(totalWG, 65535);
    const wgY = Math.ceil(totalWG / 65535);
    const params = device.createBuffer({
      size: 20, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, mappedAtCreation: true,
    });
    new Uint32Array(params.getMappedRange()).set([Np, C, H, W, wgX]);
    params.unmap();

    const rearrangedBuf = createEmptyBuffer(device, total * 4);
    const bg = device.createBindGroup({
      layout: pipelines._rearrangePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: weights.tokenizer.embeddings } },
        { binding: 2, resource: { buffer: rearrangedBuf } },
      ],
    });
    const pass = rearrangeEncoder.beginComputePass();
    pass.setPipeline(pipelines._rearrangePipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
    device.queue.submit([rearrangeEncoder.finish()]);

    weights.backbone._rearrangedEmbeddings = rearrangedBuf;
  }
  weights.backbone.tokenizer_embeddings_buf = weights.backbone._rearrangedEmbeddings;

  // Diagnostic: check tokenizer embeddings
  {
    const embSample = await readBuffer(device, weights.tokenizer.embeddings, Math.min(4096, 28311552 * 4));
    let min = Infinity, max = -Infinity, nan = 0, zero = 0;
    for (let i = 0; i < embSample.length; i++) {
      if (isNaN(embSample[i])) { nan++; continue; }
      if (embSample[i] === 0) zero++;
      if (embSample[i] < min) min = embSample[i];
      if (embSample[i] > max) max = embSample[i];
    }
    console.log(`Tokenizer embeddings diagnostic: min=${min}, max=${max}, NaN=${nan}, zero=${zero}/${embSample.length}`);
  }

  // Push error scope to catch validation errors during backbone dispatch
  device.pushErrorScope('validation');

  const encoder3 = device.createCommandEncoder();

  const backboneResult = pipelines.twoStream.forward(
    encoder3, dinov2Result.tokensBuf, dinov2Result.N, weights.backbone);
  device.queue.submit([encoder3.finish()]);

  const bbError = await device.popErrorScope();
  if (bbError) {
    console.error(`Backbone validation error: ${bbError.message}`);
  } else {
    console.log('Backbone: no validation errors');
  }

  // DINOv2 point check
  {
    const d2Full = await readBuffer(device, dinov2Result.tokensBuf, dinov2Result.N * 1024 * 4);
    // PyTorch ref: DINOv2[0,:5] (CLS) = [0.234, 0.098, -1.094, 0.157, 0.608]
    console.log(`DINOv2[0,:5] (CLS): [${Array.from(d2Full.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`DINOv2[1,:5] (patch0): [${Array.from(d2Full.slice(1024, 1024+5)).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`DINOv2[100,:5]: [${Array.from(d2Full.slice(100*1024, 100*1024+5)).map(v => v.toFixed(4)).join(', ')}]`);
  }

  // Point-check backbone inputs against PyTorch reference
  if (pipelines.twoStream._diagnosticBuffers) {
    const triProj = pipelines.twoStream._diagnosticBuffers['triProjBuf'];
    const latent = pipelines.twoStream._diagnosticBuffers['latentBuf'];
    if (triProj) {
      const tp = await readBuffer(device, triProj, Math.min(triProj.size, 27648 * 1024 * 4));
      // PyTorch ref: triProjBuf[0, :5] = [-0.425, -0.179, -0.134, 0.484, 0.449]
      console.log(`triProjBuf[0,:5]: [${Array.from(tp.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}]`);
      console.log(`triProjBuf[100,:5]: [${Array.from(tp.slice(100*1024, 100*1024+5)).map(v => v.toFixed(4)).join(', ')}]`);
      console.log(`triProjBuf[9216,:5]: [${Array.from(tp.slice(9216*1024, 9216*1024+5)).map(v => v.toFixed(4)).join(', ')}]`);
    }
    if (latent) {
      const lt = await readBuffer(device, latent, Math.min(latent.size, 3089 * 1024 * 4));
      // PyTorch ref: latentBuf[0, :5] = [-0.778, -0.872, 0.427, 1.171, 1.312]
      console.log(`latentBuf[0,:5]: [${Array.from(lt.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}]`);
      console.log(`latentBuf[1297,:5]: [${Array.from(lt.slice(1297*1024, 1297*1024+5)).map(v => v.toFixed(4)).join(', ')}]`);
    }
  }

  // Stage-by-stage backbone diagnostics (compare with PyTorch reference)
  // PyTorch reference (chair image):
  //   after GroupNorm:      min=-2.50, max=2.51, std=0.45
  //   after proj_triplane:  min=-6.64, max=6.94, std=0.43
  //   block 0 triplane:     min=-7.64, max=9.77, std=0.59
  //   block 3 triplane:     min=-10.09, max=16.46, std=0.85
  //   proj_out (permuted):  min=-5.68, max=5.44, std=0.36
  //   final (with residual): min=-5.67, max=5.45, std=0.36

  // 5. PixelShuffle post-processing (GPU)
  report('Running post-processor...');
  const encoder4 = device.createCommandEncoder();
  const triplaneResult = dispatchPostProcessor(
    device, encoder4, backboneResult.buffer, weights.postProcessor);
  device.queue.submit([encoder4.finish()]);

  // Ensure GPU work is done before reading
  await device.queue.onSubmittedWorkDone();

  // Diagnostic: check backbone output
  {
    const bbTotal = backboneResult.C * backboneResult.N * 4;
    const bbSample = await readBuffer(device, backboneResult.buffer, bbTotal);
    let min = Infinity, max = -Infinity, nan = 0, zero = 0;
    for (let i = 0; i < bbSample.length; i++) {
      if (isNaN(bbSample[i])) { nan++; continue; }
      if (bbSample[i] === 0) zero++;
      if (bbSample[i] < min) min = bbSample[i];
      if (bbSample[i] > max) max = bbSample[i];
    }
    console.log(`Backbone output diagnostic (first ${bbSample.length} f32): min=${min}, max=${max}, NaN=${nan}, zero=${zero}/${bbSample.length}`);
    // Point comparison with PyTorch
    // PyTorch backbone[0, 0:5] = [-0.0785, -0.0825, -0.1648, -0.0960, -0.2965]
    // PyTorch backbone[0, 48*96+48] = 0.1221
    console.log(`Backbone [0,0:5]: [${Array.from(bbSample.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`Backbone [0, 4656] (ch0 plane0 center): ${bbSample[4656].toFixed(6)}`);
    console.log(`Backbone [0, 9216:9221] (ch0 plane1 first 5): [${Array.from(bbSample.slice(9216, 9221)).map(v => v.toFixed(4)).join(', ')}]`);
  }

  // Diagnostic: check triplane features after post-processor
  {
    const tpSample = await readBuffer(device, triplaneResult.buffer, Math.min(4096, triplaneResult.C * triplaneResult.H * triplaneResult.W * 4));
    let min = Infinity, max = -Infinity, nan = 0, zero = 0;
    for (let i = 0; i < tpSample.length; i++) {
      if (isNaN(tpSample[i])) { nan++; continue; }
      if (tpSample[i] === 0) zero++;
      if (tpSample[i] < min) min = tpSample[i];
      if (tpSample[i] > max) max = tpSample[i];
    }
    console.log(`Triplane features diagnostic (first 1024 f32): min=${min}, max=${max}, NaN=${nan}, zero=${zero}/${tpSample.length}`);
  }

  // Compare triplane features with PyTorch reference
  // PyTorch reference (chair image):
  //   Plane 0 ch0 center row first 20: [13.04, 13.33, 16.92, 17.89, 19.20, ...]
  //   Plane 0: min=-28.64, max=34.45, std=10.53
  //   Plane 1: min=-36.33, max=39.11, std=12.02
  //   Plane 2: min=-27.64, max=15.02, std=4.44
  {
    // Read the full triplane buffer
    const fullTP = await readBuffer(device, triplaneResult.buffer, 3 * 40 * 384 * 384 * 4);
    const tpArr = fullTP;

    // Plane 0, ch 0, center row
    const rowStart = 192 * 384;
    console.log(`WebGPU plane0 ch0 center row (first 20): [${Array.from(tpArr.slice(rowStart, rowStart + 20)).map(v => v.toFixed(2)).join(', ')}]`);

    // Per-plane stats
    const planeSize = 40 * 384 * 384;
    for (let p = 0; p < 3; p++) {
      let pmin = Infinity, pmax = -Infinity, psum = 0, psum2 = 0;
      for (let i = p * planeSize; i < (p + 1) * planeSize; i++) {
        const v = tpArr[i];
        if (v < pmin) pmin = v;
        if (v > pmax) pmax = v;
        psum += v;
        psum2 += v * v;
      }
      const mean = psum / planeSize;
      const std = Math.sqrt(psum2 / planeSize - mean * mean);
      console.log(`WebGPU Plane ${p}: min=${pmin.toFixed(2)}, max=${pmax.toFixed(2)}, mean=${mean.toFixed(4)}, std=${std.toFixed(4)}`);
    }
  }

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

  // Diagnostic: check raw density values
  let minD = Infinity, maxD = -Infinity, nanCount = 0, zeroCount = 0;
  for (let i = 0; i < sdf.length; i++) {
    if (isNaN(sdf[i])) { nanCount++; continue; }
    if (sdf[i] === 0) zeroCount++;
    if (sdf[i] < minD) minD = sdf[i];
    if (sdf[i] > maxD) maxD = sdf[i];
  }
  console.log(`SDF diagnostic: min=${minD}, max=${maxD}, NaN=${nanCount}, zero=${zeroCount}, total=${sdf.length}`);

  // Point-check: compare with PyTorch at known inside vertices
  const checkIndices = [136813, 150448, 150449, 150452];
  console.log('Density point-check (PyTorch ref: 11.32, 13.83, 15.00, 11.35):');
  for (const idx of checkIndices) {
    console.log(`  vertex ${idx}: density=${sdf[idx].toFixed(4)}`);
  }

  // Find our inside vertices to compare spatial location
  const insideIndices = [];
  for (let i = 0; i < sdf.length; i++) {
    if (sdf[i] > 10.0) insideIndices.push(i);
  }
  console.log(`WebGPU inside vertices (density > 10): ${insideIndices.length}`);
  if (insideIndices.length > 0) {
    // Show first 5 inside vertices with their positions
    for (let j = 0; j < Math.min(5, insideIndices.length); j++) {
      const idx = insideIndices[j];
      const px = gridPositions[idx * 3], py = gridPositions[idx * 3 + 1], pz = gridPositions[idx * 3 + 2];
      console.log(`  inside vertex ${idx}: pos=[${px.toFixed(4)}, ${py.toFixed(4)}, ${pz.toFixed(4)}], density=${sdf[idx].toFixed(4)}`);
    }
  }

  // Save raw density for comparison
  if (typeof window !== 'undefined') {
    window._lastDensity = new Float32Array(sdf);
  }

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
