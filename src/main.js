/**
 * SF3D WebGPU — Main entry point.
 *
 * Orchestrates:
 *   1. WebGPU initialization
 *   2. Weight loading
 *   3. Image input handling
 *   4. Inference pipeline dispatch
 *   5. Texture baking + GLB export
 */

import { initGPU } from './lib/gpu.js';
import { loadWeights } from './lib/weights.js';
import { initPipelines, runInference } from './lib/inference.js';
import { unwrapUV, rasterizeUV, bakeTexture, exportGLB } from './lib/texture_baker.js';
import { estimateMaterials } from './lib/clip_estimator.js';

const statusEl = document.getElementById('status');
const progressFill = document.getElementById('progress-fill');
const runBtn = document.getElementById('run-btn');
const downloadBtn = document.getElementById('download-btn');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const demoBtn = document.getElementById('demo-btn');

let device = null;
let pipelines = null;
let weights = null;
let inputImage = null;
let lastGLB = null;

function setStatus(msg) {
  statusEl.textContent = msg;
  console.log(msg);
}

function setProgress(pct) {
  progressFill.style.width = `${Math.round(pct)}%`;
}

async function init() {
  try {
    // 1. Init WebGPU
    const gpu = await initGPU();
    device = gpu.device;

    const adapterInfo = gpu.adapter.info || (gpu.adapter.requestAdapterInfo ? await gpu.adapter.requestAdapterInfo() : {});
    setStatus(`WebGPU ready: ${adapterInfo.description || adapterInfo.device || 'GPU'}. Loading weights...`);
    console.log('Adapter limits:', {
      maxBufferSize: gpu.adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: gpu.adapter.limits.maxStorageBufferBindingSize,
    });

    // 2. Load weights
    setProgress(0);
    const t0 = performance.now();
    weights = await loadWeights(device, 'weights.bin', (received, total) => {
      if (total > 0) {
        const pct = (received / total) * 100;
        setProgress(pct);
        setStatus(`Loading weights... ${(received / 1024 / 1024).toFixed(0)} / ${(total / 1024 / 1024).toFixed(0)} MB`);
      }
    });
    const loadTime = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Weights loaded in ${loadTime}s. Initializing pipelines...`);
    setProgress(100);

    // 3. Init compute pipelines
    pipelines = initPipelines(device);
    setStatus(`Ready. Drop an image to generate a 3D mesh.`);

    // Expose for test harness
    window._sf3d_device = device;
    window._sf3d_weights = weights;
    window._sf3d_pipelines = pipelines;
    demoBtn.disabled = false;

  } catch (e) {
    setStatus(`Error: ${e.message}`);
    console.error(e);
  }
}

// --- Image input ---
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      inputImage = img;
      dropZone.innerHTML = '';
      dropZone.appendChild(img);
      setStatus('Image loaded. Click "Generate 3D Mesh" to run.');
      runBtn.disabled = !weights;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// --- Demo image ---
demoBtn.addEventListener('click', async () => {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    inputImage = img;
    dropZone.innerHTML = '';
    dropZone.appendChild(img);
    setStatus('Demo image loaded. Click "Generate 3D Mesh" to run.');
    runBtn.disabled = !weights;
  };
  img.src = 'demo_chair.png';
});

// --- Inference + texture baking ---
runBtn.addEventListener('click', async () => {
  if (!inputImage || !weights || !pipelines) return;
  runBtn.disabled = true;
  downloadBtn.disabled = true;
  lastGLB = null;
  setProgress(0);

  try {
    const t0 = performance.now();

    // Step 1: Run inference to get untextured mesh + triplane data
    const meshResult = await runInference(device, pipelines, weights, inputImage, (msg) => {
      setStatus(msg);
    });

    const meshTime = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Mesh in ${meshTime}s (${meshResult.numVertices} verts). UV unwrapping...`);

    // Step 2: CLIP material estimation (runs on GPU, uses input image)
    // Get masked float RGBA pixels for CLIP input
    const clipCanvas = document.createElement('canvas');
    const clipSrcW = inputImage.naturalWidth || inputImage.width;
    const clipSrcH = inputImage.naturalHeight || inputImage.height;
    clipCanvas.width = clipSrcW;
    clipCanvas.height = clipSrcH;
    const clipCtx = clipCanvas.getContext('2d');
    clipCtx.drawImage(inputImage, 0, 0);
    const clipRaw = clipCtx.getImageData(0, 0, clipSrcW, clipSrcH).data;
    const clipPixels = new Float32Array(clipSrcW * clipSrcH * 4);
    for (let i = 0; i < clipRaw.length; i++) clipPixels[i] = clipRaw[i] / 255.0;
    // Apply alpha blending with grey background (matching PyTorch: rgb_cond * mask_cond)
    for (let i = 0; i < clipSrcW * clipSrcH; i++) {
      const a = clipPixels[i * 4 + 3];
      clipPixels[i * 4]     = clipPixels[i * 4] * a + 0.5 * (1 - a);
      clipPixels[i * 4 + 1] = clipPixels[i * 4 + 1] * a + 0.5 * (1 - a);
      clipPixels[i * 4 + 2] = clipPixels[i * 4 + 2] * a + 0.5 * (1 - a);
    }
    const { roughness, metallic } = await estimateMaterials(device, clipPixels, clipSrcW, clipSrcH, weights);

    // Step 3: UV unwrap (synchronous cube projection)
    const uvResult = unwrapUV(
      meshResult.vertices, meshResult.faces,
      meshResult.numVertices, meshResult.numFaces);
    setStatus(`UV unwrap done (${uvResult.newNumVertices} verts). Rasterizing UV space...`);

    // Step 4: Rasterize UV space to get 3D positions per texel
    const texResolution = 1024;
    const rasterResult = rasterizeUV(
      uvResult.uvs, uvResult.newVertices, uvResult.newFaces,
      uvResult.newNumFaces, texResolution, uvResult.faceAssignment);
    setStatus(`Rasterized ${rasterResult.mask.reduce((a, b) => a + b, 0)} texels. Baking texture...`);

    // Step 5: Bake textures (GPU triplane query + features/perturb_normal decoder)
    const bakeResult = await bakeTexture(
      device, meshResult._triplaneDecoder, meshResult._triplanesBuf,
      meshResult._decoderWeights, rasterResult.positions3D, rasterResult.mask,
      rasterResult.tbnData, texResolution);
    setStatus('Textures baked. Building GLB...');

    // Step 6: Export as GLB with CLIP-estimated materials
    lastGLB = await exportGLB(
      uvResult.newVertices, uvResult.newNormals, uvResult.newFaces, uvResult.uvs,
      bakeResult.albedo, bakeResult.normalMap,
      uvResult.newNumVertices, uvResult.newNumFaces, texResolution,
      roughness, metallic);

    const totalTime = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Done in ${totalTime}s: ${meshResult.numVertices} vertices, ${meshResult.numFaces} faces, textured GLB ready`);
    setProgress(100);
    downloadBtn.disabled = false;
    runBtn.disabled = false;

    window._lastMeshResult = meshResult;
    window._lastGLB = lastGLB;

  } catch (err) {
    console.error('Inference failed:', err);
    setStatus(`Error: ${err.message}`);
    runBtn.disabled = false;
  }
});

// --- Download GLB ---
downloadBtn.addEventListener('click', () => {
  if (!lastGLB) return;

  const blob = new Blob([lastGLB], { type: 'model/gltf-binary' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'sf3d_mesh.glb';
  a.click();
  URL.revokeObjectURL(url);
});

// --- Boot ---
init();
