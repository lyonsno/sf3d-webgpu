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
import { initPipelines } from './lib/inference.js';
import { runFullPipelineToGlb } from './lib/full_pipeline.js';
import {
  createProductRouteOptions,
  createProductRouteWorkers,
  describeProductRouteOptions,
} from './lib/product_route.js';
import {
  createSf3dImageToMeshRouteReceipt,
  createStagedSubmitProfile,
  addStagedSubmitStage,
  createWebGpuBackendIdentity,
  SF3D_IMAGE_TO_MESH_ROUTE_ID,
  WEBGPU_INFERENCE_KIT_VERSION,
  validateRouteReceipt,
} from '@kaminos/webgpu-inference-kit';

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
let routeWorkers = null;
let _adapterInfo = null;
let _adapterLimits = null;
let _adapterFeatures = null;

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

    _adapterInfo = gpu.adapter.info || (gpu.adapter.requestAdapterInfo ? await gpu.adapter.requestAdapterInfo() : {});
    _adapterLimits = gpu.adapter.limits;
    _adapterFeatures = [...gpu.adapter.features];
    setStatus(`WebGPU ready: ${_adapterInfo.description || _adapterInfo.device || 'GPU'}. Loading weights...`);
    console.log('Adapter limits:', {
      maxBufferSize: _adapterLimits.maxBufferSize,
      maxStorageBufferBindingSize: _adapterLimits.maxStorageBufferBindingSize,
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
    // Long-lived CPU offload workers for the product route (preprocess, CLIP
    // prep, marching tet, UV unwrap, texture materialization).
    routeWorkers = createProductRouteWorkers();
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
    const REQUIRED_STAGES = [
      'image-preprocess', 'dinov2-tokenizer', 'two-stream-backbone',
      'triplane-decode', 'marching-tet', 'texture-bake', 'glb-export',
    ];
    const profile = createStagedSubmitProfile({
      route: SF3D_IMAGE_TO_MESH_ROUTE_ID,
      timingSource: 'performance-now-wall-clock',
      requiredStages: REQUIRED_STAGES,
    });

    const t0 = performance.now();

    // The product route: every proven foreground-liveness mechanism on by
    // default (cooperative GPU duties, decoder arena, five CPU offload workers)
    // — the same runFullPipelineToGlb the witness harnesses drive.
    const options = createProductRouteOptions({ workers: routeWorkers });
    const result = await runFullPipelineToGlb(device, pipelines, weights, inputImage, options, (msg) => {
      setStatus(msg);
    });
    const meshResult = {
      numVertices: result.numVertices,
      numFaces: result.numFaces,
      vertices: result.vertices,
      faces: result.faces,
    };

    // Stage timings for the route receipt: inference substages come back as
    // timings; the outer stages as absolute spans.
    const stageTimings = result.stageTimings || {};
    for (const name of ['image-preprocess', 'dinov2-tokenizer', 'two-stream-backbone', 'triplane-decode', 'marching-tet']) {
      addStagedSubmitStage(profile, { name, ms: stageTimings[name] || 0 });
    }
    const spanMs = (name) => {
      const span = (result.stageSpans || []).find(s => s.name === name);
      return span ? span.end - span.start : 0;
    };
    addStagedSubmitStage(profile, { name: 'texture-bake', ms: spanMs('texture-bake') });
    addStagedSubmitStage(profile, { name: 'glb-export', ms: spanMs('glb-export') });

    const texResolution = 1024;
    lastGLB = result.glb;
    window._lastRouteOptions = describeProductRouteOptions(options);
    window._lastOffloads = result.offloads;
    window._lastCooperativeReports = result.cooperativeReports;
    window._lastStageSpans = result.stageSpans;
    window._lastDensity = result.sdf;

    const totalTime = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Done in ${totalTime}s: ${meshResult.numVertices} vertices, ${meshResult.numFaces} faces, textured GLB ready`);
    setProgress(100);
    downloadBtn.disabled = false;
    runBtn.disabled = false;

    // Emit route receipt
    const backend = createWebGpuBackendIdentity({
      adapterName: _adapterInfo.description || _adapterInfo.device || 'unknown',
      browser: navigator.userAgent,
      requestedFeatures: _adapterFeatures,
      effectiveFeatures: _adapterFeatures,
      limits: {
        maxBufferSize: _adapterLimits.maxBufferSize,
        maxStorageBufferBindingSize: _adapterLimits.maxStorageBufferBindingSize,
        maxComputeInvocationsPerWorkgroup: _adapterLimits.maxComputeInvocationsPerWorkgroup,
      },
      timestampQuery: _adapterFeatures.includes('timestamp-query') ? 'available' : 'unavailable',
    });

    const receipt = createSf3dImageToMeshRouteReceipt({
      input: {
        artifactId: `source-image:${inputImage.naturalWidth || inputImage.width}x${inputImage.naturalHeight || inputImage.height}`,
        sha256: 'not-computed',
        shape: [inputImage.naturalHeight || inputImage.height, inputImage.naturalWidth || inputImage.width, 4],
      },
      outputs: {
        meshGlb: {
          artifactId: `mesh-glb:${meshResult.numVertices}v-${meshResult.numFaces}f`,
          sha256: 'not-computed',
          shape: [lastGLB.byteLength],
        },
        albedoTexture: {
          artifactId: `albedo-texture:${texResolution}`,
          sha256: 'not-computed',
          shape: [texResolution, texResolution, 4],
        },
        normalMap: {
          artifactId: `normal-map:${texResolution}`,
          sha256: 'not-computed',
          shape: [texResolution, texResolution, 4],
        },
      },
      backend,
      model: {
        revision: 'v1.0.0-webgpu',
        weightsHash: 'not-computed',
      },
      kernel: {
        kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
        profile: 'dinov2-two-stream-triplane-marching-tet-texture-bake',
        commit: typeof __COMMIT_HASH__ !== 'undefined' ? __COMMIT_HASH__ : 'dev',
      },
      profile,
    });

    const validation = validateRouteReceipt(receipt);
    if (!validation.ok) {
      console.warn('Route receipt validation failed:', validation.errors);
    }
    window._lastRouteReceipt = receipt;
    console.log('Route receipt emitted:', receipt.requestedRouteId);

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
