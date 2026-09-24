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

import { createSf3dProducer } from './lib/sf3d_producer.js';

const statusEl = document.getElementById('status');
const progressFill = document.getElementById('progress-fill');
const runBtn = document.getElementById('run-btn');
const downloadBtn = document.getElementById('download-btn');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const demoBtn = document.getElementById('demo-btn');

let producer = null;
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
    // The producer owns WebGPU init, weight loading, pipelines and the five
    // long-lived CPU offload workers. A Kaminos host composes the same
    // createSf3dProducer with its own { device, adapter } injected.
    setProgress(0);
    const t0 = performance.now();
    let announced = false;
    producer = await createSf3dProducer({
      onPhase: typeof window.__sf3dParentPhase === 'function'
        ? (event) => window.__sf3dParentPhase(event)
        : null,
      onWeightsProgress: (received, total) => {
        if (!announced) { announced = true; }
        if (total > 0) {
          const pct = (received / total) * 100;
          setProgress(pct);
          setStatus(`Loading weights... ${(received / 1024 / 1024).toFixed(0)} / ${(total / 1024 / 1024).toFixed(0)} MB`);
        }
      },
    });
    const info = producer.adapterInfo;
    const loadTime = ((performance.now() - t0) / 1000).toFixed(1);
    console.log(`WebGPU ready: ${info.description || info.device || 'GPU'}; weights + pipelines in ${loadTime}s`);
    console.log('Adapter limits:', {
      maxBufferSize: producer.adapterLimits.maxBufferSize,
      maxStorageBufferBindingSize: producer.adapterLimits.maxStorageBufferBindingSize,
    });
    setProgress(100);
    setStatus(`Ready. Drop an image to generate a 3D mesh.`);

    // Expose for test harnesses (witness/parity smokes drive these directly).
    window._sf3d_producer = producer;
    window._sf3d_device = producer.device;
    window._sf3d_weights = producer.weights;
    window._sf3d_pipelines = producer.pipelines;
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
      runBtn.disabled = !producer;
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
    runBtn.disabled = !producer;
  };
  img.src = 'demo_chair.png';
});

// --- Inference + texture baking ---
runBtn.addEventListener('click', async () => {
  if (!inputImage || !producer) return;
  runBtn.disabled = true;
  downloadBtn.disabled = true;
  lastGLB = null;
  setProgress(0);

  try {
    const t0 = performance.now();
    // The product route through the producer: every proven foreground-liveness
    // mechanism on by default, one kit foreground-opportunity interlock per
    // run, and the validated route receipt built by the producer.
    const result = await producer.run(inputImage, { onProgress: setStatus });
    const meshResult = {
      numVertices: result.numVertices,
      numFaces: result.numFaces,
      vertices: result.vertices,
      faces: result.faces,
    };
    lastGLB = result.glb;
    window._lastRouteOptions = result.routeOptions;
    window._lastOffloads = result.offloads;
    window._lastCooperativeReports = result.cooperativeReports;
    window._lastStageSpans = result.stageSpans;
    window._lastDensity = result.sdf;
    window._lastForegroundOpportunityReport = result.foregroundOpportunityReport;

    const totalTime = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Done in ${totalTime}s: ${meshResult.numVertices} vertices, ${meshResult.numFaces} faces, textured GLB ready`);
    setProgress(100);
    downloadBtn.disabled = false;
    runBtn.disabled = false;

    if (!result.receiptValidation.ok) {
      console.warn('Route receipt validation failed:', result.receiptValidation.errors);
    }
    window._lastRouteReceipt = result.receipt;
    console.log('Route receipt emitted:', result.receipt.requestedRouteId);

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
