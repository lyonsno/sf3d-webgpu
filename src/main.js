/**
 * SF3D WebGPU — Main entry point.
 *
 * Orchestrates:
 *   1. WebGPU initialization
 *   2. Weight loading
 *   3. Image input handling
 *   4. Inference pipeline dispatch
 *   5. OBJ mesh export
 */

import { initGPU } from './lib/gpu.js';
import { loadWeights } from './lib/weights.js';
import { initPipelines, runInference } from './lib/inference.js';

const statusEl = document.getElementById('status');
const progressFill = document.getElementById('progress-fill');
const runBtn = document.getElementById('run-btn');
const downloadBtn = document.getElementById('download-btn');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

let device = null;
let pipelines = null;
let weights = null;
let inputImage = null;
let lastMesh = null;

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

    const adapterInfo = await gpu.adapter.requestAdapterInfo();
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

// --- Inference ---
runBtn.addEventListener('click', async () => {
  if (!inputImage || !weights || !pipelines) return;
  runBtn.disabled = true;
  downloadBtn.disabled = true;
  setProgress(0);

  try {
    const t0 = performance.now();
    lastMesh = await runInference(device, pipelines, weights, inputImage, (msg) => {
      setStatus(msg);
    });
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Done in ${dt}s: ${lastMesh.numVertices} vertices, ${lastMesh.numFaces} faces`);
    setProgress(100);
    downloadBtn.disabled = false;
    runBtn.disabled = false;
  } catch (err) {
    console.error('Inference failed:', err);
    setStatus(`Error: ${err.message}`);
    runBtn.disabled = false;
  }
});

// --- Download OBJ ---
downloadBtn.addEventListener('click', () => {
  if (!lastMesh) return;

  const { vertices, faces, numVertices, numFaces } = lastMesh;
  let obj = '# SF3D WebGPU mesh output\n';
  obj += `# ${numVertices} vertices, ${numFaces} faces\n\n`;

  for (let i = 0; i < numVertices; i++) {
    obj += `v ${vertices[i * 3].toFixed(6)} ${vertices[i * 3 + 1].toFixed(6)} ${vertices[i * 3 + 2].toFixed(6)}\n`;
  }
  obj += '\n';
  for (let i = 0; i < numFaces; i++) {
    obj += `f ${faces[i * 3] + 1} ${faces[i * 3 + 1] + 1} ${faces[i * 3 + 2] + 1}\n`;
  }

  const blob = new Blob([obj], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'sf3d_mesh.obj';
  a.click();
  URL.revokeObjectURL(url);
});

// --- Boot ---
init();
