/**
 * SF3D WebGPU — Main entry point.
 *
 * Orchestrates:
 *   1. WebGPU initialization
 *   2. Weight loading
 *   3. Image input handling
 *   4. Inference pipeline dispatch
 *   5. GLB mesh export
 */

import { initGPU } from './lib/gpu.js';

const statusEl = document.getElementById('status');
const progressFill = document.getElementById('progress-fill');
const runBtn = document.getElementById('run-btn');
const downloadBtn = document.getElementById('download-btn');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

let device = null;
let weights = null;
let inputImage = null;

function setStatus(msg) {
  statusEl.textContent = msg;
  console.log(msg);
}

function setProgress(pct) {
  progressFill.style.width = `${pct}%`;
}

async function init() {
  try {
    const gpu = await initGPU();
    device = gpu.device;
    setStatus('WebGPU ready. Drop an image to begin.');

    // TODO: Load weights
    // weights = await loadWeights(device, '/weights.bin', (received, total) => {
    //   const pct = total ? (received / total * 100) : 0;
    //   setProgress(pct);
    //   setStatus(`Loading weights: ${(received / 1024 / 1024).toFixed(0)} MB`);
    // });
    // setStatus('Model loaded. Drop an image to generate.');
    // runBtn.disabled = false;
  } catch (e) {
    setStatus(`Error: ${e.message}`);
    console.error(e);
  }
}

// Image input handling
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
      // runBtn.disabled = !weights;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

runBtn.addEventListener('click', async () => {
  if (!inputImage || !weights) return;
  runBtn.disabled = true;
  setStatus('Running inference...');
  // TODO: Run inference pipeline
});

init();
