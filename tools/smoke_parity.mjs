#!/usr/bin/env node
/**
 * Dump per-stage WebGPU intermediate tensors for numerical parity comparison
 * against PyTorch reference (tools/dump_parity_reference.py).
 *
 * Usage: node tools/smoke_parity.mjs [--reference /tmp/sf3d-parity-ref]
 */
import puppeteer from 'puppeteer-core';
import { execSync, spawn } from 'child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 5177;
const IMAGE = process.env.IMAGE ||
  `${process.env.HOME}/.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png`;
const REF_DIR = process.argv.includes('--reference')
  ? process.argv[process.argv.indexOf('--reference') + 1]
  : '/tmp/sf3d-parity-ref';
const OUT_DIR = '/tmp/sf3d-parity-webgpu';

mkdirSync(OUT_DIR, { recursive: true });

// Start vite
const vite = spawn('npx', ['vite', '--port', String(PORT)], {
  stdio: ['pipe', 'pipe', 'pipe'],
  cwd: process.cwd(),
});
await new Promise(r => setTimeout(r, 3000));

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: false,
  args: [
    '--enable-features=Vulkan,UseSkiaRenderer',
    '--enable-unsafe-webgpu',
    '--disable-dawn-features=disallow_unsafe_apis',
    '--no-first-run',
    '--no-default-browser-check',
  ],
});

const page = await browser.newPage();
const logs = [];
page.on('console', msg => {
  const text = msg.text();
  logs.push(text);
  if (text.startsWith('PARITY:')) console.log(text);
});

try {
  console.log('=== SF3D Parity Verification ===');
  console.log(`Image: ${IMAGE}`);
  console.log(`Reference: ${REF_DIR}`);

  await page.goto(`http://localhost:${PORT}/`, { waitUntil: 'networkidle0', timeout: 60000 });

  // Wait for ready
  await page.waitForFunction(() => window._sf3d_weights, { timeout: 120000 });
  console.log('Weights loaded. Uploading image...');

  // Upload image
  const imgData = readFileSync(IMAGE);
  const imgB64 = imgData.toString('base64');
  const mime = IMAGE.endsWith('.png') ? 'image/png' : 'image/jpeg';
  await page.evaluate(async (b64, mimeType) => {
    const res = await fetch(`data:${mimeType};base64,${b64}`);
    const blob = await res.blob();
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = URL.createObjectURL(blob);
    });
    window._testImage = img;
    // Trigger the UI
    const dropZone = document.getElementById('drop-zone');
    dropZone.innerHTML = '';
    dropZone.appendChild(img);
    window._sf3d_inputImage = img;
  }, imgB64, mime);

  console.log('Running inference with parity dumps...');

  // Run inference and capture intermediates
  const result = await page.evaluate(async () => {
    const { initPipelines, runInference } = await import('/src/lib/inference.js');
    const { createStorageBuffer, readBuffer } = await import('/src/lib/gpu.js');

    const device = window._sf3d_device;
    const weights = window._sf3d_weights;
    const pipelines = window._sf3d_pipelines;
    const img = window._testImage;

    const meshResult = await runInference(device, pipelines, weights, img, (msg) => {
      console.log(`PARITY: ${msg}`);
    });

    // Read back triplane buffer (post-processed scene codes: 3 × 40 × 384 × 384)
    const triplaneSize = 3 * 40 * 384 * 384;
    const triplaneData = await readBuffer(device, meshResult._triplanesBuf,
      triplaneSize * 4);

    // Read stage timings
    const timings = meshResult._stageTimings;

    // Compute triplane stats efficiently
    let tMin = Infinity, tMax = -Infinity, tSum = 0;
    for (let i = 0; i < triplaneData.length; i++) {
      const v = triplaneData[i];
      if (v < tMin) tMin = v;
      if (v > tMax) tMax = v;
      tSum += v;
    }

    return {
      numVertices: meshResult.numVertices,
      numFaces: meshResult.numFaces,
      timings,
      triplaneFirst8: Array.from(triplaneData.slice(0, 8)),
      triplaneStats: { min: tMin, max: tMax, mean: tSum / triplaneData.length },
      // Raw marching tet vertices (before GLB coordinate transform)
      firstVerts: Array.from(meshResult.vertices.slice(0, 15)),
    };
  });

  console.log(`\nPARITY: Mesh: ${result.numVertices} vertices, ${result.numFaces} faces`);
  console.log(`PARITY: Triplane first 8: ${result.triplaneFirst8.map(v => v.toFixed(6)).join(', ')}`);
  console.log(`PARITY: Triplane range: [${result.triplaneStats.min.toFixed(4)}, ${result.triplaneStats.max.toFixed(4)}]`);
  console.log(`PARITY: Stage timings: ${JSON.stringify(result.timings)}`);

  // Load PyTorch reference and compare
  if (existsSync(path.join(REF_DIR, 'summary.json'))) {
    const ref = JSON.parse(readFileSync(path.join(REF_DIR, 'summary.json'), 'utf-8'));

    console.log('\n=== Parity Comparison ===');

    // Mesh
    console.log(`\nMesh vertices:  WebGPU=${result.numVertices}  PyTorch=${ref.mesh.num_vertices}  diff=${result.numVertices - ref.mesh.num_vertices}`);
    console.log(`Mesh faces:     WebGPU=${result.numFaces}  PyTorch=${ref.mesh.num_faces}  diff=${result.numFaces - ref.mesh.num_faces}`);

    // Compare first 5 vertices
    if (ref.mesh.first_5_verts) {
      console.log('\nFirst 5 vertices comparison:');
      for (let i = 0; i < 5; i++) {
        const wx = result.firstVerts[i*3], wy = result.firstVerts[i*3+1], wz = result.firstVerts[i*3+2];
        const [px, py, pz] = ref.mesh.first_5_verts[i];
        const dx = Math.abs(wx - px), dy = Math.abs(wy - py), dz = Math.abs(wz - pz);
        console.log(`  v${i}: WebGPU=[${wx.toFixed(6)}, ${wy.toFixed(6)}, ${wz.toFixed(6)}]  PyTorch=[${px.toFixed(6)}, ${py.toFixed(6)}, ${pz.toFixed(6)}]  maxDiff=${Math.max(dx,dy,dz).toFixed(6)}`);
      }
    }

    // Scene codes / triplane
    if (ref.scene_codes) {
      console.log(`\nTriplane (scene codes):`);
      console.log(`  WebGPU range: [${result.triplaneStats.min.toFixed(4)}, ${result.triplaneStats.max.toFixed(4)}]`);
      console.log(`  PyTorch range: [${ref.scene_codes.min.toFixed(4)}, ${ref.scene_codes.max.toFixed(4)}]`);
      console.log(`  WebGPU first 8: ${result.triplaneFirst8.map(v => v.toFixed(6)).join(', ')}`);
      console.log(`  PyTorch first 8: ${ref.scene_codes.first_8.map(v => v.toFixed(6)).join(', ')}`);
    }

    // Density
    if (ref.density) {
      console.log(`\nDensity:`);
      console.log(`  PyTorch range: [${ref.density.min.toFixed(4)}, ${ref.density.max.toFixed(4)}]`);
      console.log(`  PyTorch positive: ${ref.density.num_positive}`);
      console.log(`  PyTorch mean: ${ref.density.mean.toFixed(6)}`);
    }

    // Materials
    if (ref.materials) {
      console.log(`\nMaterials:`);
      console.log(`  PyTorch roughness: ${ref.materials.roughness.toFixed(6)}`);
      console.log(`  PyTorch metallic: ${ref.materials.metallic.toFixed(6)}`);
    }

    // Write comparison report
    const report = {
      webgpu: {
        num_vertices: result.numVertices,
        num_faces: result.numFaces,
        triplane_range: [result.triplaneStats.min, result.triplaneStats.max],
        triplane_first_8: result.triplaneFirst8,
        first_5_verts: [],
        timings: result.timings,
      },
      pytorch: ref,
      comparison: {
        vertex_diff: result.numVertices - ref.mesh.num_vertices,
        face_diff: result.numFaces - ref.mesh.num_faces,
      },
    };
    for (let i = 0; i < 5; i++) {
      report.webgpu.first_5_verts.push([
        result.firstVerts[i*3], result.firstVerts[i*3+1], result.firstVerts[i*3+2]
      ]);
    }
    writeFileSync(path.join(OUT_DIR, 'parity_report.json'), JSON.stringify(report, null, 2));
    console.log(`\nReport written to ${OUT_DIR}/parity_report.json`);
  } else {
    console.log(`\nNo PyTorch reference found at ${REF_DIR}/summary.json — run dump_parity_reference.py first`);
  }

} catch (err) {
  console.error('Parity smoke failed:', err.message);
} finally {
  await browser.close();
  vite.kill();
}
