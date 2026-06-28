#!/usr/bin/env node
/**
 * Inference smoke test — loads page, waits for weights, uploads a test image,
 * clicks Generate, captures all console errors until inference finishes or fails.
 *
 * Writes a durable error report to /tmp/sf3d-inference-smoke-report.txt
 *
 * Usage: node tools/smoke_inference.mjs [--image path/to/image.png]
 */

import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import http from 'http';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const DEFAULT_IMAGE = path.join(process.env.HOME, '.local/state/gpu-greenroom/outputs/b4fe3aa9e629/input.png');
const REPORT_PATH = process.argv.includes('--report')
  ? process.argv[process.argv.indexOf('--report') + 1]
  : '/tmp/sf3d-inference-smoke-report.txt';

// Auto-start vite dev server if not running
async function ensureViteServer() {
  const ports = [5177, 5176, 5178];
  for (const port of ports) {
    try {
      await new Promise((resolve, reject) => {
        const req = http.get(`http://localhost:${port}/`, (res) => { res.resume(); resolve(port); });
        req.on('error', reject);
        req.setTimeout(1000, () => { req.destroy(); reject(new Error('timeout')); });
      });
      return { port, serverProcess: null };
    } catch {}
  }
  // Start vite
  console.log('Starting vite dev server...');
  const proc = spawn('npx', ['vite', '--port', '5177'], {
    cwd: path.dirname(new URL(import.meta.url).pathname).replace(/\/tools$/, ''),
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false,
  });
  // Wait for server to be ready
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Vite startup timeout')), 30000);
    proc.stdout.on('data', (data) => {
      const str = data.toString();
      if (str.includes('ready') || str.includes('Local:')) {
        clearTimeout(timeout);
        resolve();
      }
    });
    proc.on('error', (err) => { clearTimeout(timeout); reject(err); });
  });
  return { port: 5177, serverProcess: proc };
}

let serverProcess = null;

const imagePath = process.argv.includes('--image')
  ? process.argv[process.argv.indexOf('--image') + 1]
  : DEFAULT_IMAGE;

if (!fs.existsSync(imagePath)) {
  console.error(`Test image not found: ${imagePath}`);
  process.exit(1);
}

const { port, serverProcess: sp } = await ensureViteServer();
serverProcess = sp;
const URL = `http://localhost:${port}`;

console.log(`\n=== SF3D WebGPU Inference Smoke ===`);
console.log(`Image: ${imagePath}`);
console.log(`URL: ${URL}\n`);

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

const consoleMessages = [];
const errors = [];
const pageErrors = [];
let lastStatus = '';

page.on('console', (msg) => {
  const type = msg.type();
  const text = msg.text();
  consoleMessages.push({ type, text, ts: Date.now() });

  if (type === 'error') {
    errors.push(text);
    console.log(`[ERR] ${text.slice(0, 300)}`);
  } else if (type === 'warning') {
    console.log(`[WRN] ${text.slice(0, 200)}`);
  } else {
    // Only log non-spammy messages
    if (!text.startsWith('Loading weights...')) {
      console.log(`[LOG] ${text.slice(0, 200)}`);
    }
  }
});

page.on('pageerror', (err) => {
  pageErrors.push({ message: err.message, stack: err.stack });
  console.log(`[PAGEERR] ${err.message.slice(0, 500)}`);
});

async function getStatus() {
  return page.$eval('#status', el => el.textContent).catch(() => '(unknown)');
}

async function waitForStatus(match, timeoutMs = 300000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const s = await getStatus();
    if (s !== lastStatus) {
      lastStatus = s;
      console.log(`[STATUS] ${s}`);
    }
    if (s.includes(match)) return s;
    if (s.startsWith('Error:')) throw new Error(`Status error: ${s}`);
    await new Promise(r => setTimeout(r, 500));
  }
  throw new Error(`Timeout waiting for status containing "${match}"`);
}

try {
  console.log('Navigating...');
  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 30000 });

  // Wait for model to be ready
  console.log('Waiting for weight loading and pipeline init...');
  await waitForStatus('Ready', 120000);

  // Upload test image via file input
  console.log(`\nUploading test image: ${imagePath}`);
  const fileInput = await page.$('#file-input');
  await fileInput.uploadFile(imagePath);

  // Wait for image to load
  await new Promise(r => setTimeout(r, 1000));
  console.log('Image uploaded.');

  // Click Generate
  console.log('Clicking "Generate 3D Mesh"...\n');
  await page.click('#run-btn');

  // Wait for inference to complete or fail (up to 5 min)
  const finalStatus = await waitForStatus('Done in', 300000).catch(async (err) => {
    // If it didn't finish cleanly, get whatever status we have
    const s = await getStatus();
    console.log(`\n[TIMEOUT/ERROR] Final status: ${s}`);
    return s;
  });

  console.log(`\nFinal status: ${finalStatus}`);

  // Screenshot
  await page.screenshot({ path: '/tmp/sf3d-inference-smoke.png', fullPage: true });
  console.log('Screenshot: /tmp/sf3d-inference-smoke.png');

  // Extract density array for comparison
  try {
    const density = await page.evaluate(() => {
      if (window._lastDensity) return Array.from(window._lastDensity);
      return null;
    });
    if (density) {
      const densityBuf = Buffer.from(new Float32Array(density).buffer);
      const densityPath = path.join(path.dirname(REPORT_PATH), 'webgpu_density.bin');
      fs.writeFileSync(densityPath, densityBuf);
      console.log(`Saved WebGPU density: ${density.length} values to ${densityPath}`);
    }
  } catch (e) {
    console.log(`Could not extract density: ${e.message}`);
  }

} catch (err) {
  console.error(`\nSmoke failed: ${err.message}`);
  await page.screenshot({ path: '/tmp/sf3d-inference-smoke.png', fullPage: true }).catch(() => {});
} finally {
  await browser.close();
}

// Write durable report
const report = [];
report.push(`=== SF3D Inference Smoke Report ===`);
report.push(`Date: ${new Date().toISOString()}`);
report.push(`Image: ${imagePath}`);
report.push(`Total console messages: ${consoleMessages.length}`);
report.push(`Errors: ${errors.length}`);
report.push(`Page errors: ${pageErrors.length}`);
report.push(`Last status: ${lastStatus}`);
report.push('');

if (pageErrors.length > 0) {
  report.push(`=== Page Errors (uncaught exceptions) ===`);
  for (const e of pageErrors) {
    report.push(e.message);
    if (e.stack) report.push(e.stack.split('\n').slice(0, 5).join('\n'));
    report.push('');
  }
}

if (errors.length > 0) {
  report.push(`=== Console Errors (first 50) ===`);
  // Deduplicate
  const seen = new Set();
  let count = 0;
  for (const e of errors) {
    const key = e.slice(0, 200);
    if (seen.has(key)) continue;
    seen.add(key);
    report.push(e.slice(0, 1000));
    report.push('');
    count++;
    if (count >= 50) {
      report.push(`... (${errors.length - count} more errors, ${errors.length} total)`);
      break;
    }
  }
}

const reportText = report.join('\n');
fs.writeFileSync(REPORT_PATH, reportText);
console.log(`\nReport written to: ${REPORT_PATH}`);
console.log(`\n${reportText}`);

// Clean up vite server if we started it
if (serverProcess) {
  serverProcess.kill();
}

process.exit(errors.length > 0 || pageErrors.length > 0 ? 1 : 0);
