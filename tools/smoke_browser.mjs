#!/usr/bin/env node
/**
 * Browser smoke test — launches Chrome, loads the SF3D WebGPU page,
 * captures console output, errors, and WebGPU shader compilation failures.
 *
 * Usage: node tools/smoke_browser.mjs [--url http://localhost:5177]
 */

import puppeteer from 'puppeteer-core';
import { execSync } from 'child_process';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const DEFAULT_URL = 'http://localhost:5177';
const WAIT_MS = parseInt(process.argv.includes('--wait') ? process.argv[process.argv.indexOf('--wait') + 1] : '15000');

const url = process.argv.includes('--url')
  ? process.argv[process.argv.indexOf('--url') + 1]
  : DEFAULT_URL;

console.log(`\n=== SF3D WebGPU Browser Smoke ===`);
console.log(`URL: ${url}\n`);

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: false, // need real GPU for WebGPU
  args: [
    '--enable-features=Vulkan,UseSkiaRenderer',
    '--enable-unsafe-webgpu',
    '--disable-dawn-features=disallow_unsafe_apis',
    '--no-first-run',
    '--no-default-browser-check',
  ],
});

const page = await browser.newPage();

// Collect all console messages
const consoleMessages = [];
page.on('console', (msg) => {
  const type = msg.type();
  const text = msg.text();
  consoleMessages.push({ type, text, ts: Date.now() });
  const prefix = type === 'error' ? 'ERR' : type === 'warning' ? 'WRN' : 'LOG';
  // Throttle weight loading spam
  if (text.startsWith('Loading weights...')) {
    const mb = parseInt(text.match(/(\d+) \//)?.[1] || '0');
    if (mb % 100 !== 0 && mb > 0) return;
  }
  console.log(`[${prefix}] ${text}`);
});

// Collect page errors (uncaught exceptions)
const pageErrors = [];
page.on('pageerror', (err) => {
  pageErrors.push({ message: err.message, stack: err.stack, ts: Date.now() });
  console.log(`[PAGEERR] ${err.message}`);
});

// Collect request failures
page.on('requestfailed', (req) => {
  console.log(`[REQFAIL] ${req.url()} — ${req.failure()?.errorText}`);
});

// Log 404s specifically
page.on('response', (res) => {
  if (res.status() >= 400) {
    console.log(`[HTTP ${res.status()}] ${res.url()}`);
  }
});

try {
  console.log('Navigating...');
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });

  // Wait for initial load and pipeline init
  console.log(`Waiting ${WAIT_MS / 1000}s for init + weight loading to begin...\n`);
  await new Promise(r => setTimeout(r, WAIT_MS));

  // Check status element
  const status = await page.$eval('#status', el => el.textContent).catch(() => '(not found)');
  console.log(`\n--- Status element: "${status}" ---`);

  // Summary
  const errors = consoleMessages.filter(m => m.type === 'error');
  const warnings = consoleMessages.filter(m => m.type === 'warning');

  console.log(`\n=== Summary ===`);
  console.log(`Total console messages: ${consoleMessages.length}`);
  console.log(`  Errors: ${errors.length}`);
  console.log(`  Warnings: ${warnings.length}`);
  console.log(`  Page errors: ${pageErrors.length}`);

  if (errors.length > 0) {
    console.log(`\n=== Errors ===`);
    for (const e of errors) {
      console.log(`  ${e.text}`);
    }
  }

  if (pageErrors.length > 0) {
    console.log(`\n=== Page Errors ===`);
    for (const e of pageErrors) {
      console.log(`  ${e.message}`);
      if (e.stack) console.log(`  ${e.stack.split('\n').slice(0, 3).join('\n  ')}`);
    }
  }

  // Take a screenshot
  const ssPath = '/tmp/sf3d-webgpu-smoke.png';
  await page.screenshot({ path: ssPath, fullPage: true });
  console.log(`\nScreenshot: ${ssPath}`);

} catch (err) {
  console.error('Smoke failed:', err.message);
} finally {
  await browser.close();
}

// Exit with error code if there were errors
const exitCode = pageErrors.length > 0 || consoleMessages.some(m => m.type === 'error') ? 1 : 0;
console.log(`\nExit: ${exitCode}`);
process.exit(exitCode);
