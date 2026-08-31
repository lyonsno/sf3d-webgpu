#!/usr/bin/env node
/**
 * Render a GLB to a hero still (and optionally a turntable) for README imagery.
 *
 * Loads the GLB into <model-viewer> with neutral studio lighting on a
 * transparent background, then captures PNG frames from chosen camera orbits.
 *
 * Usage:
 *   node tools/render_glb_hero.mjs --glb /tmp/sf3d-inference-smoke.glb --out docs/assets/hero.png
 *   node tools/render_glb_hero.mjs --glb model.glb --out out.png --orbit "30deg 75deg 105%"
 *   node tools/render_glb_hero.mjs --glb model.glb --contact-out docs/assets/wireframe.png --orbit "-35deg 80deg 110%"
 *
 * Requires Google Chrome (headless=new) and network access for the
 * model-viewer module (pinned CDN version).
 */

import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const MODEL_VIEWER_URL = 'https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js';

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

const glbPath = arg('--glb', '/tmp/sf3d-inference-smoke.glb');
const outPath = arg('--out', '/tmp/sf3d-hero.png');
const orbit = arg('--orbit', '25deg 78deg 105%');
const width = parseInt(arg('--width', '1280'), 10);
const height = parseInt(arg('--height', '960'), 10);
const bg = arg('--bg', 'transparent'); // 'transparent' or a CSS color

if (!fs.existsSync(glbPath)) {
  console.error(`GLB not found: ${glbPath}`);
  process.exit(1);
}

const glbBytes = fs.readFileSync(glbPath);
const glbBase64 = glbBytes.toString('base64');
console.log(`GLB: ${glbPath} (${(glbBytes.length / 1024).toFixed(0)} KB)`);
console.log(`Orbit: ${orbit}  Size: ${width}x${height}  BG: ${bg}`);

const html = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; width: ${width}px; height: ${height}px; }
  body { background: ${bg === 'transparent' ? 'rgba(0,0,0,0)' : bg}; }
  model-viewer {
    width: ${width}px; height: ${height}px;
    --poster-color: transparent;
    background-color: ${bg === 'transparent' ? 'rgba(0,0,0,0)' : bg};
  }
</style>
<script type="module" src="${MODEL_VIEWER_URL}"></script>
</head>
<body>
  <model-viewer
    id="mv"
    src="data:model/gltf-binary;base64,${glbBase64}"
    camera-orbit="${orbit}"
    exposure="1.0"
    shadow-intensity="1"
    shadow-softness="0.75"
    environment-image="neutral"
    interaction-prompt="none"
    disable-zoom
    disable-pan>
  </model-viewer>
</body>
</html>`;

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: 'new',
  args: [
    '--enable-features=Vulkan,UseSkiaRenderer',
    '--enable-unsafe-webgpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--force-color-profile=srgb',
    '--hide-scrollbars',
  ],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: 2 });
  page.on('console', (m) => console.log(`[page:${m.type()}] ${m.text().slice(0, 200)}`));
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message.slice(0, 300)}`));

  await page.setContent(html, { waitUntil: 'networkidle0', timeout: 45000 });

  // Wait for model-viewer to load and render the model.
  await page.waitForFunction(() => {
    const mv = document.querySelector('#mv');
    return mv && mv.loaded === true && mv.modelIsVisible === true;
  }, { timeout: 45000 }).catch(() => console.log('[warn] model load flag not observed; capturing anyway'));

  // Small settle for shadow/env baking.
  await new Promise((r) => setTimeout(r, 1200));

  const el = await page.$('#mv');
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  await el.screenshot({ path: outPath, omitBackground: bg === 'transparent' });
  console.log(`Wrote: ${outPath} (${(fs.statSync(outPath).size / 1024).toFixed(0)} KB)`);
} catch (err) {
  console.error(`Render failed: ${err.message}`);
  process.exitCode = 1;
} finally {
  await browser.close();
}
