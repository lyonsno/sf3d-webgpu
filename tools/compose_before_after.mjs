#!/usr/bin/env node
/**
 * Compose an "input photo -> generated 3D" side-by-side banner for the README.
 *
 * Renders both images into a labelled canvas on a transparent background and
 * writes a PNG. Uses Chrome via puppeteer for canvas compositing (no native
 * image toolchain required).
 *
 * Usage:
 *   node tools/compose_before_after.mjs \
 *     --input public/demo_chair.png \
 *     --render docs/assets/hero-chair.png \
 *     --out docs/assets/hero-before-after.png
 */

import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

const inputPath = arg('--input', 'public/demo_chair.png');
const renderPath = arg('--render', 'docs/assets/hero-chair.png');
const outPath = arg('--out', 'docs/assets/hero-before-after.png');

for (const [label, p] of [['input', inputPath], ['render', renderPath]]) {
  if (!fs.existsSync(p)) { console.error(`${label} not found: ${p}`); process.exit(1); }
}

const inputB64 = fs.readFileSync(inputPath).toString('base64');
const renderB64 = fs.readFileSync(renderPath).toString('base64');

// Layout: two equal cells with a centred arrow between them.
const CELL = 720;          // square cell edge
const GAP = 150;           // arrow gutter
const PAD = 40;            // outer padding
const LABEL_H = 56;
const W = PAD * 2 + CELL * 2 + GAP;
const H = PAD * 2 + LABEL_H + CELL;

const html = `<!doctype html>
<html><head><meta charset="utf-8"><style>
  html,body{margin:0;padding:0}
  canvas{display:block}
</style></head><body>
<canvas id="c" width="${W}" height="${H}"></canvas>
<script>
const c = document.getElementById('c');
const ctx = c.getContext('2d');
const PAD=${PAD}, CELL=${CELL}, GAP=${GAP}, LABEL_H=${LABEL_H};

function fitDraw(img, cx, cy, cw, ch) {
  const s = Math.min(cw / img.width, ch / img.height);
  const w = img.width * s, h = img.height * s;
  ctx.drawImage(img, cx + (cw - w) / 2, cy + (ch - h) / 2, w, h);
}

function label(text, cx, cw, y) {
  ctx.font = '600 30px -apple-system, Helvetica, Arial, sans-serif';
  ctx.fillStyle = '#6b7280';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, cx + cw / 2, y);
}

function loadImg(src){return new Promise((res,rej)=>{const i=new Image();i.onload=()=>res(i);i.onerror=rej;i.src=src;});}

(async () => {
  const input = await loadImg('data:image/png;base64,${inputB64}');
  const render = await loadImg('data:image/png;base64,${renderB64}');

  const cellY = PAD + LABEL_H;
  const leftX = PAD;
  const rightX = PAD + CELL + GAP;

  // Labels.
  label('Input photo', leftX, CELL, PAD + LABEL_H / 2);
  label('Generated 3D mesh', rightX, CELL, PAD + LABEL_H / 2);

  // Input: subtle rounded card so the photo edge reads cleanly.
  fitDraw(input, leftX, cellY, CELL, CELL);
  fitDraw(render, rightX, cellY, CELL, CELL);

  // Arrow in the gutter.
  const ay = cellY + CELL / 2;
  const ax0 = leftX + CELL + 34;
  const ax1 = rightX - 34;
  ctx.strokeStyle = '#9ca3af';
  ctx.fillStyle = '#9ca3af';
  ctx.lineWidth = 8;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(ax0, ay);
  ctx.lineTo(ax1 - 22, ay);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(ax1, ay);
  ctx.lineTo(ax1 - 26, ay - 18);
  ctx.lineTo(ax1 - 26, ay + 18);
  ctx.closePath();
  ctx.fill();

  window.__done = true;
})();
</script></body></html>`;

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: 'new',
  args: ['--no-first-run', '--no-default-browser-check', '--force-color-profile=srgb', '--hide-scrollbars'],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: W, height: H, deviceScaleFactor: 2 });
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message.slice(0, 300)}`));
  await page.setContent(html, { waitUntil: 'load', timeout: 30000 });
  await page.waitForFunction(() => window.__done === true, { timeout: 30000 });
  await new Promise((r) => setTimeout(r, 300));
  const el = await page.$('#c');
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  await el.screenshot({ path: outPath, omitBackground: true });
  console.log(`Wrote: ${outPath} (${(fs.statSync(outPath).size / 1024).toFixed(0)} KB)`);
} catch (err) {
  console.error(`Compose failed: ${err.message}`);
  process.exitCode = 1;
} finally {
  await browser.close();
}
