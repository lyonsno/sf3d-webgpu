#!/usr/bin/env node
/**
 * Compose an "input photo -> generated 3D" side-by-side banner for the README.
 *
 * Both sides are drawn into equal rounded cards on a neutral page background so
 * a studio-rendered mesh (with its own backdrop) and a cut-out product photo
 * read as one consistent figure. Each image can be independently scaled so the
 * subject sizes match visually.
 *
 * Usage:
 *   node tools/compose_before_after.mjs \
 *     --input public/demo_chair.png \
 *     --render docs/assets/hero-chair.png \
 *     --out docs/assets/hero-before-after.png \
 *     [--input-scale 0.9] [--render-scale 1.0]
 *
 * Uses Chrome via puppeteer for canvas compositing (no native toolchain).
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
// Per-image subject-scale (fraction of the card the image fills). Lets a
// cut-out photo and a full studio render show subjects at matching size.
const inputScale = parseFloat(arg('--input-scale', '0.82'));
const renderScale = parseFloat(arg('--render-scale', '1.0'));
const cardBg = arg('--card-bg', '#eef0f2'); // neutral card fill behind cut-outs

for (const [label, p] of [['input', inputPath], ['render', renderPath]]) {
  if (!fs.existsSync(p)) { console.error(`${label} not found: ${p}`); process.exit(1); }
}

const inputB64 = fs.readFileSync(inputPath).toString('base64');
const renderB64 = fs.readFileSync(renderPath).toString('base64');

const CELL = 760;          // square card edge
const GAP = 150;           // arrow gutter
const PAD = 48;            // outer padding
const LABEL_H = 60;
const RADIUS = 24;
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
const PAD=${PAD}, CELL=${CELL}, GAP=${GAP}, LABEL_H=${LABEL_H}, RADIUS=${RADIUS};
const CARD_BG='${cardBg}';

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// Draw an image contained within a card, scaled by \`scale\` about the card
// centre, clipped to the card's rounded rect.
function drawInCard(img, x, y, size, scale) {
  ctx.save();
  roundRect(x, y, size, size, RADIUS);
  ctx.clip();
  // Card fill (shows through transparent cut-outs; hidden by opaque renders).
  ctx.fillStyle = CARD_BG;
  ctx.fillRect(x, y, size, size);
  const base = Math.min(size / img.width, size / img.height);
  const s = base * scale;
  const w = img.width * s, h = img.height * s;
  ctx.drawImage(img, x + (size - w) / 2, y + (size - h) / 2, w, h);
  ctx.restore();
}

function label(text, cx, cw, y) {
  ctx.font = '600 32px -apple-system, Helvetica, Arial, sans-serif';
  ctx.fillStyle = '#4b5563';
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

  label('Input photo', leftX, CELL, PAD + LABEL_H / 2);
  label('Generated 3D mesh', rightX, CELL, PAD + LABEL_H / 2);

  // Soft drop shadow under each card.
  ctx.save();
  ctx.shadowColor = 'rgba(15,23,42,0.18)';
  ctx.shadowBlur = 28;
  ctx.shadowOffsetY = 10;
  ctx.fillStyle = '#ffffff';
  roundRect(leftX, cellY, CELL, CELL, RADIUS); ctx.fill();
  roundRect(rightX, cellY, CELL, CELL, RADIUS); ctx.fill();
  ctx.restore();

  drawInCard(input, leftX, cellY, CELL, ${inputScale});
  drawInCard(render, rightX, cellY, CELL, ${renderScale});

  // Hairline border on each card.
  ctx.strokeStyle = 'rgba(15,23,42,0.10)';
  ctx.lineWidth = 2;
  roundRect(leftX, cellY, CELL, CELL, RADIUS); ctx.stroke();
  roundRect(rightX, cellY, CELL, CELL, RADIUS); ctx.stroke();

  // Arrow in the gutter.
  const ay = cellY + CELL / 2;
  const ax1 = rightX - 30;
  ctx.strokeStyle = '#9ca3af';
  ctx.fillStyle = '#9ca3af';
  ctx.lineWidth = 9;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(leftX + CELL + 34, ay);
  ctx.lineTo(ax1 - 24, ay);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(ax1, ay);
  ctx.lineTo(ax1 - 28, ay - 20);
  ctx.lineTo(ax1 - 28, ay + 20);
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
