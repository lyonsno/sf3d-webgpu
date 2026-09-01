// Composite an alpha PNG onto a solid background color. Mechanical image op.
// Usage: node tools/composite_on_bg.mjs --in a.png --out b.png --bg "#1e1e1e"
import puppeteer from 'puppeteer-core';
import fs from 'fs';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
function arg(n, d) { const i = process.argv.indexOf(n); return i >= 0 ? process.argv[i + 1] : d; }
const inPath = arg('--in');
const outPath = arg('--out');
const bg = arg('--bg', '#1e1e1e');
const b64 = fs.readFileSync(inPath).toString('base64');

const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--force-color-profile=srgb', '--hide-scrollbars'] });
const page = await browser.newPage();
const dim = await (async () => {
  await page.setContent(`<img id="i" src="data:image/png;base64,${b64}">`);
  return page.$eval('#i', el => ({ w: el.naturalWidth, h: el.naturalHeight }));
})();
await page.setViewport({ width: dim.w, height: dim.h, deviceScaleFactor: 1 });
await page.setContent(`<!doctype html><html><head><style>html,body{margin:0}canvas{display:block}</style></head><body>
<canvas id="c" width="${dim.w}" height="${dim.h}"></canvas>
<script>
const ctx = document.getElementById('c').getContext('2d');
const img = new Image();
img.onload = () => { ctx.fillStyle='${bg}'; ctx.fillRect(0,0,${dim.w},${dim.h}); ctx.drawImage(img,0,0); window.__done=true; };
img.src='data:image/png;base64,${b64}';
</script></body></html>`, { waitUntil: 'load' });
await page.waitForFunction(() => window.__done === true, { timeout: 15000 });
const el = await page.$('#c');
await el.screenshot({ path: outPath });
console.log('wrote', outPath, dim);
await browser.close();
