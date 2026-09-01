// One-off: drive the BUILT/hosted demo against HF-hosted weights end to end.
import puppeteer from 'puppeteer-core';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const APP = 'http://127.0.0.1:4181/sf3d-webgpu/';

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-first-run', '--no-default-browser-check'],
});
const page = await browser.newPage();
let lastStatus = '';
const errors = [];
page.on('console', (m) => {
  const t = m.text();
  if (m.type() === 'error') { errors.push(t); console.log('[ERR]', t.slice(0, 200)); }
  else if (/weights|Ready|Adapter limits|maxBuffer|Done in|Loading weights/i.test(t)) console.log('[LOG]', t.slice(0, 200));
});
page.on('pageerror', (e) => { errors.push(e.message); console.log('[PAGEERR]', e.message.slice(0, 200)); });

async function status() { return page.$eval('#status', el => el.textContent).catch(() => '(none)'); }
async function waitFor(match, ms) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    const s = await status();
    if (s !== lastStatus) { lastStatus = s; console.log('[STATUS]', s); }
    if (s.includes(match)) return s;
    if (s.startsWith('Error:')) throw new Error('status error: ' + s);
    await new Promise(r => setTimeout(r, 500));
  }
  throw new Error(`timeout waiting for "${match}" (last: ${lastStatus})`);
}

try {
  console.log('navigating to hosted build:', APP);
  await page.goto(APP, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await waitFor('Ready', 600000); // weights stream from HF CDN
  console.log('\n>>> Reached Ready — weights loaded from HF CDN. Generating...');
  const fileInput = await page.$('#file-input');
  await fileInput.uploadFile('/private/tmp/sf3d-webgpu-readme-polish-0831/public/demo_chair.png');
  await new Promise(r => setTimeout(r, 1000));
  await page.click('#run-btn');
  const final = await waitFor('Done in', 300000);
  console.log('\n>>> GENERATION COMPLETE:', final);
  console.log('errors:', errors.length);
} catch (err) {
  console.error('\nHOSTED SMOKE FAILED:', err.message);
  process.exitCode = 1;
} finally {
  await browser.close();
}
