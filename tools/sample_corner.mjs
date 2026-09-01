import puppeteer from 'puppeteer-core';
import fs from 'fs';
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const b64 = fs.readFileSync(process.argv[2]).toString('base64');
const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--force-color-profile=srgb'] });
const page = await browser.newPage();
await page.setContent(`<canvas id="c"></canvas><script>
const img=new Image();img.onload=()=>{const c=document.getElementById('c');c.width=img.width;c.height=img.height;const x=c.getContext('2d');x.drawImage(img,0,0);
const pts={topLeft:[10,10],topRight:[img.width-10,10],midLeft:[10,Math.floor(img.height/2)]};
const out={};for(const k in pts){const d=x.getImageData(pts[k][0],pts[k][1],1,1).data;out[k]='#'+[d[0],d[1],d[2]].map(v=>v.toString(16).padStart(2,'0')).join('');}
window.__c=out;};img.src='data:image/png;base64,${b64}';
</script>`, { waitUntil: 'load' });
await page.waitForFunction(() => window.__c, { timeout: 15000 });
console.log(JSON.stringify(await page.evaluate(() => window.__c)));
await browser.close();
