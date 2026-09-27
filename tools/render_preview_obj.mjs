#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const objPath = path.resolve(process.argv[2] || '');
const pngPath = path.resolve(process.argv[3] || '');
if (!fs.existsSync(objPath) || !process.argv[3]) {
  throw new Error('usage: node tools/render_preview_obj.mjs INPUT.obj OUTPUT.png');
}
const vertices = [];
const faces = [];
for (const line of fs.readFileSync(objPath, 'utf8').split('\n')) {
  const parts = line.split(' ');
  if (parts[0] === 'v') vertices.push(parts.slice(1).map(Number));
  if (parts[0] === 'f') faces.push(parts.slice(1).map((value) => Number(value) - 1));
}
if (!vertices.length || !faces.length || vertices.some(v => v.length !== 3 || v.some(x => !Number.isFinite(x)))
    || faces.some(f => f.length !== 3 || f.some(i => !Number.isSafeInteger(i) || i < 0 || i >= vertices.length))) {
  throw new Error('OBJ is empty or malformed');
}
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  args: ['--no-first-run', '--no-default-browser-check'],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 440, deviceScaleFactor: 1 });
  await page.setContent('<body style="margin:0;background:#f1f4f3"><canvas width="1200" height="440"></canvas></body>');
  await page.evaluate(({ vertices, faces }) => {
    const ctx = document.querySelector('canvas').getContext('2d');
    const views = [0.15, 1.25, 2.5];
    ctx.fillStyle = '#f1f4f3';
    ctx.fillRect(0, 0, 1200, 440);
    for (let view = 0; view < views.length; view += 1) {
      const yaw = views[view];
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const pitch = 0.17, cp = Math.cos(pitch), sp = Math.sin(pitch);
      const rotated = vertices.map(([x, y, z]) => {
        const rx = cy * x + sy * y;
        const depth = -sy * x + cy * y;
        return [rx, cp * z - sp * depth, sp * z + cp * depth];
      });
      const projected = rotated.map(([x, y]) => [view * 400 + 200 + x * 195, 220 - y * 195]);
      const triangles = faces.map(([a, b, c]) => {
        const p = rotated[a], q = rotated[b], r = rotated[c];
        const ux = q[0] - p[0], uy = q[1] - p[1], uz = q[2] - p[2];
        const vx = r[0] - p[0], vy = r[1] - p[1], vz = r[2] - p[2];
        const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const len = Math.hypot(nx, ny, nz) || 1;
        const shade = Math.max(0, Math.min(1, 0.38 + 0.62 * (nx * -0.35 + ny * 0.75 + nz * 0.55) / len));
        return { a, b, c, depth: (p[2] + q[2] + r[2]) / 3, shade };
      }).sort((a, b) => a.depth - b.depth);
      for (const triangle of triangles) {
        const [a, b, c] = [triangle.a, triangle.b, triangle.c].map(i => projected[i]);
        const v = Math.round(45 + triangle.shade * 175);
        ctx.fillStyle = `rgb(${v - 10},${v},${v - 5})`;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(c[0], c[1]); ctx.closePath(); ctx.fill();
      }
      ctx.fillStyle = '#24322d';
      ctx.font = '14px sans-serif';
      ctx.fillText(['Front', 'Side', 'Reverse'][view], view * 400 + 20, 30);
    }
  }, { vertices, faces });
  fs.mkdirSync(path.dirname(pngPath), { recursive: true });
  await page.screenshot({ path: pngPath });
  console.log(`${vertices.length} vertices, ${faces.length} faces: ${pngPath}`);
} finally {
  await browser.close();
}
