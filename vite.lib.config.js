// Library build of the SF3D producer for hosts that mount it (the Kaminos
// live-flame composition): one ESM entry with the inference kit bundled,
// the five worker chunks emitted beside it under assets/, the CLIP shaders
// inlined, and the tet grid copied to tets/. Relative base so every URL
// resolves against the module's own location, wherever the host serves it.
// Model weights are NOT copied: the host serves weights.bin (2.13 GB) itself.
//   npx vite build -c vite.lib.config.js     → dist-lib/
import { defineConfig } from 'vite';
import { cp } from 'node:fs/promises';
import path from 'node:path';

function copyTetGrid() {
  let outDir;
  return {
    name: 'sf3d-lib-copy-tet-grid',
    apply: 'build',
    configResolved(config) { outDir = path.resolve(config.root, config.build.outDir); },
    async closeBundle() {
      await cp(path.resolve('public/tets'), path.join(outDir, 'tets'), { recursive: true, force: true });
    },
  };
}

export default defineConfig({
  base: './',
  publicDir: false,
  plugins: [copyTetGrid()],
  build: {
    outDir: 'dist-lib',
    emptyOutDir: true,
    target: 'esnext',
    minify: false,
    sourcemap: true,
    lib: {
      entry: 'src/lib/sf3d_producer.js',
      formats: ['es'],
      fileName: () => 'sf3d-producer.js',
    },
    rollupOptions: { external: [], output: { inlineDynamicImports: false } },
  },
  worker: { format: 'es' },
});
