import { defineConfig, loadEnv } from 'vite';
import { access, cp, rm } from 'node:fs/promises';
import path from 'node:path';

const BUILD_LOCAL_MODEL_ASSETS = new Set(['weights.bin']);

function toBundlePath(filePath) {
  return filePath.split(path.sep).join('/');
}

function copyPublicAssetsWithoutLocalModels() {
  let resolvedConfig;

  return {
    name: 'sf3d-copy-public-assets-without-local-models',
    apply: 'build',
    configResolved(config) {
      resolvedConfig = config;
    },
    async writeBundle(_options, bundle) {
      if (!resolvedConfig.publicDir) {
        return;
      }

      const publicDir = path.resolve(resolvedConfig.publicDir);
      const outDir = path.resolve(resolvedConfig.root, resolvedConfig.build.outDir);
      const generatedOutputPaths = new Set(Object.keys(bundle));

      try {
        await access(publicDir);
      } catch (error) {
        if (error.code === 'ENOENT') {
          return;
        }
        throw error;
      }

      for (const asset of BUILD_LOCAL_MODEL_ASSETS) {
        await rm(path.join(outDir, asset), { force: true });
      }

      await cp(publicDir, outDir, {
        recursive: true,
        force: true,
        filter(source) {
          const relativePath = toBundlePath(path.relative(publicDir, source));
          return (
            !BUILD_LOCAL_MODEL_ASSETS.has(relativePath) &&
            !generatedOutputPaths.has(relativePath)
          );
        },
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  // Load .env[.mode] so config-time options can read VITE_* values the same way
  // the app does. (vite.config runs in Node and does not auto-load .env files.)
  const env = loadEnv(mode, process.cwd(), '');
  return {
    // Base path for hosted builds. GitHub Pages project sites serve from
    // /<repo>/, so the hosted build sets VITE_BASE=/sf3d-webgpu/. Dev and
    // root-domain deploys leave it at '/'.
    base: env.VITE_BASE || '/',
    plugins: [copyPublicAssetsWithoutLocalModels()],
    server: {
      port: 5176,
      open: true,
    },
    build: {
      copyPublicDir: false,
      target: 'esnext',
    },
  };
});
