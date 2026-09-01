import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vite';

function filesystemPath(value) {
  return value instanceof URL ? fileURLToPath(value) : value;
}

export function createAssayWeightMiddleware(weightsPath) {
  const file = filesystemPath(weightsPath);
  if (!file || !fs.statSync(file).isFile()) {
    throw new Error(`SF3D assay weight artifact is not a file: ${file}`);
  }
  return (request, response, next) => {
    const pathname = new URL(request.url || '/', 'http://sf3d-assay.local').pathname;
    if (pathname !== '/weights.bin') {
      next();
      return;
    }
    const { size } = fs.statSync(file);
    response.statusCode = 200;
    response.setHeader('content-type', 'application/octet-stream');
    response.setHeader('content-length', String(size));
    response.setHeader('cache-control', 'no-store');
    const stream = fs.createReadStream(file);
    stream.on('error', error => response.destroy(error));
    stream.pipe(response);
  };
}

export function createDinoAssayViteConfig(weightsPath) {
  return defineConfig({
    plugins: [{
      name: 'sf3d-authenticated-assay-weights',
      configureServer(server) {
        server.middlewares.use(createAssayWeightMiddleware(weightsPath));
      },
    }],
  });
}

export default defineConfig(() => createDinoAssayViteConfig(
  process.env.SF3D_ASSAY_WEIGHTS_PATH,
));
