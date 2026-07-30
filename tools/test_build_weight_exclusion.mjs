import assert from 'node:assert/strict';
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const fixtureRoot = await mkdtemp(path.join(repoRoot, '.sf3d-build-weight-exclusion-'));
const publicRoot = path.join(fixtureRoot, 'public');
const outDir = path.join(fixtureRoot, 'dist');

async function exists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

try {
  await mkdir(path.join(publicRoot, 'tets'), { recursive: true });
  await mkdir(outDir, { recursive: true });
  await writeFile(
    path.join(fixtureRoot, 'index.html'),
    '<!doctype html><script type="module" src="/src.js"></script>',
  );
  await writeFile(path.join(fixtureRoot, 'src.js'), 'document.body.dataset.built = "true";');
  await writeFile(path.join(outDir, 'weights.bin'), 'stale-model-payload-must-not-survive');
  await writeFile(path.join(publicRoot, 'weights.bin'), 'local-model-payload-must-not-be-built');
  await writeFile(path.join(publicRoot, 'weights.json'), '{"fixture":true}\n');
  await writeFile(path.join(publicRoot, 'demo_chair.png'), 'fixture-image');
  await writeFile(path.join(publicRoot, 'tets', 'indices.bin'), 'fixture-tets');
  await writeFile(path.join(publicRoot, 'index.html'), 'public-index-must-not-win');
  await writeFile(
    path.join(fixtureRoot, 'vite.config.js'),
    await readFile(path.join(repoRoot, 'vite.config.js')),
  );

  const result = spawnSync(
    process.execPath,
    [
      path.join(repoRoot, 'node_modules', 'vite', 'bin', 'vite.js'),
      'build',
      fixtureRoot,
      '--config',
      path.join(fixtureRoot, 'vite.config.js'),
      '--outDir',
      outDir,
      '--logLevel',
      'error',
    ],
    {
      cwd: repoRoot,
      encoding: 'utf8',
    },
  );

  assert.equal(
    result.status,
    0,
    `fixture production build failed\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`,
  );
  assert.equal(
    await exists(path.join(outDir, 'weights.bin')),
    false,
    'ordinary production build copied public/weights.bin into dist',
  );
  assert.equal(
    await readFile(path.join(outDir, 'weights.json'), 'utf8'),
    '{"fixture":true}\n',
    'weight manifest was not preserved',
  );
  assert.equal(
    await readFile(path.join(outDir, 'demo_chair.png'), 'utf8'),
    'fixture-image',
    'demo image was not preserved',
  );
  assert.equal(
    await readFile(path.join(outDir, 'tets', 'indices.bin'), 'utf8'),
    'fixture-tets',
    'tetrahedral lookup asset was not preserved',
  );
  assert.equal(await exists(path.join(outDir, 'index.html')), true, 'application build is missing');
  assert.notEqual(
    await readFile(path.join(outDir, 'index.html'), 'utf8'),
    'public-index-must-not-win',
    'public asset overwrote generated application output',
  );

  console.log('SF3D build weight exclusion contract passed');
} finally {
  await rm(fixtureRoot, { recursive: true, force: true });
}
