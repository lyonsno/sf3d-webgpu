import assert from 'node:assert/strict';
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const fixtureRoot = await mkdtemp(path.join(repoRoot, '.sf3d-build-no-public-'));
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
  await writeFile(
    path.join(fixtureRoot, 'index.html'),
    '<!doctype html><script type="module" src="/src.js"></script>',
  );
  await writeFile(path.join(fixtureRoot, 'src.js'), 'document.body.dataset.built = "true";');
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
    `production build without public directory failed\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`,
  );
  assert.equal(await exists(path.join(outDir, 'index.html')), true, 'application build is missing');
  assert.equal(await exists(path.join(outDir, 'weights.bin')), false, 'build emitted model weights');

  console.log('SF3D no-public-directory build contract passed');
} finally {
  await rm(fixtureRoot, { recursive: true, force: true });
}
