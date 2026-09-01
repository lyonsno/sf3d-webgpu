import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const SHA256_HEX = /^[0-9a-f]{64}$/;
const GIT_SHA_HEX = /^[0-9a-f]{40}$/;
const sha256 = data => crypto.createHash('sha256').update(data).digest('hex');
const command = (file, args, options = {}) => execFileSync(file, args, {
  encoding: options.encoding ?? null,
  maxBuffer: options.maxBuffer ?? 128 * 1024 * 1024,
  stdio: ['ignore', 'pipe', 'pipe'],
  ...options,
});

function listFiles(root, prefix = '') {
  const files = [];
  for (const entry of fs.readdirSync(path.join(root, prefix), { withFileTypes: true })) {
    const relative = path.posix.join(prefix.split(path.sep).join('/'), entry.name);
    const absolute = path.join(root, relative);
    if (entry.isSymbolicLink()) throw new Error(`installed package symlink is not admitted: ${relative}`);
    if (entry.isDirectory()) files.push(...listFiles(root, relative));
    else if (entry.isFile()) files.push(relative);
    else throw new Error(`unsupported installed package entry: ${relative}`);
  }
  return files.sort();
}

function canonicalManifest(entries) {
  const lines = entries.map(entry => `${entry.path}\0${entry.size}\0${entry.sha256}\n`).join('');
  return sha256(Buffer.from(lines));
}

function gitOutput(repo, args, encoding = 'utf8') {
  try {
    return command('git', args, { cwd: repo, encoding }).toString().trim();
  } catch (error) {
    throw new Error(`producer revision or repository lookup failed: ${error.message}`);
  }
}

export function verifyPackageArtifactIdentity({
  producerRepo,
  producerRevision,
  producerRemote,
  packageSubdir,
  packageVersion,
  tarballPath,
  tarballSha256,
  installedPackagePath,
}) {
  if (!GIT_SHA_HEX.test(producerRevision || '')) {
    throw new Error('producer revision must be an exact Git SHA');
  }
  if (!SHA256_HEX.test(tarballSha256 || '')) {
    throw new Error('tarball SHA-256 must be an exact digest');
  }
  for (const [name, value] of Object.entries({
    producerRepo,
    producerRemote,
    packageSubdir,
    packageVersion,
    tarballPath,
    installedPackagePath,
  })) {
    if (typeof value !== 'string' || value.trim().length === 0) {
      throw new Error(`${name} is required`);
    }
  }
  if (!fs.statSync(tarballPath, { throwIfNoEntry: false })?.isFile()) {
    throw new Error('tarball path must name a file');
  }
  if (!fs.statSync(installedPackagePath, { throwIfNoEntry: false })?.isDirectory()) {
    throw new Error('installed package path must name a directory');
  }

  const effectiveRevision = gitOutput(producerRepo, ['rev-parse', '--verify', `${producerRevision}^{commit}`]);
  if (effectiveRevision !== producerRevision) throw new Error('producer revision did not resolve exactly');
  const effectiveRemote = gitOutput(producerRepo, ['remote', 'get-url', 'origin']);
  if (effectiveRemote !== producerRemote) throw new Error('producer remote does not match expected identity');
  const producerTree = gitOutput(producerRepo, ['rev-parse', `${producerRevision}:${packageSubdir}`]);
  if (!GIT_SHA_HEX.test(producerTree)) throw new Error('producer package tree did not resolve');

  const effectiveTarballSha256 = sha256(fs.readFileSync(tarballPath));
  if (effectiveTarballSha256 !== tarballSha256) throw new Error('tarball SHA-256 does not match expected identity');
  const tarListing = command('tar', ['-tzf', tarballPath], { encoding: 'utf8' })
    .split('\n')
    .filter(Boolean);
  const tarFiles = tarListing.filter(entry => !entry.endsWith('/')).map(entry => {
    if (!entry.startsWith('package/')) throw new Error(`tarball entry is outside package/: ${entry}`);
    const relative = entry.slice('package/'.length);
    if (!relative || relative.startsWith('/') || relative.split('/').includes('..')) {
      throw new Error(`unsafe tarball entry: ${entry}`);
    }
    return { entry, relative };
  }).sort((left, right) => (left.relative < right.relative ? -1 : left.relative > right.relative ? 1 : 0));
  if (tarFiles.length === 0) throw new Error('tarball contains no package files');

  const installedFiles = listFiles(installedPackagePath);
  const expectedFiles = tarFiles.map(file => file.relative);
  if (JSON.stringify(installedFiles) !== JSON.stringify(expectedFiles)) {
    throw new Error('installed package file set does not match tarball');
  }

  const tarballEntries = [];
  const installedEntries = [];
  const sourceEntries = [];
  for (const { entry, relative } of tarFiles) {
    const tarBytes = command('tar', ['-xOzf', tarballPath, entry]);
    const installedBytes = fs.readFileSync(path.join(installedPackagePath, relative));
    let sourceBytes;
    try {
      sourceBytes = command('git', ['show', `${producerRevision}:${packageSubdir}/${relative}`], {
        cwd: producerRepo,
      });
    } catch (error) {
      throw new Error(`producer source is missing tarball file ${relative}: ${error.message}`);
    }
    const tarEntry = { path: relative, size: tarBytes.byteLength, sha256: sha256(tarBytes) };
    const installedEntry = {
      path: relative,
      size: installedBytes.byteLength,
      sha256: sha256(installedBytes),
    };
    const sourceEntry = { path: relative, size: sourceBytes.byteLength, sha256: sha256(sourceBytes) };
    tarballEntries.push(tarEntry);
    installedEntries.push(installedEntry);
    sourceEntries.push(sourceEntry);
    if (tarEntry.size !== installedEntry.size || tarEntry.sha256 !== installedEntry.sha256) {
      throw new Error(`installed package does not match tarball at ${relative}`);
    }
    if (tarEntry.size !== sourceEntry.size || tarEntry.sha256 !== sourceEntry.sha256) {
      throw new Error(`producer source does not match tarball at ${relative}`);
    }
  }

  const packageJson = JSON.parse(command('tar', [
    '-xOzf', tarballPath, 'package/package.json',
  ], { encoding: 'utf8' }));
  if (packageJson.version !== packageVersion) {
    throw new Error('tarball package version does not match expected identity');
  }
  const tarballManifestSha256 = canonicalManifest(tarballEntries);
  const installedManifestSha256 = canonicalManifest(installedEntries);
  const sourceManifestSha256 = canonicalManifest(sourceEntries);
  if (tarballManifestSha256 !== installedManifestSha256) {
    throw new Error('installed package manifest does not match tarball manifest');
  }
  if (tarballManifestSha256 !== sourceManifestSha256) {
    throw new Error('producer source manifest does not match tarball manifest');
  }

  return Object.freeze({
    packageVersion,
    producerRevision,
    producerRemote,
    producerTree,
    tarballSha256: effectiveTarballSha256,
    tarballManifestSha256,
    installedManifestSha256,
    sourceManifestSha256,
    fileCount: tarFiles.length,
  });
}
