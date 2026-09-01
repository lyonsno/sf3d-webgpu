import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

import { verifyPackageArtifactIdentity } from '../src/lib/package_artifact_identity.js';

const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'sf3d-package-identity-'));
const producerRepo = path.join(temp, 'producer');
const packageSubdir = 'webgpu-inference-kit';
const sourcePackage = path.join(producerRepo, packageSubdir);
const stage = path.join(temp, 'stage');
const stagedPackage = path.join(stage, 'package');
const installedPackage = path.join(temp, 'installed');
const tarball = path.join(temp, 'package.tgz');
const producerRemote = 'https://example.invalid/lyonsno/kaminos.git';
const sha256File = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const git = args => execFileSync('git', args, { cwd: producerRepo, encoding: 'utf8' }).trim();

fs.mkdirSync(path.join(sourcePackage, 'src'), { recursive: true });
fs.writeFileSync(path.join(sourcePackage, 'package.json'), `${JSON.stringify({
  name: '@kaminos/webgpu-inference-kit',
  version: '9.8.7',
  files: ['src'],
}, null, 2)}\n`);
fs.writeFileSync(path.join(sourcePackage, 'src/index.js'), 'export const identity = "exact";\n');
fs.writeFileSync(path.join(sourcePackage, 'src/weight.js'), 'export const packed = true;\n');
execFileSync('git', ['init', '--quiet'], { cwd: producerRepo });
execFileSync('git', ['remote', 'add', 'origin', producerRemote], { cwd: producerRepo });
execFileSync('git', ['add', '.'], { cwd: producerRepo });
execFileSync('git', [
  '-c', 'user.name=Fixture',
  '-c', 'user.email=fixture@example.invalid',
  'commit', '--quiet', '-m', 'fixture package',
], { cwd: producerRepo });
const producerRevision = git(['rev-parse', 'HEAD']);

fs.mkdirSync(stage, { recursive: true });
fs.cpSync(sourcePackage, stagedPackage, { recursive: true });
fs.cpSync(sourcePackage, installedPackage, { recursive: true });
execFileSync('tar', ['-czf', tarball, '-C', stage, 'package']);

const request = {
  producerRepo,
  producerRevision,
  producerRemote,
  packageSubdir,
  packageVersion: '9.8.7',
  tarballPath: tarball,
  tarballSha256: sha256File(tarball),
  installedPackagePath: installedPackage,
};
const identity = verifyPackageArtifactIdentity(request);
assert.equal(identity.producerRevision, producerRevision);
assert.equal(identity.producerRemote, producerRemote);
assert.match(identity.producerTree, /^[0-9a-f]{40}$/);
assert.equal(identity.tarballManifestSha256, identity.installedManifestSha256);
assert.equal(identity.tarballManifestSha256, identity.sourceManifestSha256);
assert.equal(identity.packageVersion, '9.8.7');

fs.writeFileSync(path.join(installedPackage, 'src/index.js'), 'export const identity = "wrong";\n');
assert.throws(() => verifyPackageArtifactIdentity(request), /installed package.*tarball|manifest/i);
fs.cpSync(path.join(sourcePackage, 'src/index.js'), path.join(installedPackage, 'src/index.js'));

const unrelatedStage = path.join(temp, 'unrelated-stage');
const unrelatedPackage = path.join(unrelatedStage, 'package');
const unrelatedTarball = path.join(temp, 'unrelated.tgz');
fs.mkdirSync(unrelatedStage, { recursive: true });
fs.cpSync(sourcePackage, unrelatedPackage, { recursive: true });
fs.writeFileSync(path.join(unrelatedPackage, 'src/weight.js'), 'export const packed = false;\n');
execFileSync('tar', ['-czf', unrelatedTarball, '-C', unrelatedStage, 'package']);
assert.throws(() => verifyPackageArtifactIdentity({
  ...request,
  tarballPath: unrelatedTarball,
  tarballSha256: sha256File(unrelatedTarball),
}), /installed package.*tarball|producer source.*tarball|manifest/i);

assert.throws(() => verifyPackageArtifactIdentity({
  ...request,
  producerRevision: 'f'.repeat(40),
}), /producer revision/i);
assert.throws(() => verifyPackageArtifactIdentity({
  ...request,
  producerRemote: 'https://example.invalid/wrong.git',
}), /producer remote/i);

console.log('packed fp16 package identity contracts passed');
