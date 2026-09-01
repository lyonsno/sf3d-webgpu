#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { readSf3dWeightInventory } from './lib/sf3d_weight_inventory.mjs';

const value = flag => {
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] : undefined;
};

const weightsPath = value('--weights');
const outputPath = value('--output');
const repoId = value('--source-repo');
const revision = value('--source-revision');
const sourceFile = value('--source-file');
const expectedSha256 = value('--expected-sha256');
if (!weightsPath || !outputPath || !repoId || !revision || !sourceFile || !expectedSha256) {
  throw new Error(
    '--weights, --output, --source-repo, --source-revision, --source-file, and --expected-sha256 are required',
  );
}
if (!/^[0-9a-f]{64}$/.test(expectedSha256)) {
  throw new Error('--expected-sha256 must be an exact lowercase SHA-256 digest');
}

const inventory = await readSf3dWeightInventory(path.resolve(weightsPath), {
  source: { kind: 'hugging-face', repoId, revision, file: sourceFile },
});
if (inventory.artifact.sha256 !== expectedSha256) {
  throw new Error(
    `weight artifact SHA-256 ${inventory.artifact.sha256} does not match expected ${expectedSha256}`,
  );
}

const absoluteOutput = path.resolve(outputPath);
fs.mkdirSync(path.dirname(absoluteOutput), { recursive: true });
const temporaryPath = `${absoluteOutput}.${process.pid}.${crypto.randomUUID()}.tmp`;
const fd = fs.openSync(temporaryPath, 'wx');
try {
  fs.writeFileSync(fd, `${JSON.stringify(inventory, null, 2)}\n`);
  fs.fsyncSync(fd);
} finally {
  fs.closeSync(fd);
}
try {
  fs.renameSync(temporaryPath, absoluteOutput);
} catch (error) {
  fs.rmSync(temporaryPath, { force: true });
  throw error;
}

console.log(JSON.stringify({
  output: absoluteOutput,
  source: inventory.source,
  artifact: inventory.artifact,
  format: inventory.format,
  summary: inventory.summary,
}, null, 2));
