import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

// One immutable in-memory snapshot supplies both the recorded digest and the
// later browser payload; the image path is never reread after this call.
export function readImageInput(filePath) {
  const resolvedPath = fs.realpathSync(filePath);
  const bytes = fs.readFileSync(resolvedPath);
  return {
    path: resolvedPath,
    bytes,
    sha256: crypto.createHash('sha256').update(bytes).digest('hex'),
  };
}

// Include relative paths and contents so same-version package edits change
// the installed-tree identity.
export async function sha256Tree(root) {
  const files = [];
  const visit = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(absolute);
      else if (entry.isFile()) files.push(absolute);
      else throw new Error(`unsupported installed-package entry: ${absolute}`);
    }
  };
  visit(root);
  files.sort();
  const hash = crypto.createHash('sha256');
  for (const file of files) {
    hash.update(path.relative(root, file));
    hash.update('\0');
    await new Promise((resolve, reject) => {
      const stream = fs.createReadStream(file);
      stream.on('data', chunk => hash.update(chunk));
      stream.on('error', reject);
      stream.on('end', resolve);
    });
    hash.update('\0');
  }
  return { sha256: hash.digest('hex'), files: files.map(file => path.relative(root, file)) };
}
