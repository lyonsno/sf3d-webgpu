import { execFileSync } from 'node:child_process';
import path from 'node:path';

export function resolveViteCommit(projectRoot) {
  const commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: path.resolve(projectRoot) })
    .toString('utf8').trim();
  if (!/^[0-9a-f]{40}$/i.test(commit)) throw new Error(`Vite source commit is not a full Git object id: ${commit}`);
  return commit;
}
