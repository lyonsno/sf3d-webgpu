import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';

// One actual installed cockpit dependency, identified by its committed lock.
// Serve canonical archive bytes, not an untracked installed-file assertion.
export async function prepareForegroundDependency({repoRoot,revision,outputDir}){
  const lockFile=path.join(repoRoot,'package-lock.json'),lockBytes=fs.readFileSync(lockFile);
  const committed=execFileSync('git',['show',revision+':package-lock.json'],{cwd:repoRoot,maxBuffer:lockBytes.length});
  if(!lockBytes.equals(committed))throw Error('foreground dependency lock differs from committed source');
  const lock=JSON.parse(committed),relative='node_modules/three-mesh-bvh/build/index.module.js';
  const entry=lock.packages?.['node_modules/three-mesh-bvh'];
  if(!entry?.version||!/^sha512-[A-Za-z0-9+/]+={0,2}$/.test(entry.integrity??'')||
    entry.resolved!=='https://registry.npmjs.org/three-mesh-bvh/-/three-mesh-bvh-'+entry.version+'.tgz')
    throw Error('canonical foreground dependency version/integrity/source required');
  const installed=fs.readFileSync(path.join(repoRoot,relative));
  const response=await fetch(entry.resolved,{redirect:'error',cache:'no-store'});
  if(!response.ok)throw Error('canonical foreground dependency unavailable: HTTP '+response.status);
  const archive=Buffer.from(await response.arrayBuffer());
  if('sha512-'+createHash('sha512').update(archive).digest('base64')!==entry.integrity)
    throw Error('foreground dependency canonical archive integrity mismatch');
  fs.mkdirSync(outputDir,{recursive:true});
  const archivePath=path.join(outputDir,'three-mesh-bvh-'+entry.version+'.tgz');
  fs.writeFileSync(archivePath,archive);
  const bytes=execFileSync('/usr/bin/tar',['-xOf',archivePath,'package/build/index.module.js'],{maxBuffer:Math.max(1,installed.length)});
  if(!bytes.equals(installed))throw Error('installed foreground dependency differs from canonical package bytes');
  const sha256=value=>createHash('sha256').update(value).digest('hex');
  return{relative,bytes,receipt:{package:'three-mesh-bvh',version:entry.version,resolved:entry.resolved,
    integrity:entry.integrity,archivePath,archiveSha256:sha256(archive),lockSha256:sha256(committed),
    sha256:sha256(bytes),byteLength:bytes.length,canonicalIntegrityVerified:true,installedBytesMatched:true}};
}
