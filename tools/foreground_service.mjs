import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {createHash} from 'node:crypto';
import {spawn,execFileSync} from 'node:child_process';
import {stopOwnedBrowser} from './owned_browser_stop.mjs';

const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const adapter=new URL('./kaminos_foreground_server.py',import.meta.url);
export function validateForegroundService(config,{repoRoot,revision,outputDir}){
  if(config?.schema!=='kaminos.runtime-config.v0'||config.source?.repoRoot!==repoRoot||config.source.commit!==revision||
    config.sharedBasinStore!==null||config.volumeSettingsStore!==path.join(outputDir,'settings')||
    config.volumeBasinSessionStore!==path.join(outputDir,'sessions')||config.volumeCockpitLayoutStore!==path.join(outputDir,'layouts'))
    throw Error('actual foreground server source/store identity mismatch');
}

export async function startForegroundService({repoRoot,revision,outputDir,onSpawn}){
  if(!path.isAbsolute(repoRoot)||!path.isAbsolute(outputDir)||!/^[a-f0-9]{40}$/.test(revision))throw Error('explicit actual server root/revision/output required');
  fs.mkdirSync(outputDir,{recursive:true});
  const object=revision+':serve.py',size=Number(execFileSync('git',['cat-file','-s',object],{cwd:repoRoot,encoding:'utf8'}));
  const bytes=fs.readFileSync(path.join(repoRoot,'serve.py'));
  if(!Number.isSafeInteger(size)||size<1||hash(bytes)!==hash(execFileSync('git',['show',object],{cwd:repoRoot,maxBuffer:size})))
    throw Error('actual server entrypoint differs from requested source');
  const names=['HOME','TMPDIR','PATH','LANG','LC_ALL','LC_CTYPE','__CF_USER_TEXT_ENCODING'];
  const env=Object.fromEntries(names.filter(name=>process.env[name]!=null).map(name=>[name,process.env[name]]));
  Object.assign(env,{KAMINOS_ASSETS_DIR:path.join(outputDir,'assets'),KAMINOS_SCENES_DIR:path.join(outputDir,'scenes'),
    KAMINOS_SCENE_LIBRARY_GLOBS:path.join(outputDir,'scenes','*')});
  const child=spawn('/usr/bin/python3',['-B','-u',path.resolve(adapter.pathname),repoRoot,outputDir],{cwd:repoRoot,env,stdio:['ignore','pipe','pipe']});
  onSpawn?.(child);
  const logPath=path.join(outputDir,'server.log'),network=[];
  child.stderr.on('data',chunk=>fs.appendFileSync(logPath,chunk));
  child.stdout.on('data',chunk=>fs.appendFileSync(logPath,chunk));
  try{
    const announced=await new Promise((resolve,reject)=>{
      let pending='';
      const cleanup=()=>{child.stdout.off('data',data);child.off('exit',exit);child.off('error',error);};
      const data=chunk=>{pending+=chunk;const newline=pending.indexOf('\n');if(newline<0)return;
        try{const value=JSON.parse(pending.slice(0,newline));cleanup();resolve(value);}catch(e){cleanup();reject(e);}};
      const exit=(code,signal)=>{cleanup();reject(Error('actual foreground server exited before listening: '+code+'/'+signal));};
      const error=e=>{cleanup();reject(e);};
      child.stdout.on('data',data);child.once('exit',exit);child.once('error',error);
    });
    if(announced.pid!==child.pid||!/^http:\/\/127\.0\.0\.1:\d+$/.test(announced.origin))throw Error('owned loopback actual-server announcement required');
    const response=await fetch(announced.origin+'/api/runtime-config');
    const raw=Buffer.from(await response.arrayBuffer());fs.writeFileSync(path.join(outputDir,'runtime-config.json'),raw);
    if(!response.ok)throw Error('actual foreground runtime config HTTP '+response.status);
    const config=JSON.parse(raw);validateForegroundService(config,{repoRoot,revision,outputDir});
    const receipt={route:'owned-actual-kaminos-handler.v0',ownedPid:child.pid,origin:announced.origin,loopback:true,
      serveSha256:hash(bytes),adapterSha256:hash(fs.readFileSync(adapter)),effective:config,logPath,network,
      childEnvironment:{policy:'positive-allowlist-with-caller-owned-stores',names:Object.keys(env),valuesRecorded:false}};
    const service={};
    const proxy=(req,res,{requestedUrl=req.url}={})=>{
      const record={url:req.url,requestedUrl,method:req.method,route:'actual-kaminos-handler',at:new Date().toISOString()};network.push(record);
      const upstream=http.request(new URL(req.url,announced.origin),{method:req.method},reply=>{
        record.status=reply.statusCode;record.contentType=reply.headers['content-type']??null;
        const digest=createHash('sha256');
        const rawPath=path.join(outputDir,'response-'+network.length+'.raw');record.rawPath=rawPath;
        const saved=fs.createWriteStream(rawPath);
        reply.on('data',chunk=>digest.update(chunk));
        reply.on('end',()=>{record.sha256=digest.digest('hex');record.complete=true;service.onServed?.(record);});
        reply.on('error',error=>{record.error=error.message;res.destroy(error);});
        reply.on('aborted',()=>{record.error='upstream response aborted';saved.end();res.destroy();});
        saved.on('error',error=>{record.evidenceError=error.message;res.destroy(error);});
        res.writeHead(reply.statusCode,reply.headers);reply.pipe(saved);reply.pipe(res);
      });
      upstream.on('error',error=>{record.error=error.message;if(!res.headersSent)res.writeHead(503,{'Content-Type':'text/plain'}).end('actual foreground service unavailable: '+error.message);else res.destroy(error);});
      req.pipe(upstream);
    };
    return Object.assign(service,{child,receipt,proxy,close:()=>stopOwnedBrowser(child)});
  }catch(error){error.foregroundServiceCleanup=await stopOwnedBrowser(child);throw error;}
}
