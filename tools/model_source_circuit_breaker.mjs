// Source-host refusal, not browser hardware discovery or positive fit authority.
// Propagate the existing low-memory full-route hold before Vite serves payload.
import fs from 'node:fs';
import path from 'node:path';
import {observeMacMemory,defaultPlanPathForObservation,runMemoryAdmission,applyExecutableMemoryCircuitBreaker} from './memory_admission.mjs';

export function modelSourceCircuitBreaker({observe=observeMacMemory,admit=runMemoryAdmission}={}){
  let config;
  const install=server=>{server.middlewares.use((req,res,next)=>{
    let requestedPath;
    try{requestedPath=decodeURIComponent(new URL(req.url,'http://localhost').pathname);}catch{return next();}
    const weightPath=path.resolve(config.root,'public/weights.bin');
    const real=value=>{try{return fs.realpathSync(value);}catch{return path.resolve(value);}};
    const target=real(weightPath);
    const candidates=requestedPath.startsWith('/@fs/')?[requestedPath.slice(4)]:[
      path.resolve(config.root,'.'+requestedPath),
      ...(config.publicDir?[path.resolve(config.publicDir,'.'+requestedPath)]:[]),
    ];
    if(requestedPath!=='/weights.bin'&&!candidates.some(candidate=>real(candidate)===target))return next();
    let observation;
    try{
      observation=observe({volumePath:path.dirname(weightPath)});
      if(!defaultPlanPathForObservation(observation,config.root))return next();
      const memoryAdmission=applyExecutableMemoryCircuitBreaker({memoryAdmission:admit({repoRoot:config.root,weightPath,observe:()=>observation})});
      // This endpoint never turns diagnostic headroom into allocation authority.
      if(memoryAdmission.verdict!=='refused'||memoryAdmission.authority!=='circuit-breaker-only')throw Error('source full-route circuit breaker lost refusal authority');
      respond({schema:'sf3d.model-source-circuit-breaker.v0',verdict:'refused',authority:'circuit-breaker-only',
        repoRoot:config.root,requestedPath,sourcePath:target,memoryAdmission,
        meaning:'unadmitted full-model source refused on this source host; not browser RAM discovery or positive phase admission'});
    }catch(error){respond({schema:'sf3d.model-source-circuit-breaker.v0',verdict:'refused',authority:'circuit-breaker-only',
      repoRoot:config.root,requestedPath,sourcePath:target,observation,error:error.message,
      meaning:'source observation/diagnostic unavailable; no model bytes served'});}
    function respond(report){
      const body=JSON.stringify(report);
      res.writeHead(503,{'Content-Type':'application/json','Cache-Control':'no-store','X-SF3D-Memory-Authority':'circuit-breaker-only','Content-Length':Buffer.byteLength(body)});
      res.end(req.method==='HEAD'?undefined:body);
    }
  });};
  return{name:'sf3d-model-source-circuit-breaker',configResolved(value){config=value;},configureServer:install,configurePreviewServer:install};
}
