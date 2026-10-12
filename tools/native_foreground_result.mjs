import fs from 'node:fs';
import path from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
const exact=(ok,message)=>{if(!ok)throw Error('foreground result: '+message);};
export function createForegroundResultReceiver({filename,runId,onState=()=>{}}){
 exact(path.isAbsolute(filename)&&typeof runId==='string'&&runId.length>0,'explicit owned output/run required');
 const state={schema:'sf3d.native-foreground-result-delivery.v0',runId,status:'awaiting-admission',deliveries:[]};
 const checkpoint=()=>{const r=onState(state);exact(!r?.then,'synchronous raw delivery checkpoint required');};
 return{state,
  async admit(request,observe){
   exact(state.status==='awaiting-admission'&&request?.runId===runId&&Number.isSafeInteger(request.bytes)&&request.bytes>0&&
    Number.isSafeInteger(4*request.bytes),'fresh full serialization identity required');
   state.expectedBytes=request.bytes;state.status='admitting';
   const demand={rangeCpuBytes:4*request.bytes,workGpuBytes:0,requiredBytes:4*request.bytes,
    authority:'additional complete JSON string/UTF8/crypto and transport backing; existing metadata remains in fresh baseline'};
   try{state.admission=await observe({name:'foreground-result-serialization',runId,bytes:request.bytes},demand);
    state.status=state.admission?.verdict==='admitted'?'admitted':'refused';checkpoint();
    exact(state.status==='admitted','serialization admission refused');return{demand};}
   catch(error){state.error=error.message;checkpoint();throw error;}
  },
  async receive(input,claimed){
   exact(state.admission?.verdict==='admitted'&&!state.receiving,'no active admitted body or concurrent delivery');
   const valid=state.status==='admitted'&&claimed?.runId===runId&&claimed.bytes===state.expectedBytes&&
    typeof claimed.sha256==='string'&&/^[a-f0-9]{64}$/.test(claimed.sha256);
   const duplicate=state.deliveries.length>0;
   const file=duplicate?filename+'.delivery-'+randomUUID():filename;
   fs.mkdirSync(path.dirname(file),{recursive:true});
   const row={path:file,status:'receiving',runId:claimed?.runId,claimedBytes:claimed?.bytes,expectedBytes:state.expectedBytes,
    claimedSha256:claimed?.sha256,bytes:0,receivedBytes:0};state.deliveries.push(row);state.receiving=true;
   let fd,hash=createHash('sha256');
   try{
    fd=fs.openSync(file,'wx');checkpoint();
    const chunks=input instanceof Uint8Array?[input]:input;
    for await(const incoming of chunks){
     exact(incoming instanceof Uint8Array,'non-byte raw body');const b=Buffer.from(incoming.buffer,incoming.byteOffset,incoming.byteLength);
     row.receivedBytes+=b.length;let offset=0;
     while(offset<b.length){const n=fs.writeSync(fd,b,offset,b.length-offset);exact(n>0,'partial storage write');
      hash.update(b.subarray(offset,offset+n));row.bytes+=n;offset+=n;}
    }
    fs.fsyncSync(fd);row.sha256=hash.digest('hex');hash=null;
    exact(!duplicate,'duplicate complete body');exact(valid,'body source identity mismatch');
    exact(row.bytes===state.expectedBytes&&fs.fstatSync(fd).size===state.expectedBytes,'partial or oversized full body');
    exact(row.sha256===claimed.sha256,'full body hash mismatch');
    row.status='retained';state.status='retained';
    state.reference={schema:'sf3d.native-foreground-run-reference.v0',status:'retained',runId,path:file,bytes:row.bytes,sha256:row.sha256};
    checkpoint();return state.reference;
   }catch(error){
    if(hash)row.sha256=hash.digest('hex');row.status='failed';row.error=error.message;state.status='failed';checkpoint();throw error;
   }finally{if(fd!==undefined)fs.closeSync(fd);state.receiving=false;}
  },
  async restore(exit,ownedPid){
   exact(exit?.exitObserved===true&&exit.ownedPid===ownedPid&&Number.isSafeInteger(ownedPid),'exact owned browser exit before expansion required');
   exact(state.status==='retained'&&state.reference,'complete full raw result required');
   const bytes=fs.readFileSync(state.reference.path);exact(bytes.length===state.reference.bytes&&
    createHash('sha256').update(bytes).digest('hex')===state.reference.sha256,'retained raw source changed');
   const value=JSON.parse(bytes.toString('utf8'));
   exact(value?.runId===runId&&value.status==='finished'&&value.sameDevice===true&&value.sameQueue===true&&
    Array.isArray(value.frames)&&value.serviceReport,'restored owned result identity mismatch');
   return value;
  }
 };
}
