import fs from 'node:fs';
import path from 'node:path';
import {createHash,randomUUID} from 'node:crypto';

const exact=(condition,message)=>{if(!condition)throw Error('DINO observation: '+message);};
const feature=(name,shape)=>({name,shape,byteLength:shape.reduce((n,v)=>n*v,1)*4});
const plan=[{kind:'weights',phase:'camera'},
  {kind:'checkpoint',phase:'camera',features:[feature('cameraEmbed',[768])]}];
for(let block=0;block<24;block++){
  plan.push({kind:'weights',phase:'dino-block-'+block});
  if(block<3||block===23)plan.push({kind:'checkpoint',phase:'dino-block-'+block,
    features:[...(block===0?[feature('patchEmbed',[1297,1024])]:[]),feature('block'+block,[1297,1024])]});
}
const equalShape=(a,b)=>Array.isArray(a)&&Array.isArray(b)&&a.length===b.length&&a.every((v,i)=>v===b[i]);
function phaseNames(table,phase){
  return [...table.keys()].filter(name=>{
    if(phase==='camera')return name.startsWith('camera_embedder.linear.');
    const block=Number(phase.slice('dino-block-'.length));
    return name.startsWith('image_tokenizer.model.encoder.layer.'+block+'.')||
      (block===0&&name.startsWith('image_tokenizer.model.embeddings.'))||
      (block===23&&name.startsWith('image_tokenizer.model.layernorm.'));
  });
}

/** Separate opt-in readbacks; never substitutes for any full-model phase. */
export function loadDinoConversionHashes(replay,canonical,table){
  exact(replay?.schema==='sf3d.canonical-dino-conversion-replay.v0'&&replay.status==='passed'&&replay.phase==='complete',
    'complete conversion reference missing');
  exact(replay.source?.wholeSHA256FromNative===canonical.sha256&&replay.source.headerSHA256===canonical.headerSha256&&
    replay.source.byteLength===canonical.byteLength,'conversion reference canonical source mismatch');
  const expected=new Set(plan.filter(p=>p.kind==='weights').flatMap(p=>phaseNames(table,p.phase)));
  exact(expected.size===536&&replay.scope?.phases===25&&replay.scope.tensors===536&&replay.rows?.length===1072,
    'complete536 conversion reference scope mismatch');
  const modes=new Map();let rawBytes=0,expandedBytes=0;
  for(const row of replay.rows){
    exact(['mapped','streamed'].includes(row.mode)&&expected.has(row.name),'unknown reference mode or selected tensor');
    const t=table.get(row.name),key=row.mode+'\0'+row.name;
    exact(!modes.has(key)&&phaseNames(table,row.phase).includes(row.name)&&row.bytes===t.size*(t.dtype===1?2:1)&&
      row.words*4===row.bytes&&/^[a-f0-9]{64}$/.test(row.convertedSHA256)&&/^[a-f0-9]{64}$/.test(row.rawSHA256),
      'partial, duplicate or malformed conversion reference');
    modes.set(key,row);
    if(row.mode==='mapped'){rawBytes+=t.size;expandedBytes+=row.bytes;}
  }
  const hashes=new Map();
  for(const name of expected){
    const a=modes.get('mapped\0'+name),b=modes.get('streamed\0'+name);
    exact(a&&b&&a.convertedSHA256===b.convertedSHA256&&a.rawSHA256===b.rawSHA256,'conversion reference routes disagree');
    hashes.set(name,a.convertedSHA256);
  }
  exact(replay.scope.rawBytes===rawBytes&&replay.scope.expandedBytes===expandedBytes,'conversion reference full-byte totals mismatch');
  return hashes;
}

export function createNativeDinoObservation({table,sourceETag,outputDir,expectedWeightHashes,onEvent=()=>{}}){
  exact(expectedWeightHashes instanceof Map,'explicit converted canonical weight hashes required');
  exact(typeof onEvent==='function','synchronous diagnostic journal required');
  const emit=(kind,payload)=>{const result=onEvent('dino-observation-'+kind,payload);
    exact(!result?.then,'synchronous diagnostic journal required');};
  const observations=[];
  return {observations,
    async admit(request,currentPhase,observe){
      exact(!observations.length||observations.at(-1).status==='complete','previous observation incomplete');
      const expected=plan[observations.length];
      exact(expected&&request?.kind===expected.kind,'unexpected kind or checkpoint');
      exact(currentPhase?.verdict==='admitted'&&currentPhase.phase===expected.phase&&
        currentPhase.descriptor?.name===expected.phase,'effective admitted model phase mismatch');
      const tensors=request.tensors;
      exact(Array.isArray(tensors)&&tensors.length>0,'missing complete tensors');
      if(expected.kind==='weights'){
        exact(request.sourceETag===sourceETag,'source ETag mismatch');
        const actual=currentPhase.descriptor.tensors,names=phaseNames(table,expected.phase);
        exact(actual?.length===names.length&&new Set(actual.map(t=>t.name)).size===names.length&&
          names.every(name=>actual.some(t=>t.name===name))&&tensors.length===actual.length,'complete selected weight set mismatch');
        tensors.forEach((t,i)=>{
          const canonical=table.get(t.name),p=actual[i];
          exact(canonical&&t.name===p.name&&[0,1].includes(canonical.dtype)&&
            ['dtype','offset','size'].every(key=>t[key]===canonical[key]&&p[key]===canonical[key])&&
            equalShape(t.shape,canonical.shape)&&equalShape(p.shape,canonical.shape)&&
            t.byteLength===canonical.size*(canonical.dtype===1?2:1)&&
            t.byteLength===canonical.shape.reduce((n,v)=>n*v,1)*4,'weight metadata, order or expanded size mismatch');
        });
      }else{
        exact(tensors.length===expected.features.length&&tensors.every((t,i)=>
          t.name===expected.features[i].name&&equalShape(t.shape,expected.features[i].shape)&&
          t.byteLength===expected.features[i].byteLength),'complete checkpoint shape/order mismatch');
      }
      const largest=Math.max(...tensors.map(t=>t.byteLength));
      exact(Number.isSafeInteger(largest)&&largest>0&&largest%4===0,'invalid full readback size');
      // One whole tensor at a time: GPU staging plus mapped/upload/server body
      // working stores. Every tensor is retained; this is not a result cap.
      const demand={workGpuBytes:largest,rangeCpuBytes:4*largest,requiredBytes:5*largest,
        authority:'additional whole-tensor diagnostic readback/persistence backing; no retired or physical credit'};
      const row={index:observations.length,kind:expected.kind,phase:expected.phase,
        tensors:JSON.parse(JSON.stringify(tensors)),demand,status:'awaiting-admission',outputs:[]};
      observations.push(row);
      row.admission=await observe({name:'dino-observation-'+row.index,kind:row.kind,tensors:row.tensors},demand);
      row.status=row.admission?.verdict==='admitted'?'admitted':'refused';
      emit('admission',{index:row.index,kind:row.kind,phase:row.phase,tensors:row.tensors,demand,admission:row.admission,status:row.status});
      exact(row.status==='admitted','fresh additional-backing admission refused');
      return {index:row.index,demand};
    },
    async admitTensor(index,tensorIndex,observe){
      const row=observations.at(-1),tensor=row?.tensors[tensorIndex];
      exact(row?.index===index&&['admitted','collecting'].includes(row.status)&&!row.delivering&&!row.tensorAdmitting&&
        tensor&&tensorIndex===row.outputs.length,'no active admitted tensor or duplicate delivery');
      exact(!row.nextTensorAdmission,'duplicate tensor admission');
      const demand={workGpuBytes:0,rangeCpuBytes:4*tensor.byteLength,requiredBytes:4*tensor.byteLength,
        authority:'new CPU transfer/persistence work; actual staging already exists and remains in fresh baseline; no retired credit'};
      row.tensorAdmitting=true;
      const record={tensorIndex,status:'awaiting-admission'};(row.tensorAdmissions??=[]).push(record);
      try{
        record.admission=await observe({name:'dino-observation-'+index+'-tensor-'+tensorIndex,tensors:[tensor]},demand);
        record.status=record.admission?.verdict==='admitted'?'admitted':'refused';
        emit('tensor-admission',{observationIndex:index,...record});
        exact(record.status==='admitted','fresh per-tensor admission refused');
        row.nextTensorAdmission=record;return {demand};
      }catch(error){row.status=record.status==='refused'?'refused':'failed';record.error=String(error?.message??error);throw error;}
      finally{row.tensorAdmitting=false;}
    },
    async persistTensor(index,tensorIndex,input){
      const row=observations.at(-1);
      exact(row?.index===index&&!row.delivering,'no active admitted observation or duplicate delivery');
      const tensor=row.tensors[tensorIndex];
      const valid=['admitted','collecting'].includes(row.status)&&tensorIndex===row.outputs.length&&tensor&&
        row.nextTensorAdmission?.tensorIndex===tensorIndex;
      const filename=path.join(outputDir,String(index).padStart(2,'0'),'received-'+randomUUID()+'.f32');
      fs.mkdirSync(path.dirname(filename),{recursive:true});
      const output={name:tensor?.name,path:filename,bytes:0,status:'receiving',
        claimedObservationIndex:index,claimedTensorIndex:tensorIndex,expectedBytes:tensor?.byteLength};
      row.delivering=true;
      let fd,hash=createHash('sha256'),finite=0,carry=Buffer.alloc(4),carryBytes=0;
      try{
        fd=fs.openSync(filename,'wx');
        emit('tensor-receiving',{observationIndex:index,tensorIndex,output});
        const chunks=Buffer.isBuffer(input)||input instanceof Uint8Array?[input]:input;
        for await(const incoming of chunks){
          exact(incoming instanceof Uint8Array,'non-byte diagnostic transport');
          const chunk=Buffer.from(incoming.buffer,incoming.byteOffset,incoming.byteLength);
          let written=0;while(written<chunk.length){const n=fs.writeSync(fd,chunk,written,chunk.length-written);exact(n>0,'partial raw evidence write');written+=n;}
          hash.update(chunk);output.bytes+=chunk.length;
          // Canonical weight equality is checked by hash, avoiding millions of
          // Buffer scalar conversions and full-body copies in the observer.
          if(row.kind==='checkpoint'){
            let offset=0;
            while(carryBytes&&offset<chunk.length){carry[carryBytes++]=chunk[offset++];
              if(carryBytes===4){if(Number.isFinite(carry.readFloatLE(0)))finite++;carryBytes=0;}}
            const words=Math.floor((chunk.length-offset)/4),view=new DataView(chunk.buffer,chunk.byteOffset+offset,words*4);
            for(let i=0;i<words;i++)if(Number.isFinite(view.getFloat32(i*4,true)))finite++;
            offset+=words*4;while(offset<chunk.length)carry[carryBytes++]=chunk[offset++];
          }
        }
        fs.fsyncSync(fd);output.sha256=hash.digest('hex');hash=null;
        exact(valid,'no active admitted tensor or duplicate/out-of-order delivery');
        exact(output.bytes===tensor.byteLength&&fs.fstatSync(fd).size===tensor.byteLength,'partial or oversized retained tensor');
        if(row.kind==='weights'){
          output.expectedSha256=expectedWeightHashes.get(tensor.name);
          exact(output.sha256===output.expectedSha256,'actual GPU weight differs from canonical conversion hash');
          output.validation='complete GPU bytes equal independently checked canonical conversion';
        }else{output.finite=finite;exact(!carryBytes&&finite===output.bytes/4,'nonfinite retained tensor');}
        output.status='retained';row.outputs.push(output);row.nextTensorAdmission=null;
        row.status=row.outputs.length===row.tensors.length?'complete':'collecting';
        emit('tensor-retained',{observationIndex:index,tensorIndex,output});
      }catch(error){
        if(hash){output.sha256=hash.digest('hex');hash=null;}
        row.status='failed';output.status='failed';output.error=String(error.message??error);
        (row.failedDeliveries??=[]).push(output);
        try{emit('tensor-failed',{observationIndex:index,tensorIndex,output});}
        catch(journalError){throw new AggregateError([error,journalError],'diagnostic body and failed-evidence journal failed',{cause:error});}
        throw error;
      }finally{if(fd!==undefined)fs.closeSync(fd);row.delivering=false;}
      return output;
    },
    assertComplete(){
      exact(observations.length===plan.length&&observations.every(o=>o.status==='complete'),'complete diagnostic scope incomplete');
      const weights=observations.filter(o=>o.kind==='weights').flatMap(o=>o.outputs);
      exact(weights.length===536&&new Set(weights.map(t=>t.name)).size===536,'complete536 actual weights missing');
      exact(observations.filter(o=>o.kind==='checkpoint').flatMap(o=>o.outputs).length===6,'complete six actual checkpoints missing');
    },
  };
}
