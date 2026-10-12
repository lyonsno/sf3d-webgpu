import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';

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
export function createNativeDinoObservation({table,sourceETag,outputDir}){
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
      exact(row.status==='admitted','fresh additional-backing admission refused');
      return {index:row.index,demand};
    },
    persistTensor(index,tensorIndex,bytes){
      const row=observations.at(-1);
      exact(row?.index===index&&['admitted','collecting'].includes(row.status),'no active admitted observation or duplicate delivery');
      const tensor=row.tensors[tensorIndex];
      exact(tensorIndex===row.outputs.length&&tensor&&bytes.length===tensor.byteLength,'partial, duplicate or out-of-order whole tensor');
      const filename=path.join(outputDir,String(index).padStart(2,'0'),String(tensorIndex).padStart(3,'0')+'.f32');
      fs.mkdirSync(path.dirname(filename),{recursive:true});
      row.status='collecting';
      const output={name:tensor.name,path:filename,bytes:bytes.length,status:'writing'};row.outputs.push(output);
      try{
        fs.writeFileSync(filename,bytes,{flag:'wx'});
        exact(fs.statSync(filename).size===tensor.byteLength,'partial retained tensor');
        output.sha256=createHash('sha256').update(bytes).digest('hex');
        let finite=0;for(let i=0;i<bytes.length;i+=4)if(Number.isFinite(bytes.readFloatLE(i)))finite++;
        output.finite=finite;exact(finite===bytes.length/4,'nonfinite retained tensor');
        output.status='retained';if(row.outputs.length===row.tensors.length)row.status='complete';
      }catch(error){row.status='failed';output.error=String(error.message??error);throw error;}
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
