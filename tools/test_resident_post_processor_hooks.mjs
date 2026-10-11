import assert from 'node:assert/strict';
import {createPostProcessorChannelDutyPlan,drivePostProcessorChannelBoundary,
  POST_PROCESSOR_CHANNEL_BOUNDARY_ID} from '../src/lib/cooperative_post_processor.js';

const plan=createPostProcessorChannelDutyPlan(16);
assert.equal(plan.duties.length,702);
function fixture({rejectAt=-1}={}){
  const trace=[];let next=0;
  const cooperative={startBoundary(id){
    assert.equal(id,POST_PROCESSOR_CHANNEL_BOUNDARY_ID);
    return{nextRange(){return next<plan.duties.length?{itemStart:next,itemEnd:++next}:null},
      async runGpuDuty(range,work){
        trace.push(['encode',range.itemStart]);work.encode();
        if(range.itemStart===rejectAt)throw Error('observed prefix rejection');
        trace.push(['prefix',range.itemStart]);
      }};
  }};
  return {trace,cooperative};
}
const f=fixture();let live=null;
const result=await drivePostProcessorChannelBoundary(f.cooperative,{
  plan,
  async withDutyGroup(group,work){
    assert.equal(live,null);live=group;
    f.trace.push(['group-enter',group.plane,group.stageIndex]);
    try{return await work()}finally{f.trace.push(['group-exit',group.plane,group.stageIndex]);live=null}
  },
  async beforeDuty(duty){assert.equal(live?.plane,duty.plane);assert.equal(live?.stageIndex,duty.stageIndex);f.trace.push(['before',duty.dutyIndex])},
  encodeDuty(duty){return {dutyIndex:duty.dutyIndex}},
  async afterDuty(duty){assert.deepEqual(f.trace.at(-1),['prefix',duty.dutyIndex]);f.trace.push(['after',duty.dutyIndex])},
});
assert.equal(f.trace.filter(x=>x[0]==='before').length,702,'fresh pre-duty hook must cover all 702 duties');
assert.equal(f.trace.filter(x=>x[0]==='after').length,702,'settled post-duty hook must cover all 702 duties');
assert.equal(f.trace.filter(x=>x[0]==='group-enter').length,18);
assert.equal(f.trace.filter(x=>x[0]==='group-exit').length,18);
assert.equal(result.completedDuties,702);
for(const duty of plan.duties){
  const b=f.trace.findIndex(x=>x[0]==='before'&&x[1]===duty.dutyIndex);
  assert.deepEqual(f.trace.slice(b,b+4),[['before',duty.dutyIndex],['encode',duty.dutyIndex],['prefix',duty.dutyIndex],['after',duty.dutyIndex]]);
}
const failed=fixture({rejectAt:2});let released=0;
await assert.rejects(drivePostProcessorChannelBoundary(failed.cooperative,{
  plan,encodeDuty(){return {}},
  async withDutyGroup(group,work){try{return await work()}finally{released++}},
  afterDuty(duty){failed.trace.push(['after',duty.dutyIndex])},
}),/observed prefix rejection/);
assert.equal(released,2,'gather and rejected first-convolution groups both unwind');
assert.equal(failed.trace.some(x=>x[0]==='after'&&x[1]===2),false);
assert.equal(failed.trace.some(x=>x[0]==='encode'&&x[1]===3),false);
const omitted=fixture();
await assert.rejects(drivePostProcessorChannelBoundary(omitted.cooperative,{
  plan,encodeDuty(){throw Error('must not encode')},withDutyGroup(){},
}),/did not execute completely/);
assert.equal(omitted.trace.length,0);
const denied=fixture();
await assert.rejects(drivePostProcessorChannelBoundary(denied.cooperative,{
  plan,encodeDuty(){throw Error('must not encode')},
  beforeDuty(){throw Error('fresh refusal')},
}),/fresh refusal/);
assert.equal(denied.trace.length,0);
console.log('PASS complete postprocessor observation/group/prefix contract');

