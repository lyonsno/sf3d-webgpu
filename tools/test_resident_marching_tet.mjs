import assert from 'node:assert/strict';
import {marchingTetrahedra} from '../src/lib/marching_tet.js';
import * as subject from '../src/lib/marching_tet.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
assert.equal(typeof subject.runResidentMarchingTetrahedra,'function',
  'complete canonical marching needs counted backing and an owned mesh consumer lifetime');
const budget=createLoaderMemoryBudget({cpuBytes:1024*1024,gpuBytes:1});
const grid=new Float32Array([0,0,0,1,0,0,0,1,0,0,0,1,1,1,1]);
const indices=new Int32Array([0,1,2,3,4,3,2,1,3,0,2,1]),offset=new Float32Array(grid.length).map((_,i)=>Math.sin(i));
const duties=[];
for(let bits=0;bits<32;bits++){
  const sdf=new Float32Array(5).map((_,i)=>(bits&(1<<i))?(i+1)/3:-(i+1)/7);
  for(const deformation of [null,offset]){
    const baseline=marchingTetrahedra(grid,sdf,indices,deformation),handle={};
    const result=await subject.runResidentMarchingTetrahedra({handle,memoryBudget:budget,gridVertices:grid,sdf,
      tetIndices:indices,vertexOffsets:deformation,onBeforePhase:async d=>duties.push(d)});
    assert.deepEqual(Buffer.from(result.vertices.buffer),Buffer.from(baseline.vertices.buffer),'exact vertex bytes/order/interpolation');
    assert.deepEqual(Buffer.from(result.faces.buffer),Buffer.from(baseline.faces.buffer),'exact triangle order/winding');
    assert.equal(result.numVertices,baseline.numVertices);assert.equal(result.numFaces,baseline.numFaces);
    assert.equal(budget.snapshot().cpu.liveBytes,result.vertices.byteLength+result.faces.byteLength,
      'only escaping mesh output remains charged to its named owner');
    assert.equal(handle._residentMarchingOwner.value,result);
    subject.disposeResidentMarchingTetrahedra(handle);
    assert.equal(budget.snapshot().cpu.liveBytes,0);
    assert.equal(result.vertices.length,0,'released handle cannot present uncharged old output as live');
  }
}
const handle={},sdf=new Float32Array([-1,1,-2,2,-3]);
await assert.rejects(subject.runResidentMarchingTetrahedra({handle,memoryBudget:budget,gridVertices:grid,sdf,
  tetIndices:indices,onBeforePhase:async()=>{throw Error('before scratch refusal');}}),/before scratch refusal/);
assert.equal(budget.snapshot().cpu.liveBytes,0);assert.equal(handle._residentMarchingOwner,null);
await assert.rejects(subject.runResidentMarchingTetrahedra({handle,memoryBudget:budget,gridVertices:grid,sdf,
  tetIndices:indices,onBeforePhase:async d=>{if(d.name==='marching-complete-mesh')throw 0;}}),e=>e===0);
assert.equal(budget.snapshot().cpu.liveBytes,0);assert.equal(handle._residentMarchingOwner,null);
await assert.rejects(subject.runResidentMarchingTetrahedra({handle,memoryBudget:budget,gridVertices:grid,sdf,
  tetIndices:new Int32Array([0,1,2,999]),onBeforePhase:async()=>assert.fail('bad index before allocation')}),/index/);
await assert.rejects(subject.runResidentMarchingTetrahedra({handle,memoryBudget:budget,gridVertices:grid,
  sdf:new Float32Array([NaN,1,2,3,4]),tetIndices:indices,onBeforePhase:async()=>assert.fail('nonfinite before allocation')}),/nonfinite/);
assert.ok(duties.every(d=>Number.isSafeInteger(d.rangeCpuBytes)&&d.workGpuBytes===0));
assert.ok(duties.some(d=>d.validTets>0&&d.name==='marching-counted-scratch'));
assert.equal(budget.snapshot().cpu.liveBytes,0);budget.restore();
console.log('PASS all occupancy cases exact complete mesh bytes, deformation/dedup/order, before-allocator refusal and escaping output custody');
