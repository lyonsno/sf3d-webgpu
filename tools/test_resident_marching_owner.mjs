import assert from 'node:assert/strict';
import {runResidentMarchingTetrahedra,disposeResidentMarchingTetrahedra} from '../src/lib/marching_tet.js';
import {createLoaderMemoryBudget} from '../src/lib/loader_memory_budget.js';
for(const boundary of ['marching-counted-scratch','marching-complete-mesh']){
  const handle={},budget=createLoaderMemoryBudget({cpuBytes:4096,gpuBytes:1});
  let eventsAfterDispose;
  await assert.rejects(runResidentMarchingTetrahedra({handle,memoryBudget:budget,
    gridVertices:new Float32Array([0,0,0,1,0,0,0,1,0,0,0,1]),sdf:new Float32Array([-1,1,-1,1]),
    tetIndices:new Int32Array([0,1,2,3]),onBeforePhase:async d=>{
      if(d.name===boundary){disposeResidentMarchingTetrahedra(handle);eventsAfterDispose=budget.events.length;}
    }}),/marching owner disposed during guard/,
    'a disposed pending owner cannot continue allocating or return detached charged mesh');
  assert.equal(handle._residentMarchingOwner,null);
  assert.equal(budget.snapshot().cpu.liveBytes,0);
  assert.deepEqual(budget.events.slice(eventsAfterDispose),[],'no backing or retirement credited after disposed guard continuation');
  budget.restore();
}
// Disposal may be followed by lawful reuse while the old call is still awaiting
// its callback. The old continuation must not dispose the successor's output.
{
  const handle={},budget=createLoaderMemoryBudget({cpuBytes:4096,gpuBytes:1});
  const input={handle,memoryBudget:budget,gridVertices:new Float32Array([0,0,0,1,0,0,0,1,0,0,0,1]),
    sdf:new Float32Array([-1,1,-1,1]),tetIndices:new Int32Array([0,1,2,3])};
  let next;
  await assert.rejects(runResidentMarchingTetrahedra({...input,onBeforePhase:async d=>{
    if(d.name==='marching-counted-scratch'){
      disposeResidentMarchingTetrahedra(handle);
      next=await runResidentMarchingTetrahedra({...input,onBeforePhase:async()=>{}});
    }
  }}),/marching owner disposed during guard/);
  assert.equal(handle._residentMarchingOwner.value,next);
  assert.equal(next.numVertices,4);assert.equal(next.numFaces,2);
  assert.equal(budget.snapshot().cpu.liveBytes,72);
  disposeResidentMarchingTetrahedra(handle);assert.equal(budget.snapshot().cpu.liveBytes,0);budget.restore();
}
console.log('PASS disposal during both awaited marching guards prevents detached allocations and unreachable charges');
