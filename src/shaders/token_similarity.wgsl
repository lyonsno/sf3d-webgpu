struct Params { dim: u32, count: u32, reference: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> tokens: array<f32>;
@group(0) @binding(1) var<storage, read_write> similarity: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> sums: array<vec3<f32>, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  var sum = vec3<f32>(0.0);
  for (var d = lane; d < params.dim; d += 64u) {
    let a = tokens[(group.x + 1u) * params.dim + d];
    let b = tokens[(params.reference + 1u) * params.dim + d];
    sum += vec3<f32>(a * b, a * a, b * b);
  }
  sums[lane] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let denom = sqrt(sums[0].y * sums[0].z);
    similarity[group.x] = select(0.0, clamp(sums[0].x / max(denom, 1e-20), -1.0, 1.0), denom > 0.0);
  }
}
