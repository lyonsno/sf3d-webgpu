struct Params {
  channels: u32,
  size: u32,
  scale: u32,
  plane: u32,
  rowStart: u32,
  rowCount: u32,
  workgroupsX: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn gather(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) local: vec3u) {
  let i = (group.x + group.y * p.workgroupsX) * 256u + local.x;
  let pixels = p.size * p.size;
  if (i >= p.channels * pixels) { return; }
  let c = i / pixels;
  output[i] = input[c * 3u * pixels + p.plane * pixels + i % pixels];
}

@compute @workgroup_size(256)
fn shuffle(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) local: vec3u) {
  let i = (group.x + group.y * p.workgroupsX) * 256u + local.x;
  let width = p.size * p.scale;
  let rows = p.rowCount * p.scale;
  if (i >= p.channels * rows * width) { return; }
  let c = i / (rows * width);
  let y = (i / width) % rows + p.rowStart * p.scale;
  let x = i % width;
  let ic = c * p.scale * p.scale + (y % p.scale) * p.scale + x % p.scale;
  let source = ic * p.size * p.size + (y / p.scale) * p.size + x / p.scale;
  let destination = (p.plane * p.channels + c) * width * width + y * width + x;
  output[destination] = input[source];
}
