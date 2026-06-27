// Transpose [N, D] → [D, N] (or equivalently [D, H, W] from [H*W, D])
// Used to convert backbone output from token-major to channel-major layout.

struct Params {
  rows: u32,  // N (tokens)
  cols: u32,  // D (dimension)
  numWorkgroupsX: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;   // [rows, cols]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [cols, rows]

const WG_SIZE: u32 = 256;

@compute @workgroup_size(WG_SIZE)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * WG_SIZE + lid.x;
  let total = params.rows * params.cols;

  if (idx >= total) { return; }

  let row = idx / params.cols;
  let col = idx % params.cols;

  // input[row, col] → output[col, row]
  output[col * params.rows + row] = input[row * params.cols + col];
}
