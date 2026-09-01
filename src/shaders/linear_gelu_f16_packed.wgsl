// Linear projection + GELU with IEEE-754 binary16 weights packed two per u32.
// Inputs, bias, accumulation, activation, and output remain f32.

struct Params {
  numRows: u32,
  inDim: u32,
  outDim: u32,
  numWorkgroupsX: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 256;

fn loadWeight(index: u32) -> f32 {
  let pair = unpack2x16float(weight[index >> 1u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}

fn gelu(x: f32) -> f32 {
  if (x > 10.0) { return x; }
  if (x < -10.0) { return 0.0; }
  let a = x * 0.7071067811865476;
  let s = sign(a);
  let tAbs = abs(a);
  let t = 1.0 / (1.0 + 0.3275911 * tAbs);
  let t2 = t * t;
  let t3 = t2 * t;
  let t4 = t3 * t;
  let t5 = t4 * t;
  let erfAbs = 1.0 - (0.254829592 * t - 0.284496736 * t2 + 1.421413741 * t3
    - 1.453152027 * t4 + 1.061405429 * t5) * exp(-tAbs * tAbs);
  return 0.5 * x * (1.0 + s * erfAbs);
}

@compute @workgroup_size(WG_SIZE)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let linearWG = wgid.x + wgid.y * params.numWorkgroupsX;
  let idx = linearWG * WG_SIZE + lid.x;
  if (idx >= params.numRows * params.outDim) { return; }

  let row = idx / params.outDim;
  let col = idx % params.outDim;
  var s0 = 0.0;
  var s1 = 0.0;
  var s2 = 0.0;
  var s3 = 0.0;
  let inBase = row * params.inDim;
  let len4 = (params.inDim / 4u) * 4u;
  for (var k = 0u; k < len4; k += 4u) {
    s0 += input[inBase + k]      * loadWeight((k)      * params.outDim + col);
    s1 += input[inBase + k + 1u] * loadWeight((k + 1u) * params.outDim + col);
    s2 += input[inBase + k + 2u] * loadWeight((k + 2u) * params.outDim + col);
    s3 += input[inBase + k + 3u] * loadWeight((k + 3u) * params.outDim + col);
  }
  for (var k = len4; k < params.inDim; k++) {
    s0 += input[inBase + k] * loadWeight(k * params.outDim + col);
  }
  output[idx] = gelu((s0 + s1) + (s2 + s3) + bias[col]);
}
