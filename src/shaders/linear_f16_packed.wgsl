// Linear projection with IEEE-754 binary16 weights packed two per u32.
// Inputs, bias, accumulation, and output remain f32.

struct Params {
  numRows: u32,
  inDim: u32,
  outDim: u32,
  numWorkgroupsX: u32,
  transposed: u32,
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

  if (params.transposed == 1u) {
    let weightBase = col;
    let stride = params.outDim;
    for (var k = 0u; k < len4; k += 4u) {
      s0 += input[inBase + k]      * loadWeight((k)      * stride + weightBase);
      s1 += input[inBase + k + 1u] * loadWeight((k + 1u) * stride + weightBase);
      s2 += input[inBase + k + 2u] * loadWeight((k + 2u) * stride + weightBase);
      s3 += input[inBase + k + 3u] * loadWeight((k + 3u) * stride + weightBase);
    }
    for (var k = len4; k < params.inDim; k++) {
      s0 += input[inBase + k] * loadWeight(k * stride + weightBase);
    }
  } else {
    let weightBase = col * params.inDim;
    for (var k = 0u; k < len4; k += 4u) {
      s0 += input[inBase + k]      * loadWeight(weightBase + k);
      s1 += input[inBase + k + 1u] * loadWeight(weightBase + k + 1u);
      s2 += input[inBase + k + 2u] * loadWeight(weightBase + k + 2u);
      s3 += input[inBase + k + 3u] * loadWeight(weightBase + k + 3u);
    }
    for (var k = len4; k < params.inDim; k++) {
      s0 += input[inBase + k] * loadWeight(weightBase + k);
    }
  }
  output[idx] = (s0 + s1) + (s2 + s3) + bias[col];
}
