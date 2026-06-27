/**
 * Modulated LayerNorm shader for SF3D's DINOv2 with AdaNorm.
 *
 * Applies standard LayerNorm, then modulates the output:
 *   output = (1 + scale) * LayerNorm(input) + shift
 *
 * where [scale, shift] = linear(modulation_cond), each of dim D.
 * The modulation linear produces [2*D] which is split into scale and shift.
 *
 * This is a single-dispatch variant: one workgroup per token.
 */

struct Params {
  N: u32,      // number of tokens
  D: u32,      // hidden dimension
  eps: f32,    // LayerNorm epsilon
  hasModulation: u32, // 0 = standard LayerNorm, 1 = modulated
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;      // [N, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;      // [D]
@group(0) @binding(3) var<storage, read> beta: array<f32>;       // [D]
@group(0) @binding(4) var<storage, read> modulation: array<f32>; // [2*D] (scale, shift) or [1] if unused
@group(0) @binding(5) var<storage, read_write> output: array<f32>; // [N, D]

@compute @workgroup_size(1)
fn layernorm_mod_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.N) { return; }

  let D = params.D;
  let base = row * D;

  // Pass 1: compute mean
  var sum: f32 = 0.0;
  for (var d: u32 = 0; d < D; d++) {
    sum += input[base + d];
  }
  let mean = sum / f32(D);

  // Pass 2: compute variance
  var varSum: f32 = 0.0;
  for (var d: u32 = 0; d < D; d++) {
    let diff = input[base + d] - mean;
    varSum += diff * diff;
  }
  let variance = varSum / f32(D);
  let invStd = 1.0 / sqrt(variance + params.eps);

  // Pass 3: normalize, apply gamma/beta, then modulation
  if (params.hasModulation == 1u) {
    for (var d: u32 = 0; d < D; d++) {
      let normalized = (input[base + d] - mean) * invStd;
      let ln_out = gamma[d] * normalized + beta[d];
      let scale = modulation[d];       // first D elements
      let shift = modulation[D + d];   // second D elements
      output[base + d] = (1.0 + scale) * ln_out + shift;
    }
  } else {
    for (var d: u32 = 0; d < D; d++) {
      let normalized = (input[base + d] - mean) * invStd;
      output[base + d] = gamma[d] * normalized + beta[d];
    }
  }
}
