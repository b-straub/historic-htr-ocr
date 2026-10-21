// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// X = gelu(X) in place, the tanh form of ggml_gelu (the projector's FFN_GELU, LlavaMultiModalProjector):
// 0.5·x·(1 + tanh(√(2/π)·x·(1 + 0.044715·x²))). The tanh argument is clamped to ±15 (tanh is ±1 to f32 precision
// there) so no backend's tanh can overflow to NaN.

struct Params {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const GELU_COEF_A: f32 = 0.044715;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var i = gid.x; i < p.n; i += nwg.x * 256u) {
        let x = X[i];
        let t = clamp(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x), -15.0, 15.0);
        X[i] = 0.5 * x * (1.0 + tanh(t));
    }
}
