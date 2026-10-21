// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Y[r] = X[r] / sqrt(mean(X[r]²) + eps) · w — ggml_rms_norm + ggml_mul (clip.cpp build_norm, NORM_TYPE_RMS).
// One 256-thread workgroup per row; the sum in f32 (ggml sums in f64 on the CPU, f32 on Metal).

struct Params {
    rows: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let base = wg.x * p.dim;
    var s = 0.0;
    for (var i = li; i < p.dim; i += 256u) {
        let v = X[base + i];
        s = fma(v, v, s);
    }
    part[li] = s;
    workgroupBarrier();
    for (var n = 128u; n > 0u; n >>= 1u) {
        if (li < n) {
            part[li] += part[li + n];
        }
        workgroupBarrier();
    }
    let scale = 1.0 / sqrt(part[0] / f32(p.dim) + p.eps);
    for (var i = li; i < p.dim; i += 256u) {
        Y[base + i] = X[base + i] * scale * w[i];
    }
}
