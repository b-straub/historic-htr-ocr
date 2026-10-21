// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The decode step's FFN input in one dispatch: ffn_norm (fused, as in dmv.wgsl's //@NORM) → gate and up projections
// (q8_0, the same rows of both per group: RA = 2R) → G = silu(gate) · up (swiglu.wgsl's formula). No h, no up buffer,
// no swiglu dispatch.

struct Params {
    N: u32,
    K: u32,
    eps: f32,
    _p0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> NW: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> Wg: array<u32>;
@group(0) @binding(4) var<storage, read> Wu: array<u32>;
@group(0) @binding(5) var<storage, read_write> G: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li & 31u;
    let row0 = wg.x * 8u * R + (li >> 5u) * R;
    let nb = p.K >> 5u;
    let scale = norm_scale(li, p.K, p.eps);
    var acc: array<f32, RA>;
    for (var b = lane; b < nb; b += 32u) {
        var xs: array<vec4<f32>, 8>;
        for (var j = 0u; j < 8u; j++) {
            xs[j] = X[b * 8u + j] * scale * NW[b * 8u + j];
        }
        for (var r = 0u; r < R; r++) {
            if (row0 + r < p.N) {
                let blk = (row0 + r) * nb + b;
                acc[r] += dot_Wg(blk, &xs);
                acc[R + r] += dot_Wu(blk, &xs);
            }
        }
    }
    row_sums(li, &acc);
    if (li < 8u * R) {
        let row = wg.x * 8u * R + li;
        if (row < p.N) {
            let g = tot[(li / R) * RA + li % R];
            let u = tot[(li / R) * RA + R + li % R];
            G[row] = g / (1.0 + exp(-g)) * u;
        }
    }
}
