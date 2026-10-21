// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// y[N] (+)= W[N,K] · x[K] with W q8_0 — the decode step's projections on the F16 paths (the F32 reference keeps
// matvec_q8_0.wgsl). 8R rows per workgroup (dmv_common.wgsl). The //@NORM variant (dmv_norm) reads x through the RMS
// norm first: x · scale · NW, scale computed per workgroup (rms_norm.wgsl's formula) — no norm dispatch, no h buffer.
// acc = 1 adds onto y (the residual stream).

struct Params {
    N: u32,
    K: u32,
    acc: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
//@NORM @group(0) @binding(2) var<storage, read> NW: array<vec4<f32>>;
@group(0) @binding(B_W) var<storage, read> W: array<u32>;
@group(0) @binding(B_Y) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li & 31u;
    let row0 = wg.x * 8u * R + (li >> 5u) * R;
    let nb = p.K >> 5u;
    //@NORM let scale = norm_scale(li, p.K, p.eps);
    var acc: array<f32, RA>;
    for (var b = lane; b < nb; b += 32u) {
        var xs: array<vec4<f32>, 8>;
        for (var j = 0u; j < 8u; j++) {
            xs[j] = X[b * 8u + j];
            //@NORM xs[j] = xs[j] * scale * NW[b * 8u + j];
        }
        for (var r = 0u; r < R; r++) {
            if (row0 + r < p.N) {
                acc[r] += dot_W((row0 + r) * nb + b, &xs);
            }
        }
    }
    row_sums(li, &acc);
    if (li < 8u * R) {
        let row = wg.x * 8u * R + li;
        if (row < p.N) {
            if (p.acc == 1u) {
                Y[row] = Y[row] + tot[li];
            } else {
                Y[row] = tot[li];
            }
        }
    }
}
