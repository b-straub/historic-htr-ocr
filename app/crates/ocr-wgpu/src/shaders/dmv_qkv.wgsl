// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The decode step's attention input in one dispatch: attn_norm (fused, dmv.wgsl's //@NORM) → q, k, v projections
// (q8_0) into ONE buffer Y = [q (nq) | k (nkv) | v (nkv)]. A workgroup's 8R rows lie in one matrix (nq, nkv are
// multiples of 8R — checked in text.rs), so the matrix choice is uniform per workgroup.

struct Params {
    nq: u32,
    nkv: u32,
    K: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> NW: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> Wq: array<u32>;
@group(0) @binding(4) var<storage, read> Wk: array<u32>;
@group(0) @binding(5) var<storage, read> Wv: array<u32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li & 31u;
    let wrow0 = wg.x * 8u * R;
    var which = 0u;
    var first = 0u;
    var n = p.nq;
    if (wrow0 >= p.nq + p.nkv) {
        which = 2u;
        first = p.nq + p.nkv;
        n = p.nkv;
    } else if (wrow0 >= p.nq) {
        which = 1u;
        first = p.nq;
        n = p.nkv;
    }
    let row0 = wrow0 - first + (li >> 5u) * R; // in the matrix
    let nb = p.K >> 5u;
    let scale = norm_scale(li, p.K, p.eps);
    var acc: array<f32, RA>;
    for (var b = lane; b < nb; b += 32u) {
        var xs: array<vec4<f32>, 8>;
        for (var j = 0u; j < 8u; j++) {
            xs[j] = X[b * 8u + j] * scale * NW[b * 8u + j];
        }
        for (var r = 0u; r < R; r++) {
            if (row0 + r < n) {
                let blk = (row0 + r) * nb + b;
                if (which == 0u) {
                    acc[r] += dot_Wq(blk, &xs);
                } else if (which == 1u) {
                    acc[r] += dot_Wk(blk, &xs);
                } else {
                    acc[r] += dot_Wv(blk, &xs);
                }
            }
        }
    }
    row_sums(li, &acc);
    if (li < 8u * R && wrow0 - first + li < n) {
        Y[wrow0 + li] = tot[li];
    }
}
