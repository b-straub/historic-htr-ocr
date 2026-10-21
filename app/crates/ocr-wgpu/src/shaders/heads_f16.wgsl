// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The vision attention's K and V as f16, head-major and padded: X [n_tok][n_head · 64] f32 → H [n_head][n_pad][64]
// f16 (rows n_tok … n_pad − 1 zero). The values attention_coop.wgsl used to round per key block (vec4<f16>(K[…])) —
// rounded once per layer, read by the matrix units straight from memory as contiguous 8 × 64 blocks.
enable f16;

struct Params {
    n_tok: u32,
    n_head: u32,
    n_pad: u32,
    _p0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Kh: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read_write> Vh: array<vec4<f16>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let total = p.n_head * p.n_pad * 16u;
    for (var i = gid.x; i < total; i += nwg.x * 256u) {
        let d4 = i % 16u;
        let t = (i / 16u) % p.n_pad;
        let h = i / (16u * p.n_pad);
        var k = vec4<f16>(0.0);
        var v = vec4<f16>(0.0);
        if (t < p.n_tok) {
            let src = t * p.n_head * 16u + h * 16u + d4;
            k = vec4<f16>(K[src]);
            v = vec4<f16>(V[src]);
        }
        Kh[i] = k;
        Vh[i] = v;
    }
}
