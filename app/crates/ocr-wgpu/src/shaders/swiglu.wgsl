// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// G = silu(G) · U in place — ggml_swiglu_split(gate, up) (clip.cpp build_ffn, FFN_SILU with a gate).
// silu(x) = x / (1 + e^-x).

struct Params {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read_write> G: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var i = gid.x; i < p.n; i += nwg.x * 256u) {
        let g = G[i];
        G[i] = g / (1.0 + exp(-g)) * U[i];
    }
}
