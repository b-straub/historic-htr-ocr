// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// K or V rows [n_tok, width] f32 → the f16 KV cache (two halves per u32, like llama.cpp's default f16 cache) at
// rows pos0 … pos0 + n_tok − 1. width is even.

struct Params {
    n_tok: u32,
    width: u32,
    pos0: u32,
    _p0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let total = p.n_tok * p.width / 2u;
    for (var i = gid.x; i < total; i += nwg.x * 256u) {
        C[p.pos0 * p.width / 2u + i] = pack2x16float(X[i]);
    }
}
