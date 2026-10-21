// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Decode attention 1/3 (one query token against the f16 KV cache, GQA like attention_causal.wgsl):
// S[h][j] = q_h · k_j · scale for j < n_keys — one thread per (head, key). Q [n_head · 128] f32, K cache
// [pos][n_kv · 128] f16 (two per u32), S [n_head][n_ctx].

struct Params {
    n_keys: u32,
    n_head: u32,
    n_kv: u32,
    n_ctx: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;

const D4: u32 = 32u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let total = p.n_head * p.n_keys;
    for (var i = gid.x; i < total; i += nwg.x * 256u) {
        let h = i / p.n_keys;
        let j = i % p.n_keys;
        let kb = j * p.n_kv * D4 + (h / (p.n_head / p.n_kv)) * D4;
        var d = vec4<f32>(0.0);
        for (var e = 0u; e < D4; e++) {
            let w = K[kb + e];
            d = fma(Q[h * D4 + e], vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)), d);
        }
        S[h * p.n_ctx + j] = (d.x + d.y + d.z + d.w) * p.scale;
    }
}
