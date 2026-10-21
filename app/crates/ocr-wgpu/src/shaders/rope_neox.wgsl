// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// RoPE NEOX in place on Q or K [n_tok, n_head · d] — ggml_rope_ext mode NEOX (Qwen3, n_rot = d): the pairs
// (x[i], x[i + d/2]) rotated by θ = pos · base^(−2i/d). The cos / sin table cs [pos][d/2] is computed on the CPU the
// way llama.cpp Metal's kernel_rope_neox does (θ = (float) pos · pow(base, −2i/d) in f32). Token t sits at pos0 + t.

struct Params {
    n_tok: u32,
    n_head: u32,
    d_head: u32,
    pos0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> cs: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> Q: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let half = p.d_head / 2u;
    let total = p.n_tok * p.n_head * half;
    for (var i = gid.x; i < total; i += nwg.x * 256u) {
        let k = i % half;
        let th = i / half; // token · n_head + head
        let t = th / p.n_head;
        let c = cs[(p.pos0 + t) * half + k];
        let o = th * p.d_head + k;
        let x0 = Q[o];
        let x1 = Q[o + half];
        Q[o] = x0 * c.x - x1 * c.y;
        Q[o + half] = x0 * c.y + x1 * c.x;
    }
}
