// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// 2D RoPE in place on Q or K [n_tok, n_head · d_head] — clip.cpp build_rope_2d(…, interleave_freq = true): two
// ggml_rope_ext NORMAL-mode passes (adjacent pairs x[2k], x[2k+1] rotated), the first on dims [0, d/2) with the patch
// row (pos_h), the second on dims [d/2, d) with the patch column (pos_w) and freq_scale = base^(-2/d), i.e. the odd
// frequencies. The cos/sin table (cs, computed on the CPU the way ggml_rope_cache_init does) is
// [half][pos][k] → (cos, sin), k < d/4.

struct Params {
    n_tok: u32,
    n_head: u32,
    d_head: u32,
    px: u32,     // patches per row: token t sits at row t / px, column t % px
    max_pos: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> cs: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> Q: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let nk = p.d_head / 4u; // pairs per half
    let total = p.n_tok * p.n_head * 2u * nk;
    for (var i = gid.x; i < total; i += nwg.x * 256u) {
        let k = i % nk;
        let half = (i / nk) % 2u;
        let h = (i / (2u * nk)) % p.n_head;
        let t = i / (2u * nk * p.n_head);
        var pos = t / p.px;
        if (half == 1u) {
            pos = t % p.px;
        }
        let c = cs[(half * p.max_pos + pos) * nk + k];
        let o = t * p.n_head * p.d_head + h * p.d_head + half * (p.d_head / 2u) + 2u * k;
        let x0 = Q[o];
        let x1 = Q[o + 1u];
        Q[o] = x0 * c.x - x1 * c.y;
        Q[o + 1u] = x0 * c.y + x1 * c.x;
    }
}
