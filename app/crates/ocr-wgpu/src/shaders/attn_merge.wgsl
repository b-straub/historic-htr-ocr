// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Decode attention for the F16 / F16Coop paths, part 2/2: one workgroup per q head merges its chunks' partials
// (attn_decode.wgsl): M = max m_c, L = Σ l_c · e^(m_c − M), O[d] = Σ o_c[d] · e^(m_c − M) / L. Thread = dim.

struct Params {
    n_keys: u32,
    n_head: u32,
    n_kv: u32,
    n_chunk: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> Part: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.x;
    let n = (p.n_keys + CH_u - 1u) / CH_u;
    var mx = -3.0e38;
    for (var c = 0u; c < n; c++) {
        mx = max(mx, Part[(h * p.n_chunk + c) * 130u]);
    }
    var l = 0.0;
    var o = 0.0;
    for (var c = 0u; c < n; c++) {
        let base = (h * p.n_chunk + c) * 130u;
        let w = exp(Part[base] - mx);
        l = fma(Part[base + 1u], w, l);
        o = fma(Part[base + 2u + li], w, o);
    }
    O[h * 128u + li] = o / l;
}
