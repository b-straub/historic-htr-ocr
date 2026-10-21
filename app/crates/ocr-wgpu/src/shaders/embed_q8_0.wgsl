// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// X[t] = token_embd[tok[t]] dequantized (ggml_get_rows on the q8_0 table) for every row whose tok[t] is not SKIP
// (0xffffffff = an image row, filled from the vision encoder's embeddings instead).

struct Params {
    n_tok: u32,
    dim: u32,
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tok: array<u32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;

fn q8(i: u32) -> f32 {
    let base = (i >> 5u) * 34u;
    let sw = W[base >> 2u];
    let d = unpack2x16float(select(sw, sw >> 16u, (base & 2u) != 0u)).x;
    let o = base + 2u + (i & 31u);
    let q = bitcast<i32>(((W[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu) << 24u) >> 24u;
    return f32(q) * d;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let total = p.n_tok * p.dim;
    for (var i = gid.x; i < total; i += nwg.x * 256u) {
        let t = i / p.dim;
        let id = tok[t];
        if (id != 0xffffffffu) {
            X[i] = q8(id * p.dim + i % p.dim);
        }
    }
}
