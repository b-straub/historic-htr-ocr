// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Decode attention for the F16 / F16Coop paths, split over the keys (flash decoding), part 1/2: one workgroup per
// (chunk of CH keys, kv head) → the chunk's softmax partials of the kv head's TWO q heads (GQA group of 2, checked in
// text.rs): max m, Σ exp(s − m), Σ exp(s − m) · v. The cache is read once per kv head (attn_scores / attn_values of the
// F32 path read it once per q head — at a page's ~2000 keys the KV reads, 8 MB per layer, are what this costs).
// f32 arithmetic, the f16 KV cache (two per u32). Part [n_head][n_chunk][2 + 128]: m, l, o[128].

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
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> V: array<u32>;
@group(0) @binding(4) var<storage, read_write> Part: array<f32>;

const D4: u32 = 32u;
const CH: u32 = CH_u; // keys per chunk (≤ 128: the workgroup's threads)

var<workgroup> qs: array<vec4<f32>, 64>; // the two q heads
var<workgroup> S: array<vec2<f32>, 128>; // exp(s − m) of both heads per key
var<workgroup> red: array<vec2<f32>, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let c = wg.x;
    let kvh = wg.y;
    let h0 = kvh * 2u;
    if (li < 2u * D4) {
        qs[li] = Q[h0 * D4 + li] * p.scale;
    }
    workgroupBarrier();
    let j = c * CH + li;
    let valid = li < CH && j < p.n_keys;
    var s = vec2<f32>(-3.0e38);
    if (valid) {
        let kb = j * p.n_kv * D4 + kvh * D4;
        var d0 = vec4<f32>(0.0);
        var d1 = vec4<f32>(0.0);
        for (var e = 0u; e < D4; e++) {
            let w = K[kb + e];
            let k = vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));
            d0 = fma(qs[e], k, d0);
            d1 = fma(qs[D4 + e], k, d1);
        }
        s = vec2<f32>(d0.x + d0.y + d0.z + d0.w, d1.x + d1.y + d1.z + d1.w);
    }
    red[li] = s;
    workgroupBarrier();
    for (var n = 64u; n > 0u; n >>= 1u) {
        if (li < n) {
            red[li] = max(red[li], red[li + n]);
        }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var e = vec2<f32>(0.0);
    if (valid) {
        e = exp(s - m);
    }
    S[li] = e;
    red[li] = e;
    workgroupBarrier();
    for (var n = 64u; n > 0u; n >>= 1u) {
        if (li < n) {
            red[li] += red[li + n];
        }
        workgroupBarrier();
    }
    // values: thread = one of the 128 dims, over the chunk's keys (V element (pos, kvh, dim) = half of a u32)
    let nk = min(CH, p.n_keys - c * CH);
    let vb = kvh * 128u + li;
    var o = vec2<f32>(0.0);
    for (var t = 0u; t < nk; t++) {
        let i = (c * CH + t) * p.n_kv * 128u + vb;
        let w = V[i >> 1u];
        o = fma(S[t], vec2<f32>(unpack2x16float(select(w, w >> 16u, (i & 1u) == 1u)).x), o);
    }
    let b0 = (h0 * p.n_chunk + c) * 130u;
    let b1 = ((h0 + 1u) * p.n_chunk + c) * 130u;
    if (li == 0u) {
        Part[b0] = m.x;
        Part[b0 + 1u] = red[0].x;
        Part[b1] = m.y;
        Part[b1 + 1u] = red[0].y;
    }
    Part[b0 + 2u + li] = o.x;
    Part[b1 + 2u + li] = o.y;
}
