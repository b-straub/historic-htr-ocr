// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Full (non-causal) softmax attention over all patches: O = softmax(Q·Kᵀ · scale) · V, clip.cpp build_attn (no mask).
// Q, K, V, O are [n_tok, n_head · 64] (O head-major per token = ggml's kqv permute + cont). Online softmax
// (flash-attention style): no n×n score matrix. 64 threads = 64 queries of one head; K / V go through workgroup memory
// 16 keys at a time (2 × 4 KB), the tile's scores in workgroup memory (4 KB). f32 throughout.
// The query and accumulator are 16 vec4 registers each, written out by gpu.rs at the @UNROLL lines (one line per
// d4 = 0…15, `#` = d4): an indexed private array would live in slow thread memory (naga's loop guards stop unrolling).
// The F32 path (bit-exact reference); the f16 paths run attention_split.wgsl (browser) / attention_coop.wgsl (native).

struct Params {
    n_tok: u32,
    n_head: u32,
    scale: f32,
    round16: u32, // 1: q, k, v rounded to f16 like llama.cpp Metal's flash attention (precision bisect only)
}

fn r16(v: vec4<f32>) -> vec4<f32> {
    if (p.round16 == 1u) {
        return vec4<f32>(unpack2x16float(pack2x16float(v.xy)), unpack2x16float(pack2x16float(v.zw)));
    }
    return v;
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> O: array<vec4<f32>>;

const D4: u32 = 16u; // d_head / 4
const TK: u32 = 16u;

var<workgroup> Ks: array<vec4<f32>, 256>; // [TK][D4]
var<workgroup> Vs: array<vec4<f32>, 256>;
var<workgroup> S: array<f32, 1024>; // [TK][64 queries]

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.y;
    let qi = min(wg.x * 64u + li, p.n_tok - 1u); // a thread past the end computes the last query again, stores nothing
    let row = p.n_head * D4; // vec4s per token
    let qb = qi * row + h * D4;
    //@UNROLL let q# = r16(Q[qb + #u]) * p.scale;
    //@UNROLL var acc# = vec4<f32>(0.0);
    var m = -3.0e38;
    var l = 0.0;
    for (var k0 = 0u; k0 < p.n_tok; k0 += TK) {
        workgroupBarrier();
        for (var e = li; e < TK * D4; e += 64u) {
            let kj = min(k0 + e / D4, p.n_tok - 1u);
            let src = kj * row + h * D4 + e % D4;
            Ks[e] = r16(K[src]);
            Vs[e] = r16(V[src]);
        }
        workgroupBarrier();
        let nk = min(TK, p.n_tok - k0);
        var tmax = m;
        for (var j = 0u; j < nk; j++) {
            let kb = j * D4;
            var d = vec4<f32>(0.0);
            //@UNROLL d = fma(q#, Ks[kb + #u], d);
            let s = d.x + d.y + d.z + d.w;
            S[j * 64u + li] = s;
            tmax = max(tmax, s);
        }
        let corr = exp(m - tmax);
        l *= corr;
        //@UNROLL acc# *= corr;
        for (var j = 0u; j < nk; j++) {
            let pj = exp(S[j * 64u + li] - tmax);
            l += pj;
            let vb = j * D4;
            //@UNROLL acc# = fma(vec4<f32>(pj), Vs[vb + #u], acc#);
        }
        m = tmax;
    }
    if (wg.x * 64u + li < p.n_tok) {
        let inv = 1.0 / l;
        //@UNROLL O[qb + #u] = acc# * inv;
    }
}
