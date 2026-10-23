// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Causal GQA attention of a prompt batch against the f16 KV cache (Qwen3 build_attn, head dim 128):
// O = softmax(Q·Kᵀ · scale, causal) · V. Query t of the batch sits at position pos0 + t and sees the keys 0 … pos0 + t
// (everything earlier in the cache + the batch up to itself). Query head h reads KV head h / (n_head / n_kv) (ggml's
// broadcast in ggml_mul_mat). Q / O [n_q, n_head · 128] f32; K / V cache [pos][n_kv · 128] f16, two per u32.
// Online softmax (flash-attention style, like attention.wgsl): 128 threads = 64 queries × 2 halves of the head dim —
// each thread holds 16 vec4 of q and of the accumulator (the vision kernel's register budget), the two halves' partial
// dot products meet in workgroup memory. K / V tiles of 8 keys: 4 + 4 KB, partial scores 4 KB — 12 KB of workgroup
// memory, inside WebGPU's 16 KB default.
// Tiles past the workgroup's last query are skipped (kend). A matrix-unit version was slower on the M3 (text.rs).

struct Params {
    n_q: u32,
    n_head: u32,
    n_kv: u32,
    pos0: u32,
    scale: f32,
    qb0: u32, // the dispatch's first query block (the prefill's attention is dispatched in query chunks: Batch::flush)
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> V: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> O: array<vec4<f32>>;

const D4: u32 = 32u; // 128 / 4
const TK: u32 = 8u;

var<workgroup> Ks: array<vec4<f32>, 256>; // [TK][D4]
var<workgroup> Vs: array<vec4<f32>, 256>;
var<workgroup> P: array<f32, 1024>; // [2 halves][TK][64 queries]

fn h4(w: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));
}

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.y;
    let ql = li & 63u;
    let half = li >> 6u;
    let qi = min((wg.x + p.qb0) * 64u + ql, p.n_q - 1u); // a thread past the end computes the last query again, stores nothing
    let qpos = p.pos0 + qi;
    let kvh = h / (p.n_head / p.n_kv);
    let qb = qi * p.n_head * D4 + h * D4 + half * 16u;
    let kvrow = p.n_kv * D4; // vec4s per cached position
    //@UNROLL let q# = Q[qb + #u] * p.scale;
    //@UNROLL var acc# = vec4<f32>(0.0);
    var m = -3.0e38;
    var l = 0.0;
    let kend = p.pos0 + min((wg.x + p.qb0) * 64u + 63u, p.n_q - 1u) + 1u; // keys the workgroup's last query sees
    for (var k0 = 0u; k0 < kend; k0 += TK) {
        workgroupBarrier();
        for (var e = li; e < TK * D4; e += 128u) {
            let kj = min(k0 + e / D4, kend - 1u);
            let src = kj * kvrow + kvh * D4 + e % D4;
            Ks[e] = h4(K[src]);
            Vs[e] = h4(V[src]);
        }
        workgroupBarrier();
        for (var j = 0u; j < TK; j++) {
            let kb = j * D4 + half * 16u;
            var d = vec4<f32>(0.0);
            //@UNROLL d = fma(q#, Ks[kb + #u], d);
            P[(half * TK + j) * 64u + ql] = d.x + d.y + d.z + d.w;
        }
        workgroupBarrier();
        let nk = min(TK, qpos + 1u - min(k0, qpos + 1u)); // keys of this tile the query sees
        var tmax = m;
        for (var j = 0u; j < nk; j++) {
            tmax = max(tmax, P[j * 64u + ql] + P[(TK + j) * 64u + ql]);
        }
        let corr = exp(m - tmax);
        l *= corr;
        //@UNROLL acc# *= corr;
        for (var j = 0u; j < nk; j++) {
            let pj = exp(P[j * 64u + ql] + P[(TK + j) * 64u + ql] - tmax);
            l += pj;
            let vb = j * D4 + half * 16u;
            //@UNROLL acc# = fma(vec4<f32>(pj), Vs[vb + #u], acc#);
        }
        m = tmax;
    }
    if ((wg.x + p.qb0) * 64u + ql < p.n_q) {
        let inv = 1.0 / l;
        //@UNROLL O[qb + #u] = acc# * inv;
    }
}
