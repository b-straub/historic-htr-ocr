// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The vision encoder's full attention for the F16 path (no matrix units: the browser) — attention.wgsl's
// O = softmax(Q·Kᵀ · scale) · V (head dim 64) as two register-blocked tile products per key tile, flash-attention
// style (online softmax): 64 queries of one head per workgroup (128 threads), keys in tiles of 16.
//   S = Q·Kᵀ    each thread 4 queries × 2 keys (8 dot products over the 64 dims: per vec4 of dims 6 workgroup reads
//               feed 8 vec4 fma); S kept transposed [key][query] in workgroup memory
//   softmax     threads 0…63 own one query row each: running max / sum (f32), the tile's P = exp(S − max) (f32) in
//               place, the row's rescale factor into Cr
//   O += P·V    each thread 4 queries × 8 dims (8 vec4 accumulators, f32): per key one vec4 of P (4 queries, the
//               transposed S) and two vec4 of V feed 8 vec4 fma
// Q (scaled), K, V as f16 in workgroup memory (K / V from heads_f16.wgsl, head-major, padded to whole tiles); the
// scores, softmax and accumulators f32; the output f16 (the o projection's input). Workgroup memory 16.75 KB: Q 8 KB,
// K 2 KB, V 2 KB, S 4 KB, Red 512 B, Cr 256 B — above WebGPU's guaranteed 16 KB: an adapter with less runs the F32
// path instead (gpu.rs Compute::choose_caps, the bytes counted from the declarations below by workgroup_bytes).
// Measured (M3 native F16 path, a 980 × 1540 page = 7700 patches, ms per layer): attention_split.wgsl (one query per 2 threads,
// K / V read per query) 238; this kernel with one thread per softmax row 213, two threads per row 198; key tiles of
// 32 with 4×4 score blocks (24 KB) 237 — the register pressure costs more than the halved barriers save.
enable f16;

struct Params {
    n_tok: u32,
    n_head: u32,
    scale: f32,
    n_pad: u32, // keys per head in Kh / Vh (a multiple of TK)
    qb0: u32, // the dispatch's first query block (the attention is dispatched in query chunks: Batch::flush)
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> Kh: array<vec4<f16>>; // [n_head][n_pad][D4]
@group(0) @binding(3) var<storage, read> Vh: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read_write> O: array<vec4<f16>>;

const D4: u32 = 16u; // d_head / 4
const TK: u32 = 16u;
const BQ: u32 = 64u;

var<workgroup> Qs: array<vec4<f16>, 1024>; // [BQ][D4]
var<workgroup> Ks: array<vec4<f16>, 256>; // [TK][D4]
var<workgroup> Vs: array<vec4<f16>, 256>;
var<workgroup> Ss: array<f32, 1024>; // [TK][BQ]: S (then P) transposed
var<workgroup> Red: array<f32, 128>; // the two halves of a row: partial max, then partial sum
var<workgroup> Cr: array<f32, 64>; // per query: this tile's rescale factor; at the end 1 / l

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.y;
    let qw = (wg.x + p.qb0) * BQ; // the workgroup's first query
    let row = p.n_head * D4; // vec4s per token
    for (var e = li; e < BQ * D4; e += 128u) {
        let qi = min(qw + e / D4, p.n_tok - 1u);
        Qs[e] = vec4<f16>(Q[qi * row + h * D4 + e % D4] * p.scale);
    }
    // S phase: queries 4·sq … +3, keys 2·sk, 2·sk + 1
    let sq = li >> 3u;
    let sk = li & 7u;
    // P·V phase: queries 4·oq … +3, dims 8·od … +7 (vec4 2·od, 2·od + 1)
    let oq = li >> 3u;
    let od = li & 7u;
    var a00 = vec4<f32>(0.0); var a01 = vec4<f32>(0.0);
    var a10 = vec4<f32>(0.0); var a11 = vec4<f32>(0.0);
    var a20 = vec4<f32>(0.0); var a21 = vec4<f32>(0.0);
    var a30 = vec4<f32>(0.0); var a31 = vec4<f32>(0.0);
    // softmax state of query row li (threads 0…63)
    var m = -3.0e38;
    var l = 0.0;
    let hb = h * p.n_pad * D4;
    for (var k0 = 0u; k0 < p.n_tok; k0 += TK) {
        workgroupBarrier();
        for (var e = li; e < TK * D4; e += 128u) {
            Ks[e] = Kh[hb + k0 * D4 + e];
            Vs[e] = Vh[hb + k0 * D4 + e];
        }
        workgroupBarrier();
        // S = Q·Kᵀ
        var s00 = vec4<f32>(0.0); var s01 = vec4<f32>(0.0);
        var s10 = vec4<f32>(0.0); var s11 = vec4<f32>(0.0);
        var s20 = vec4<f32>(0.0); var s21 = vec4<f32>(0.0);
        var s30 = vec4<f32>(0.0); var s31 = vec4<f32>(0.0);
        for (var d = 0u; d < D4; d++) {
            let q0 = vec4<f32>(Qs[(sq * 4u) * D4 + d]);
            let q1 = vec4<f32>(Qs[(sq * 4u + 1u) * D4 + d]);
            let q2 = vec4<f32>(Qs[(sq * 4u + 2u) * D4 + d]);
            let q3 = vec4<f32>(Qs[(sq * 4u + 3u) * D4 + d]);
            let k0v = vec4<f32>(Ks[(sk * 2u) * D4 + d]);
            let k1v = vec4<f32>(Ks[(sk * 2u + 1u) * D4 + d]);
            s00 = fma(q0, k0v, s00); s01 = fma(q0, k1v, s01);
            s10 = fma(q1, k0v, s10); s11 = fma(q1, k1v, s11);
            s20 = fma(q2, k0v, s20); s21 = fma(q2, k1v, s21);
            s30 = fma(q3, k0v, s30); s31 = fma(q3, k1v, s31);
        }
        let sa = (sk * 2u) * BQ + sq * 4u;
        Ss[sa] = hsum(s00); Ss[sa + 1u] = hsum(s10); Ss[sa + 2u] = hsum(s20); Ss[sa + 3u] = hsum(s30);
        Ss[sa + BQ] = hsum(s01); Ss[sa + BQ + 1u] = hsum(s11); Ss[sa + BQ + 2u] = hsum(s21); Ss[sa + BQ + 3u] = hsum(s31);
        workgroupBarrier();
        // online softmax of row sr over the tile's valid keys, two threads per row (8 keys each); P in place
        let sr = li & 63u;
        let half = li >> 6u;
        let nk = min(TK, p.n_tok - k0);
        var pmax = -3.0e38;
        for (var j = half * 8u; j < half * 8u + 8u; j++) {
            if (j < nk) {
                pmax = max(pmax, Ss[j * BQ + sr]);
            }
        }
        Red[li] = pmax;
        workgroupBarrier();
        let tmax = max(m, max(Red[sr], Red[sr + 64u]));
        let corr = exp(m - tmax);
        var ps = 0.0;
        for (var j = half * 8u; j < half * 8u + 8u; j++) {
            var pj = 0.0;
            if (j < nk) {
                pj = exp(Ss[j * BQ + sr] - tmax);
            }
            ps += pj;
            Ss[j * BQ + sr] = pj;
        }
        workgroupBarrier();
        Red[li] = ps;
        if (half == 0u) {
            Cr[sr] = corr;
        }
        m = tmax;
        workgroupBarrier();
        if (half == 0u) {
            l = l * corr + Red[sr] + Red[sr + 64u];
        }
        // O = O · corr + P·V
        let c = vec4<f32>(Cr[oq * 4u], Cr[oq * 4u + 1u], Cr[oq * 4u + 2u], Cr[oq * 4u + 3u]);
        a00 *= c.x; a01 *= c.x;
        a10 *= c.y; a11 *= c.y;
        a20 *= c.z; a21 *= c.z;
        a30 *= c.w; a31 *= c.w;
        for (var j = 0u; j < TK; j++) {
            let pb = j * BQ + oq * 4u;
            let pv = vec4<f32>(Ss[pb], Ss[pb + 1u], Ss[pb + 2u], Ss[pb + 3u]);
            let v0 = vec4<f32>(Vs[j * D4 + od * 2u]);
            let v1 = vec4<f32>(Vs[j * D4 + od * 2u + 1u]);
            a00 = fma(vec4<f32>(pv.x), v0, a00); a01 = fma(vec4<f32>(pv.x), v1, a01);
            a10 = fma(vec4<f32>(pv.y), v0, a10); a11 = fma(vec4<f32>(pv.y), v1, a11);
            a20 = fma(vec4<f32>(pv.z), v0, a20); a21 = fma(vec4<f32>(pv.z), v1, a21);
            a30 = fma(vec4<f32>(pv.w), v0, a30); a31 = fma(vec4<f32>(pv.w), v1, a31);
        }
    }
    workgroupBarrier();
    if (li < BQ) {
        Cr[li] = 1.0 / l;
    }
    workgroupBarrier();
    put(qw + oq * 4u, od, a00 * Cr[oq * 4u], a01 * Cr[oq * 4u], h, row);
    put(qw + oq * 4u + 1u, od, a10 * Cr[oq * 4u + 1u], a11 * Cr[oq * 4u + 1u], h, row);
    put(qw + oq * 4u + 2u, od, a20 * Cr[oq * 4u + 2u], a21 * Cr[oq * 4u + 2u], h, row);
    put(qw + oq * 4u + 3u, od, a30 * Cr[oq * 4u + 3u], a31 * Cr[oq * 4u + 3u], h, row);
}

fn hsum(v: vec4<f32>) -> f32 {
    return v.x + v.y + v.z + v.w;
}

// query qi's dims 8·od … +7 out as f16
fn put(qi: u32, od: u32, v0: vec4<f32>, v1: vec4<f32>, h: u32, row: u32) {
    if (qi < p.n_tok) {
        let dst = qi * row + h * D4 + od * 2u;
        O[dst] = vec4<f16>(v0);
        O[dst + 1u] = vec4<f16>(v1);
    }
}
