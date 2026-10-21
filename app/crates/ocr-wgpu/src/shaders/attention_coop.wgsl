// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The vision encoder's full attention (attention.wgsl's O = softmax(Q·Kᵀ · scale) · V, head dim 64) on the GPU's
// matrix units — the F16Coop path (Metal simdgroup matrices through wgpu_cooperative_matrix, subgroups of 32).
// Flash attention with online softmax, like llama.cpp's Metal kernel: Q, K, V as f16 (q pre-scaled), scores
// S = Q·Kᵀ accumulated in f32 8×8 matrices, the running max / sum per query row in f32, P = exp(S − max) as f16 into
// P·V, the output accumulated in f32.
// Workgroup = 4 simdgroups × 8 queries (32 queries of one head); keys in blocks of 32. K and V come as f16, head-major
// and zero-padded to whole blocks (heads_f16.wgsl): the matrix units load their 8×8 blocks straight from memory — no
// workgroup staging, no workgroup barrier in the key loop (the simdgroups run independently).
// Per simdgroup: Q = 8 A-matrices (the 64 dims) kept in registers; per key block S = 4 C-matrices (8 q × 32 keys)
// → the simdgroup's S scratch → row max / exp / sum by the lanes (4 lanes per row, subgroup shuffles) → P (f16) →
// O (8 q × 64 dims, f32 in workgroup memory, rescaled by the lanes) += P·V via 8 C-matrices.
// Measured (M3, a full test page, ms per layer): K / V staged through workgroup memory as f32 → f16 per block 170;
// read as f16 blocks from memory 119 (vision GPU 7.8 → 6.6 s). Earlier alternatives (staged): O kept in registers with a lazy rescale 176;
// 16 queries per simdgroup (64 per workgroup) 200. Round 3 (a 980 × 1540 page = 7700 patches, 138 ms per layer with this kernel):
// O in registers, rescaled through memory only when a row max rose (uniform per workgroup) 140; 16 queries per
// simdgroup with direct loads 226 (loads interleaved 263); key blocks of 64 333; K / V f16 staged in workgroup
// memory 191 — the register pressure of more matrices costs more than the K / V loads it saves (removing the loads
// alone: 80).
enable f16;
enable wgpu_cooperative_matrix;

struct Params {
    n_tok: u32,
    n_head: u32,
    scale: f32,
    n_pad: u32, // keys per head in Kh / Vh (n_tok rounded up to BKV)
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> Kh: array<f16>; // [n_head][n_pad][64]
@group(0) @binding(3) var<storage, read> Vh: array<f16>;
@group(0) @binding(4) var<storage, read_write> O: array<vec4<f16>>; // f16: the o projection's input (matmul_coopd)

const D: u32 = 64u;
const D4: u32 = 16u;
const BKV: u32 = 32u;

var<workgroup> Qs: array<f16, 2048>; // [4 sg][8 q][D]
var<workgroup> Ss: array<f32, 1024>; // [4 sg][8 q][BKV]
var<workgroup> Ps: array<f16, 1024>; // [4 sg][8 q][BKV]
var<workgroup> Os: array<f32, 2048>; // [4 sg][8 q][D]

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.y;
    let sg = li >> 5u;
    let lane = li & 31u;
    let row = p.n_head * D4; // vec4s per token
    let q0 = wg.x * 32u + sg * 8u; // the simdgroup's first query
    // the simdgroup's 8 queries → Qs (f16, scaled), zero O
    for (var e = lane; e < 8u * D4; e += 32u) {
        let qi = min(q0 + e / D4, p.n_tok - 1u);
        let v = vec4<f16>(Q[qi * row + h * D4 + e % D4] * p.scale);
        let o = sg * 512u + e * 4u;
        Qs[o] = v.x;
        Qs[o + 1u] = v.y;
        Qs[o + 2u] = v.z;
        Qs[o + 3u] = v.w;
        Os[o] = 0.0;
        Os[o + 1u] = 0.0;
        Os[o + 2u] = 0.0;
        Os[o + 3u] = 0.0;
    }
    workgroupBarrier();
    //@D8 let q_I = coopLoadT<coop_mat8x8<f16, A>>(&Qs[sg * 512u + I * 8u], D);
    // the lane's row of the softmax state: row r = lane / 4, its 8 keys of a block = (lane % 4) · 8 …
    let r = lane >> 2u;
    let kc = (lane & 3u) * 8u;
    let sb = sg * 256u + r * BKV + kc;
    let ob = sg * 512u + r * D + (lane & 3u) * 16u;
    var m = -3.0e38;
    var l = 0.0;
    let hb = h * p.n_pad * D; // the head's first K / V element
    for (var k0 = 0u; k0 < p.n_tok; k0 += BKV) {
        let kb = hb + k0 * D; // the key block's first element
        // S = Q·Kᵀ: 4 key blocks of 8 × 8 dim blocks
        //@KB var s_J = coop_mat8x8<f32, C>();
        //@KBD8 s_J = coopMultiplyAdd(q_I, coopLoad<coop_mat8x8<f16, B>>(&Kh[kb + J * 8u * D + I * 8u], D), s_J);
        //@KB coopStoreT(s_J, &Ss[sg * 256u + J * 8u], BKV);
        subgroupBarrier();
        // online softmax of the lane's 8 scores
        var mx = m;
        for (var j = 0u; j < 8u; j++) {
            if (k0 + kc + j < p.n_tok) {
                mx = max(mx, Ss[sb + j]);
            }
        }
        mx = max(mx, subgroupShuffleXor(mx, 1u));
        mx = max(mx, subgroupShuffleXor(mx, 2u));
        let corr = exp(m - mx);
        var ps = 0.0;
        for (var j = 0u; j < 8u; j++) {
            var pj = 0.0;
            if (k0 + kc + j < p.n_tok) {
                pj = exp(Ss[sb + j] - mx);
            }
            let ph = f16(pj);
            Ps[sb + j] = ph;
            ps += f32(ph);
        }
        ps += subgroupShuffleXor(ps, 1u);
        ps += subgroupShuffleXor(ps, 2u);
        l = l * corr + ps;
        m = mx;
        // the lane's 16 output values of row r (dims (lane % 4) · 16 …) rescaled
        for (var j = 0u; j < 16u; j++) {
            Os[ob + j] *= corr;
        }
        subgroupBarrier();
        // O += P·V: 8 dim blocks × 4 key blocks
        //@D8 var o_I = coopLoadT<coop_mat8x8<f32, C>>(&Os[sg * 512u + I * 8u], D);
        //@KB let p_J = coopLoadT<coop_mat8x8<f16, A>>(&Ps[sg * 256u + J * 8u], BKV);
        //@KBD8 o_I = coopMultiplyAdd(p_J, coopLoadT<coop_mat8x8<f16, B>>(&Vh[kb + J * 8u * D + I * 8u], D), o_I);
        //@D8 coopStoreT(o_I, &Os[sg * 512u + I * 8u], D);
        subgroupBarrier();
    }
    // out: row r of the simdgroup, the lane's 16 dims, / l
    let qi = q0 + r;
    if (qi < p.n_tok) {
        let inv = 1.0 / l;
        let dst = qi * row + h * D4 + (lane & 3u) * 4u;
        for (var j = 0u; j < 4u; j++) {
            O[dst + j] = vec4<f16>(vec4<f32>(Os[ob + j * 4u], Os[ob + j * 4u + 1u], Os[ob + j * 4u + 2u], Os[ob + j * 4u + 3u]) * inv);
        }
    }
}
