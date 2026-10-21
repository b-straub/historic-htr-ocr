// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Shared by the decode matvec kernels (dmv.wgsl, dmv_qkv.wgsl, dmv_ffn.wgsl; gpu.rs prepends it together with one
// reduction file — dmv_reduce_wg.wgsl or dmv_reduce_sg.wgsl — and one dmv_dot.wgsl per weight binding).
// Layout: 256 threads = 8 groups of 32 lanes; a group computes R rows (RA accumulators per lane: R, or 2R where a group
// computes the rows of two matrices); a lane takes the row's q8_0 blocks lane, lane + 32, … and loads the block's 32
// inputs once for all R rows. f32 arithmetic throughout.

const R: u32 = R_u;
const RA: u32 = RA_u;

// the group sums: tot[g · RA + a] = Σ over group g's lanes of accumulator a (row_sums)
var<workgroup> tot: array<f32, TOT_u>;

// the four signed bytes of a word
fn i8x4(w: u32) -> vec4<f32> {
    return vec4<f32>(vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w)) >> vec4<u32>(24u));
}

// 1 / sqrt(mean(X[0 … k)²) + eps) — the rms_norm.wgsl scale of the input row, computed by every workgroup (the row is
// ≤ 12 KB, read from cache): the norm needs no dispatch of its own
fn norm_scale(li: u32, k: u32, eps: f32) -> f32 {
    var s = 0.0;
    for (var i = li; i < k / 4u; i += 256u) {
        let v = X[i];
        s += dot(v, v);
    }
    return 1.0 / sqrt(wg_sum(li, s) / f32(k) + eps);
}
