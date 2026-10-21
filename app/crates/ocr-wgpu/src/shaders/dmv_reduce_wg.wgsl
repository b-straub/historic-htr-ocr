// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// the decode kernels' reductions in workgroup memory (no subgroup operations: the browser path)
var<workgroup> red: array<f32, RED_u>; // 256 · RA

// Σ of v over the 256 threads (every thread gets it)
fn wg_sum(li: u32, v: f32) -> f32 {
    red[li] = v;
    workgroupBarrier();
    for (var n = 128u; n > 0u; n >>= 1u) {
        if (li < n) {
            red[li] += red[li + n];
        }
        workgroupBarrier();
    }
    let s = red[0];
    workgroupBarrier();
    return s;
}

// each group's 32-lane sums of every accumulator → tot[g · RA + a]
fn row_sums(li: u32, acc: ptr<function, array<f32, RA>>) {
    for (var a = 0u; a < RA; a++) {
        red[a * 256u + li] = (*acc)[a];
    }
    workgroupBarrier();
    if (li < 8u * RA) {
        let g = li / RA;
        let a = li % RA;
        var s = 0.0;
        for (var k = 0u; k < 32u; k++) {
            s += red[a * 256u + g * 32u + k];
        }
        tot[li] = s;
    }
    workgroupBarrier();
}
