// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// the decode kernels' reductions with subgroup sums (subgroups of 32 — checked in gpu.rs; the device feature
// `subgroups`, natively only: naga rejects the `enable subgroups;` directive for now, wgpu's WebGPU backend does not
// request the browser's feature yet)

var<workgroup> red: array<f32, 8>;

// Σ of v over the 256 threads (every thread gets it)
fn wg_sum(li: u32, v: f32) -> f32 {
    let s = subgroupAdd(v);
    if ((li & 31u) == 0u) {
        red[li >> 5u] = s;
    }
    workgroupBarrier();
    var t = 0.0;
    for (var k = 0u; k < 8u; k++) {
        t += red[k];
    }
    workgroupBarrier();
    return t;
}

// each group's 32-lane sums of every accumulator → tot[g · RA + a]
fn row_sums(li: u32, acc: ptr<function, array<f32, RA>>) {
    for (var a = 0u; a < RA; a++) {
        let s = subgroupAdd((*acc)[a]);
        if ((li & 31u) == 0u) {
            tot[(li >> 5u) * RA + a] = s;
        }
    }
    workgroupBarrier();
}
