// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// GPU token choice (greedy presets), part 2/3: the frequency penalty + the argmax, stage 1 — every workgroup reduces
// its grid-stride share of the logits to (best value, its index) in Part. With use_fp: logit − F[i] for a token the
// output already holds (C[i] > 0; F = fp · count, one multiply stored by choose_pick — so this is a plain subtraction,
// rounded like ocr-decode's `logits[t] -= fp * count`, and no compiler can fuse it into an fma). Ties → the lowest
// index (ocr-decode's argmax keeps the first maximum); NaN never wins.

struct Params {
    n: u32,
    use_fp: u32,
    // −inf's bits (0xff800000), from the host: −inf is no WGSL constant (a folded bitcast or the lowest finite literal
    // is rejected by the browser's compiler after wgpu re-emits the WGSL)
    neg_inf: u32,
    _p1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> L: array<f32>;
@group(0) @binding(2) var<storage, read> C: array<u32>;
@group(0) @binding(3) var<storage, read> F: array<f32>;
@group(0) @binding(4) var<storage, read_write> Part: array<u32>;

var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;

fn better(v: f32, i: u32, w: f32, j: u32) -> bool {
    return v > w || (v == w && i < j);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    // −inf with "no index": the first value ≥ −inf replaces it (all −inf → the lowest index, like the CPU)
    var v = bitcast<f32>(p.neg_inf);
    var i = 0xffffffffu;
    for (var k = gid.x; k < p.n; k += nwg.x * 256u) {
        var x = L[k];
        if (p.use_fp != 0u && C[k] != 0u) {
            x = x - F[k];
        }
        if (better(x, k, v, i)) {
            v = x;
            i = k;
        }
    }
    bv[li] = v;
    bi[li] = i;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s && better(bv[li + s], bi[li + s], bv[li], bi[li])) {
            bv[li] = bv[li + s];
            bi[li] = bi[li + s];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        Part[2u * wg.x] = bitcast<u32>(bv[0]);
        Part[2u * wg.x + 1u] = bi[0];
    }
}
