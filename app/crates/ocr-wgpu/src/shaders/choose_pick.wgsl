// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// GPU token choice (greedy presets), part 3/3: the argmax over choose_max's partials → the chosen token, written where
// the next decode step's embedding gather reads it (Tok[0]) and appended to the history (Hist[at]: the repetition
// window of the next choices, and what the CPU reads back for the text). With use_fp the token's output count goes up
// and F = fp · count is stored for choose_max (ocr-decode counts every chosen token; after an end-of-generation token
// the read stops, so counting it too changes nothing).

struct Params {
    n_part: u32,
    at: u32,
    use_fp: u32,
    fp: f32,
    // −inf's bits (0xff800000), from the host (choose_max.wgsl)
    neg_inf: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> Part: array<u32>;
@group(0) @binding(2) var<storage, read_write> Tok: array<u32>;
@group(0) @binding(3) var<storage, read_write> Hist: array<u32>;
@group(0) @binding(4) var<storage, read_write> C: array<u32>;
@group(0) @binding(5) var<storage, read_write> F: array<f32>;

var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;

fn better(v: f32, i: u32, w: f32, j: u32) -> bool {
    return v > w || (v == w && i < j);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_index) li: u32) {
    var v = bitcast<f32>(p.neg_inf);
    var i = 0xffffffffu;
    for (var k = li; k < p.n_part; k += 256u) {
        let x = bitcast<f32>(Part[2u * k]);
        let j = Part[2u * k + 1u];
        if (better(x, j, v, i)) {
            v = x;
            i = j;
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
        // every logit NaN: token 0 (the CPU's argmax starts at index 0)
        let t = select(bi[0], 0u, bi[0] == 0xffffffffu);
        Tok[0] = t;
        Hist[p.at] = t;
        if (p.use_fp != 0u) {
            let c = C[t] + 1u;
            C[t] = c;
            F[t] = p.fp * f32(c);
        }
    }
}
