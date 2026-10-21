// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Decode attention 2/3: S[h][0 … n_keys) = softmax over the row, in place — one 256-thread workgroup per head.

struct Params {
    n_keys: u32,
    n_head: u32,
    n_kv: u32,
    n_ctx: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let base = wg.x * p.n_ctx;
    var m = -3.0e38;
    for (var j = li; j < p.n_keys; j += 256u) {
        m = max(m, S[base + j]);
    }
    part[li] = m;
    workgroupBarrier();
    for (var n = 128u; n > 0u; n >>= 1u) {
        if (li < n) {
            part[li] = max(part[li], part[li + n]);
        }
        workgroupBarrier();
    }
    m = part[0];
    workgroupBarrier();
    var s = 0.0;
    for (var j = li; j < p.n_keys; j += 256u) {
        let e = exp(S[base + j] - m);
        S[base + j] = e;
        s += e;
    }
    part[li] = s;
    workgroupBarrier();
    for (var n = 128u; n > 0u; n >>= 1u) {
        if (li < n) {
            part[li] += part[li + n];
        }
        workgroupBarrier();
    }
    let inv = 1.0 / part[0];
    for (var j = li; j < p.n_keys; j += 256u) {
        S[base + j] *= inv;
    }
}
