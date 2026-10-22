// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// D[dst0 + i] = S[src0 + i] for i < n (32-bit words) — Batch::copy when source and destination are slices of the same
// pool block: WebGPU's buffer-to-buffer copy refuses one buffer as both (the KV cache's growth), two writable
// non-overlapping bindings of it are allowed.

struct Params {
    n: u32,
    src0: u32,
    dst0: u32,
    _p0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> S: array<u32>;
@group(0) @binding(2) var<storage, read_write> D: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var i = gid.x; i < p.n; i += nwg.x * 256u) {
        D[p.dst0 + i] = S[p.src0 + i];
    }
}
