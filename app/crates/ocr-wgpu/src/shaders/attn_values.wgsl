// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Decode attention 3/3: O[h] = Σ_j S[h][j] · v_j — one workgroup per head, 256 threads = 32 vec4 of the head dim ×
// 8 key lanes (key j on lane j % 8), the 8 partial sums added in workgroup memory. V cache f16, two per u32.

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
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> O: array<vec4<f32>>;

const D4: u32 = 32u;

var<workgroup> part: array<vec4<f32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.x;
    let e = li % D4;
    let lane = li / D4;
    let vb = (h / (p.n_head / p.n_kv)) * D4 + e;
    var acc = vec4<f32>(0.0);
    for (var j = lane; j < p.n_keys; j += 8u) {
        let w = V[j * p.n_kv * D4 + vb];
        acc = fma(vec4<f32>(S[h * p.n_ctx + j]), vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)), acc);
    }
    part[li] = acc;
    workgroupBarrier();
    if (lane == 0u) {
        var s = part[e];
        for (var k = 1u; k < 8u; k++) {
            s += part[k * D4 + e];
        }
        O[h * D4 + e] = s;
    }
}
