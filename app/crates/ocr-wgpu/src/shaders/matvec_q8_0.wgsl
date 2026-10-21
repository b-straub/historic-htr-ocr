// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// y[N] (+)= W[N,K] · x[K] with W q8_0 (load_q8_0.wgsl's layout) — the decode step's projections (one token).
// 256 threads = 8 rows × 32 lanes; a lane takes the row's q8_0 blocks lane, lane + 32, … (K / 32 blocks), the 32
// lanes are summed in workgroup memory. acc = 1 adds onto y (the residual stream). f32 accumulation.
// The F32 reference path's decode projections (its reads are the regression baseline); the f16 paths run the fused
// dmv*.wgsl kernels (several rows per lane group, the norms / SwiGLU folded in, subgroup sums where available).

struct Params {
    N: u32,
    K: u32,
    acc: u32,
    _p0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

var<workgroup> part: array<f32, 256>;

// the four signed bytes of a word
fn i8x4(w: u32) -> vec4<f32> {
    return vec4<f32>(vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w)) >> vec4<u32>(24u));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lane = li & 31u;
    let row = wg.x * 8u + (li >> 5u);
    let nb = p.K >> 5u;
    var s = 0.0;
    if (row < p.N) {
        for (var b = lane; b < nb; b += 32u) {
            let base = (row * nb + b) * 34u;
            let sw = W[base >> 2u];
            let aligned = (base & 2u) != 0u; // quants start on a word boundary
            let d = unpack2x16float(select(sw, sw >> 16u, aligned)).x;
            let w0 = (base + 2u) >> 2u;
            let xb = b * 8u;
            var dot = 0.0;
            for (var j = 0u; j < 8u; j++) {
                var q: u32;
                if (aligned) {
                    q = W[w0 + j];
                } else {
                    q = (W[w0 + j] >> 16u) | (W[w0 + j + 1u] << 16u);
                }
                let x = X[xb + j];
                let v = i8x4(q) * x;
                dot += v.x + v.y + v.z + v.w;
            }
            s = fma(dot, d, s);
        }
    }
    part[li] = s;
    workgroupBarrier();
    for (var n = 16u; n > 0u; n >>= 1u) {
        if (lane < n) {
            part[li] += part[li + n];
        }
        workgroupBarrier();
    }
    if (lane == 0u && row < p.N) {
        if (p.acc == 1u) {
            Y[row] = Y[row] + part[li];
        } else {
            Y[row] = part[li];
        }
    }
}
