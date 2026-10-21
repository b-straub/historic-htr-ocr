// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// bf16 weights, two per u32 (load_bf16.wgsl's layout), four consecutive values from element i (i % 4 == 0: two
// whole words) — exact
@group(0) @binding(2) var<storage, read> W: array<u32>;

fn load_w4(i: u32) -> vec4<f32> {
    let w0 = W[i >> 1u];
    let w1 = W[(i >> 1u) + 1u];
    return vec4<f32>(bitcast<f32>(w0 << 16u), bitcast<f32>(w0 & 0xffff0000u), bitcast<f32>(w1 << 16u), bitcast<f32>(w1 & 0xffff0000u));
}
