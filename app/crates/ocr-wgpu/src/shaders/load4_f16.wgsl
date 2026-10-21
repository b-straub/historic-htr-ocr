// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// f16 weights (load_f16.wgsl's layout), four consecutive values from element i (i % 4 == 0: two whole words)
@group(0) @binding(2) var<storage, read> W: array<vec2<u32>>;

fn load_w4(i: u32) -> vec4<f32> {
    let w = W[i >> 2u];
    return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));
}
