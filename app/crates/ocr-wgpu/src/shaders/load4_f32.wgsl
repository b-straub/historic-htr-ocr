// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// f32 weights, four consecutive values from element i (i % 4 == 0)
@group(0) @binding(2) var<storage, read> W: array<vec4<f32>>;

fn load_w4(i: u32) -> vec4<f32> {
    return W[i >> 2u];
}
