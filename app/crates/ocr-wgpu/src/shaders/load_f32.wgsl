// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// f32 weights (patch embedding, patch merger)
@group(0) @binding(2) var<storage, read> W: array<f32>;

fn load_w(i: u32) -> f32 {
    return W[i];
}
