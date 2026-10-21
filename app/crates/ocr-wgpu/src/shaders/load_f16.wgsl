// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// f16 weights, two per u32 (bf16 converted once at load on the f16 paths, f16_from_bf16.wgsl): element i is the low
// half of word i/2 when i is even
@group(0) @binding(2) var<storage, read> W: array<u32>;

fn load_w(i: u32) -> f32 {
    return unpack2x16float(W[i >> 1u])[i & 1u];
}
