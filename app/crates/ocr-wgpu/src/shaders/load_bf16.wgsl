// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// bf16 weights, two per u32 (the GGUF bytes as they are): element i is the low half of word i/2 when i is even.
// bf16 → f32 is exact (the bf16 bits are the top 16 bits of the f32).
@group(0) @binding(2) var<storage, read> W: array<u32>;

fn load_w(i: u32) -> f32 {
    let w = W[i >> 1u];
    return bitcast<f32>(select(w << 16u, w & 0xffff0000u, (i & 1u) == 1u));
}
