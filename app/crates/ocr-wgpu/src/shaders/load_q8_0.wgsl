// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// q8_0 weights as the GGUF bytes, read as u32 words: blocks of 34 bytes = f16 scale d + 32 × i8 q, value = q · d
// (ggml dequantize_row_q8_0: exact in f32). A block starts at byte 34·b — always 2-aligned, so the scale is one
// half of one word.
@group(0) @binding(2) var<storage, read> W: array<u32>;

fn load_w(i: u32) -> f32 {
    let base = (i >> 5u) * 34u;
    let sw = W[base >> 2u];
    let d = unpack2x16float(select(sw, sw >> 16u, (base & 2u) != 0u)).x;
    let o = base + 2u + (i & 31u);
    let q = bitcast<i32>(((W[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu) << 24u) >> 24u;
    return f32(q) * d;
}
