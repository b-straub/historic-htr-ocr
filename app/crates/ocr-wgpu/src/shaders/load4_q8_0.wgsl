// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// q8_0 weights (load_q8_0.wgsl's layout), four consecutive values from element i (i % 4 == 0: inside one block, the
// four quants are the bytes 2 + i % 32 … of the block — one word when the block starts at byte ≡ 2 mod 4, else the
// halves of two words). value = q · d, exact in f32 like load_q8_0.wgsl.
@group(0) @binding(2) var<storage, read> W: array<u32>;

fn load_w4(i: u32) -> vec4<f32> {
    let base = (i >> 5u) * 34u;
    let sw = W[base >> 2u];
    let aligned = (base & 2u) != 0u; // quants start on a word boundary
    let d = unpack2x16float(select(sw, sw >> 16u, aligned)).x;
    let o = (base + 2u + (i & 31u)) >> 2u;
    var q: u32;
    if (aligned) {
        q = W[o];
    } else {
        q = (W[o] >> 16u) | (W[o + 1u] << 16u);
    }
    let v = vec4<i32>(bitcast<i32>(q << 24u), bitcast<i32>(q << 16u), bitcast<i32>(q << 8u), bitcast<i32>(q)) >> vec4<u32>(24u);
    return vec4<f32>(v) * d;
}
