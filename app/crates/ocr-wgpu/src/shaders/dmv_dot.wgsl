// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// one q8_0 block of WNAME (34 bytes: f16 scale + 32 signed bytes, packed back to back in u32 words) · 32 inputs. The
// block spans 9 words from word ⌊base / 4⌋ whether it starts on a word boundary or half-way (alternate blocks of an
// odd block count); all nine are loaded once, the quant words assembled from them — no second pass, no branch.
fn dot_WNAME(blk: u32, xs: ptr<function, array<vec4<f32>, 8>>) -> f32 {
    let base = blk * 34u;
    let w = base >> 2u;
    let hi = (base & 2u) != 0u; // the scale is the word's upper half, the quants start on the next word
    let a0 = WNAME[w];
    let a1 = WNAME[w + 1u];
    let a2 = WNAME[w + 2u];
    let a3 = WNAME[w + 3u];
    let a4 = WNAME[w + 4u];
    let a5 = WNAME[w + 5u];
    let a6 = WNAME[w + 6u];
    let a7 = WNAME[w + 7u];
    let a8 = WNAME[w + 8u];
    let d = unpack2x16float(select(a0, a0 >> 16u, hi)).x;
    var s = dot(i8x4(select((a0 >> 16u) | (a1 << 16u), a1, hi)), (*xs)[0]);
    s += dot(i8x4(select((a1 >> 16u) | (a2 << 16u), a2, hi)), (*xs)[1]);
    s += dot(i8x4(select((a2 >> 16u) | (a3 << 16u), a3, hi)), (*xs)[2]);
    s += dot(i8x4(select((a3 >> 16u) | (a4 << 16u), a4, hi)), (*xs)[3]);
    s += dot(i8x4(select((a4 >> 16u) | (a5 << 16u), a5, hi)), (*xs)[4]);
    s += dot(i8x4(select((a5 >> 16u) | (a6 << 16u), a6, hi)), (*xs)[5]);
    s += dot(i8x4(select((a6 >> 16u) | (a7 << 16u), a7, hi)), (*xs)[6]);
    s += dot(i8x4(select((a7 >> 16u) | (a8 << 16u), a8, hi)), (*xs)[7]);
    return s * d;
}
