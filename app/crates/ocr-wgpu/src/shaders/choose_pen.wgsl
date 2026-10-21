// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// GPU token choice (greedy presets), part 1/3: the windowed repetition penalty, in place on the logits — exactly
// ocr-decode's choose(): each DISTINCT token among the last `window` history entries Hist[len − window … len) (the
// prompt tail as the read seeded it, then every chosen token) → logit < 0: × rp, else ÷ rp. Bit-identical to the CPU:
// the product is one correctly rounded f32 multiply (a product below f32's normal range may flush to zero on a GPU —
// a logit × rp never gets near 1e-38); the quotient is computed exactly in integers (div_rn, subnormals included)
// because WGSL's f32 division is only accurate to 2.5 ULP. One workgroup; a token's first occurrence in the window applies
// it (distinct tokens → no two threads write the same logit).

struct Params {
    len: u32,
    window: u32,
    rp: f32,
    _p0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> Hist: array<u32>;
@group(0) @binding(2) var<storage, read_write> L: array<f32>;

// x / y rounded to nearest-even like IEEE-754 f32 division, for y a positive normal f32 (rp, checked on the host)
fn div_rn(x: f32, y: f32) -> f32 {
    let bx = bitcast<u32>(x);
    let ax = bx & 0x7fffffffu;
    // ±0 / y = ±0, ±inf / y = ±inf, NaN stays NaN
    if (ax == 0u || ax >= 0x7f800000u) {
        return x;
    }
    var ex = i32(ax >> 23u);
    var mx = ax & 0x7fffffu;
    if (ex == 0) {
        ex = 1;
        while (mx < 0x800000u) {
            mx = mx << 1u;
            ex -= 1;
        }
    } else {
        mx = mx | 0x800000u;
    }
    let by = bitcast<u32>(y);
    let my = (by & 0x7fffffu) | 0x800000u;
    var e = ex - i32(by >> 23u) + 127;
    if (mx < my) {
        mx = mx << 1u;
        e -= 1;
    }
    // q = floor(mx · 2^25 / my) ∈ [2^25, 2^26); r ≠ 0 ⇔ a remainder below the last bit
    var r = mx;
    var q = 0u;
    for (var k = 0u; k < 26u; k++) {
        q = q << 1u;
        if (r >= my) {
            r -= my;
            q = q | 1u;
        }
        r = r << 1u;
    }
    // 24 bits stay for a normal result, fewer for a subnormal one (q < 2^26: a shift of 27 leaves 0)
    var sh = 2u;
    if (e < 1) {
        sh = min(u32(1 - e) + 2u, 27u);
    }
    var m = q >> sh;
    let rem = q & ((1u << sh) - 1u);
    let half = 1u << (sh - 1u);
    if (rem > half || (rem == half && (r != 0u || (m & 1u) == 1u))) {
        m += 1u;
    }
    var bits: u32;
    if (e < 1) {
        // m = 2^23 after rounding up is the smallest normal's encoding
        bits = m;
    } else {
        // m ∈ [2^23, 2^24]: its implicit bit adds to the exponent field (2^24 = the next binade)
        bits = (u32(e - 1) << 23u) + m;
        if (bits >= 0x7f800000u) {
            bits = 0x7f800000u;
        }
    }
    return bitcast<f32>(bits | (bx & 0x80000000u));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_index) li: u32) {
    let start = p.len - min(p.len, p.window);
    for (var j = start + li; j < p.len; j += 64u) {
        let t = Hist[j];
        var first = true;
        for (var k = start; k < j; k++) {
            if (Hist[k] == t) {
                first = false;
                break;
            }
        }
        if (first) {
            let x = L[t];
            if (x < 0.0) {
                L[t] = x * p.rp;
            } else {
                L[t] = div_rn(x, p.rp);
            }
        }
    }
}
