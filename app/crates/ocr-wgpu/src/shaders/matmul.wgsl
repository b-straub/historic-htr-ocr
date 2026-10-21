// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Y[M,N] (+)= X[M,K] · W[N,K]ᵀ — ggml_mul_mat(W, X) with W in ggml order [K, N] (K contiguous) and X / Y row-major
// f32. The weight loader (bf16 packed two per u32, or f32) is spliced in at WEIGHT_LOADER by gpu.rs. Accumulation in
// f32. acc = 1 adds onto Y (the residual stream: attn_out / ffn_down write straight into it).
// xround = 1 / 2 rounds X to bf16 / f16 on load: llama.cpp Metal's kernel_mul_mm casts the activations to the weight's
// type (bf16, and half for f32 weights) — only for the precision bisect (Precision::LlamaMetal), never the default.
// Tiling: 64×64 output tile per 16×16 workgroup, 4×4 outputs per thread (rows ty+16i, cols tx+16j), K in steps of 16.
// The 4×4 accumulator is four vec4 registers, never an indexed array: naga's loop guards stop Metal from unrolling, and
// an indexed private array then lives in slow thread memory (measured: 30× slower).
// The big products run matmul128.wgsl (same bits) or, on the f16 paths, its f16 tiles / matmul_coop.wgsl; this
// kernel stays for K % 4 ≠ 0 and the precision bisect.

struct Params {
    M: u32,
    N: u32,
    K: u32,
    acc: u32,
    xround: u32,
}

fn round_x(v: f32) -> f32 {
    if (p.xround == 1u) {
        let b = bitcast<u32>(v);
        return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
    }
    if (p.xround == 2u) {
        return unpack2x16float(pack2x16float(vec2<f32>(v, 0.0))).x;
    }
    return v;
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

//WEIGHT_LOADER

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;

var<workgroup> As: array<f32, 1024>; // [BK][BM]
var<workgroup> Bs: array<f32, 1024>; // [BK][BN]

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wg.y * BM;
    let col0 = wg.x * BN;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < p.K; k0 += BK) {
        for (var i = 0u; i < 4u; i++) {
            let e = li + i * 256u;
            let r = e / BK;
            let c = e % BK;
            let gk = k0 + c;
            var a = 0.0;
            let gr = row0 + r;
            if (gr < p.M && gk < p.K) {
                a = round_x(X[gr * p.K + gk]);
            }
            As[c * BM + r] = a;
            var b = 0.0;
            let gn = col0 + r;
            if (gn < p.N && gk < p.K) {
                b = load_w(gn * p.K + gk);
            }
            Bs[c * BN + r] = b;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk++) {
            let ar = kk * BM + lid.y;
            let br = kk * BN + lid.x;
            let a = vec4<f32>(As[ar], As[ar + 16u], As[ar + 32u], As[ar + 48u]);
            let b = vec4<f32>(Bs[br], Bs[br + 16u], Bs[br + 32u], Bs[br + 48u]);
            acc0 = fma(vec4<f32>(a.x), b, acc0);
            acc1 = fma(vec4<f32>(a.y), b, acc1);
            acc2 = fma(vec4<f32>(a.z), b, acc2);
            acc3 = fma(vec4<f32>(a.w), b, acc3);
        }
        workgroupBarrier();
    }
    store_row(row0 + lid.y, col0 + lid.x, acc0);
    store_row(row0 + lid.y + 16u, col0 + lid.x, acc1);
    store_row(row0 + lid.y + 32u, col0 + lid.x, acc2);
    store_row(row0 + lid.y + 48u, col0 + lid.x, acc3);
}

// one thread's 4 outputs of row r: columns c, c+16, c+32, c+48
fn store_row(r: u32, c: u32, v: vec4<f32>) {
    if (r >= p.M) {
        return;
    }
    store(r, c, v.x);
    store(r, c + 16u, v.y);
    store(r, c + 32u, v.z);
    store(r, c + 48u, v.w);
}

fn store(r: u32, c: u32, v: f32) {
    if (c >= p.N) {
        return;
    }
    let o = r * p.N + c;
    if (p.acc == 1u) {
        Y[o] = Y[o] + v;
    } else {
        Y[o] = v;
    }
}
