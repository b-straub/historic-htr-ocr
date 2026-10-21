// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Y[M,N] (+)= X[M,K] · W[N,K]ᵀ on the GPU's matrix units (Metal simdgroup matrices through wgpu's
// wgpu_cooperative_matrix, native only): the f16 compute path's big products. X and W go through workgroup memory as
// f16 (W from its GGUF encoding via the vec4 loader spliced in at WEIGHT_LOADER4), products accumulate in f32 8×8
// matrices. 128 threads = 4 simdgroups of 32 (the adapter's subgroup size is checked to be 32 in gpu.rs); output tile
// 64×64, each simdgroup 32×32 = 4×4 accumulators; k in steps of 32 (two tiles of 64 × 32 f16 = 8 KB).
// The accumulators leave through a per-simdgroup 8×8 f32 scratch (bounds + acc handled per element).
// Measured (M3, a full test page's vision encoder, gate / up GPU ms per call): this 42–47; a 128×128 tile (8 simdgroups of 64×32,
// half the tile reloads, bit-identical) 47 — the tile traffic is not the limit; the next tile prefetched into
// registers during the products (bit-identical) 46.
enable f16;
enable wgpu_cooperative_matrix;

struct Params {
    M: u32,
    N: u32,
    K: u32,
    acc: u32,
    xround: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

//WEIGHT_LOADER4

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 32u;

var<workgroup> As: array<f16, 2048>; // [BM][BK] row-major
var<workgroup> Bs: array<f16, 2048>; // [BN][BK]: column n of Wᵀ contiguous
var<workgroup> Cs: array<f32, 256>; // 4 simdgroups × 8×8

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wg.y * BM;
    let col0 = wg.x * BN;
    let k4 = p.K >> 2u;
    let sg = li >> 5u;
    let lane = li & 31u;
    let sr = (sg >> 1u) * 32u; // the simdgroup's rows / columns in the tile
    let sc = (sg & 1u) * 32u;
    //@COOP var c_I_J = coop_mat8x8<f32, C>();
    for (var k0 = 0u; k0 < p.K; k0 += BK) {
        // 64 rows × 32 k = 512 vec4 per tile: 4 per thread for X, 4 for W
        for (var i = 0u; i < 4u; i++) {
            let e = li + i * 128u;
            let r = e >> 3u;
            let kq = (e & 7u) * 4u;
            let gk = k0 + kq;
            var a = vec4<f32>(0.0);
            if (row0 + r < p.M && gk < p.K) {
                a = X[(row0 + r) * k4 + (gk >> 2u)];
            }
            var b = vec4<f32>(0.0);
            if (col0 + r < p.N && gk < p.K) {
                b = load_w4((col0 + r) * p.K + gk);
            }
            let o = r * BK + kq;
            As[o] = f16(a.x);
            As[o + 1u] = f16(a.y);
            As[o + 2u] = f16(a.z);
            As[o + 3u] = f16(a.w);
            Bs[o] = f16(b.x);
            Bs[o + 1u] = f16(b.y);
            Bs[o + 2u] = f16(b.z);
            Bs[o + 3u] = f16(b.w);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk += 8u) {
            //@COOPA let a_I = coopLoadT<coop_mat8x8<f16, A>>(&As[(sr + I * 8u) * BK + kk], BK);
            //@COOPA let b_I = coopLoad<coop_mat8x8<f16, B>>(&Bs[(sc + I * 8u) * BK + kk], BK);
            //@COOP c_I_J = coopMultiplyAdd(a_I, b_J, c_I_J);
        }
        workgroupBarrier();
    }
    //@COOP put(c_I_J, sg, lane, row0 + sr + I * 8u, col0 + sc + J * 8u);
}

// one 8×8 accumulator out: through the simdgroup's scratch, two elements per lane
fn put(c: coop_mat8x8<f32, C>, sg: u32, lane: u32, r0: u32, c0: u32) {
    coopStoreT(c, &Cs[sg * 64u], 8u);
    workgroupBarrier();
    for (var e = lane; e < 64u; e += 32u) {
        let r = r0 + e / 8u;
        let col = c0 + e % 8u;
        if (r < p.M && col < p.N) {
            let o = r * p.N + col;
            let v = Cs[sg * 64u + e];
            if (p.acc == 1u) {
                Y[o] = Y[o] + v;
            } else {
                Y[o] = v;
            }
        }
    }
    workgroupBarrier();
}
