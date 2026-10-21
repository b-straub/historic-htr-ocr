// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Y[M,N] (+)= X[M,K] · W[N,K]ᵀ on the matrix units with X and W both f16 in memory (the vision encoder on F16Coop:
// X from an f16-writing producer — rms_norm_h, attention_coop, the FFN below — W converted bf16 → f16 at load): the
// 8×8 blocks are loaded straight from memory (no workgroup staging, no barrier: the four simdgroups run
// independently and share their rows / columns through the cache). Output tile 64×64 per workgroup, 32×32 per
// simdgroup = 4×4 f32 accumulators, k in steps of 8 in order (matmul_coop.wgsl's order: the same sums).
// Preconditions (gpu.rs matmul_f16x): N % 64 == 0, K % 8 == 0, X and Y hold whole 64-row tiles (rows M … padded:
// read as they are, their outputs written into Y's padding). acc: Y's values are the accumulators' start.
enable f16;
enable wgpu_cooperative_matrix;

struct Params {
    M: u32,
    N: u32,
    K: u32,
    acc: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read> W: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let sg = li >> 5u;
    let row0 = wg.y * 64u + (sg >> 1u) * 32u;
    let col0 = wg.x * 64u + (sg & 1u) * 32u;
    //@COOP var c_I_J = coop_mat8x8<f32, C>();
    if (p.acc == 1u) {
        //@COOP c_I_J = coopLoadT<coop_mat8x8<f32, C>>(&Y[(row0 + I * 8u) * p.N + col0 + J * 8u], p.N);
    }
    for (var k = 0u; k < p.K; k += 8u) {
        //@COOPA let a_I = coopLoadT<coop_mat8x8<f16, A>>(&X[(row0 + I * 8u) * p.K + k], p.K);
        //@COOPA let b_I = coopLoad<coop_mat8x8<f16, B>>(&W[(col0 + I * 8u) * p.K + k], p.K);
        //@COOP c_I_J = coopMultiplyAdd(a_I, b_J, c_I_J);
    }
    //@COOP coopStoreT(c_I_J, &Y[(row0 + I * 8u) * p.N + col0 + J * 8u], p.N);
}
