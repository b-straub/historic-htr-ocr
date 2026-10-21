// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The vision FFN's first half fused on the matrix units: G[M,F] = silu(X·Wgᵀ) · (X·Wuᵀ) as f16 (swiglu.wgsl's
// formula on the f32 sums), stored times out_scale (1 / FFN_DOWN_SCALE, vision.rs: the product reaches 3.6e4 on a
// real page, f16 ends at 65504; the down weights carry the factor back) — matmul_coopd.wgsl's direct f16 loads for both products, the gate / up sums never leave
// the GPU core (no f32 gate / up buffers: 2 × 4 bytes × M × F written and read back per layer before). Output tile
// 64 rows × 32 columns per workgroup, 32 × 16 per simdgroup = 4×2 accumulators per product; the epilogue goes through
// the simdgroup's 2 × 8×8 f32 scratch. Preconditions as matmul_coopd (F % 32 == 0, whole 64-row tiles).
enable f16;
enable wgpu_cooperative_matrix;

struct Params {
    M: u32,
    N: u32,
    K: u32,
    out_scale: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read> Wg: array<f16>;
@group(0) @binding(3) var<storage, read> Wu: array<f16>;
@group(0) @binding(4) var<storage, read_write> G: array<f16>;

var<workgroup> Cs: array<f32, 512>; // 4 simdgroups × (gate, up) × 8×8

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let sg = li >> 5u;
    let lane = li & 31u;
    let row0 = wg.y * 64u + (sg >> 1u) * 32u;
    let col0 = wg.x * 32u + (sg & 1u) * 16u;
    //@C42 var g_I_J = coop_mat8x8<f32, C>();
    //@C42 var u_I_J = coop_mat8x8<f32, C>();
    for (var k = 0u; k < p.K; k += 8u) {
        //@COOPA let a_I = coopLoadT<coop_mat8x8<f16, A>>(&X[(row0 + I * 8u) * p.K + k], p.K);
        //@J2 let bg_J = coopLoad<coop_mat8x8<f16, B>>(&Wg[(col0 + J * 8u) * p.K + k], p.K);
        //@J2 let bu_J = coopLoad<coop_mat8x8<f16, B>>(&Wu[(col0 + J * 8u) * p.K + k], p.K);
        //@C42 g_I_J = coopMultiplyAdd(a_I, bg_J, g_I_J);
        //@C42 u_I_J = coopMultiplyAdd(a_I, bu_J, u_I_J);
    }
    //@C42 put(g_I_J, u_I_J, sg, lane, row0 + I * 8u, col0 + J * 8u);
}

// one 8×8 output block: gate and up through the scratch, two elements per lane
fn put(g: coop_mat8x8<f32, C>, u: coop_mat8x8<f32, C>, sg: u32, lane: u32, r0: u32, c0: u32) {
    coopStoreT(g, &Cs[sg * 128u], 8u);
    coopStoreT(u, &Cs[sg * 128u + 64u], 8u);
    subgroupBarrier();
    for (var e = lane; e < 64u; e += 32u) {
        let gv = Cs[sg * 128u + e];
        let uv = Cs[sg * 128u + 64u + e];
        G[(r0 + e / 8u) * p.N + c0 + e % 8u] = f16(gv / (1.0 + exp(-gv)) * uv * p.out_scale);
    }
    subgroupBarrier();
}
