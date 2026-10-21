// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The decode step between the q / k / v projections and the attention, in one dispatch over the [q | k | v] buffer
// of dmv_qkv.wgsl (one token at position pos), one workgroup of 128 threads (the head dim) per head:
//   q head h < n_head          per-head RMSNorm (q_norm) → RoPE NEOX → back in place (the attention reads it there)
//   k head n_head + j          per-head RMSNorm (k_norm) → RoPE NEOX → the f16 K cache row pos
//   v head n_head + n_kv + j   → the f16 V cache row pos
// The formulas of rms_norm.wgsl, rope_neox.wgsl and kv_store.wgsl; replaces their six dispatches per layer.

struct Params {
    n_head: u32,
    n_kv: u32,
    pos: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> cs: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> QKV: array<f32>;
@group(0) @binding(3) var<storage, read> QN: array<f32>;
@group(0) @binding(4) var<storage, read> KN: array<f32>;
@group(0) @binding(5) var<storage, read_write> Kc: array<u32>;
@group(0) @binding(6) var<storage, read_write> Vc: array<u32>;

const D: u32 = 128u;
const HALF: u32 = 64u;

var<workgroup> sh: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let h = wg.x;
    let kvd2 = p.n_kv * HALF; // u32 words per cache row
    if (h >= p.n_head + p.n_kv) {
        // v: f32 pairs → f16 pairs
        let vh = h - p.n_head - p.n_kv;
        if (i < HALF) {
            let o = h * D + 2u * i;
            Vc[p.pos * kvd2 + vh * HALF + i] = pack2x16float(vec2<f32>(QKV[o], QKV[o + 1u]));
        }
        return;
    }
    let o = h * D;
    let x = QKV[o + i];
    sh[i] = x * x;
    workgroupBarrier();
    for (var n = 64u; n > 0u; n >>= 1u) {
        if (i < n) {
            sh[i] += sh[i + n];
        }
        workgroupBarrier();
    }
    let scale = 1.0 / sqrt(sh[0] / f32(D) + p.eps);
    workgroupBarrier();
    let is_q = h < p.n_head;
    var y: f32;
    if (is_q) {
        y = x * scale * QN[i];
    } else {
        y = x * scale * KN[i];
    }
    sh[i] = y;
    workgroupBarrier();
    let k = i % HALF;
    let c = cs[p.pos * HALF + k];
    var r: f32;
    if (i < HALF) {
        r = sh[i] * c.x - sh[i + HALF] * c.y;
    } else {
        r = sh[i - HALF] * c.y + sh[i] * c.x;
    }
    if (is_q) {
        QKV[o + i] = r;
        return;
    }
    workgroupBarrier();
    sh[i] = r;
    workgroupBarrier();
    if (i < HALF) {
        Kc[p.pos * kvd2 + (h - p.n_head) * HALF + i] = pack2x16float(vec2<f32>(sh[2u * i], sh[2u * i + 1u]));
    }
}
