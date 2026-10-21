// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Y[M,N] (+)= X[M,K] · W[N,K]ᵀ — matmul.wgsl's product with a 128×128 output tile: the big products (the vision
// encoder over a page's ~7000 patches, the text prefill). Same arithmetic as matmul.wgsl, bit for bit: every output is
// ONE fma chain over k = 0, 1, … K−1 in order starting from 0 (the zero padding past K only adds 0·0), then stored or
// added onto Y. Only the work split differs: 256 threads (16×16), each 8 rows × 8 columns = 16 vec4 accumulators
// (rows 4·ty … +3 and 64 + 4·ty … +3, columns 4·tx … +3 and 64 + 4·tx … +3); per k four vec4 reads from workgroup
// memory feed 64 fma (matmul.wgsl: eight scalar reads for 16). X and W are read four k at a time (K % 4 == 0, checked
// in gpu.rs) by the vec4 weight loader spliced in at WEIGHT_LOADER4. Workgroup memory: 2 × 16 × 128 f32 = 16 KB
// (WebGPU's default limit). The F16 path builds this kernel with f16 tiles (TILE = vec4<f16>: the inputs rounded to
// f16 on their way into workgroup memory, the fma f32 — 1.45× faster on the M3); BK = 32 with f16 tiles measured equal.
// The tile loads: threads 0…127 fetch X, 128…255 W — four rows' four k each, transposed in registers and stored as
// whole vec4 (a store to one component of a workgroup vec4 lost the other components on Metal).
// The 16 accumulators are named registers, written out at the @ACC lines by gpu.rs (an indexed private array would
// live in slow thread memory).

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
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

fn load_x4(i: u32) -> vec4<f32> {
    let v = X[i];
    return vec4<f32>(round_x(v.x), round_x(v.y), round_x(v.z), round_x(v.w));
}

//WEIGHT_LOADER4

const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 16u;

// TILE = the workgroup tiles' element type: vec4<f32> (F32 path, the identity conversions below) or vec4<f16> (F16
// path: half the workgroup memory traffic, the fma stay f32) — set by gpu.rs
alias Tile = TILE;
var<workgroup> As: array<Tile, 512>; // [BK][BM / 4]
var<workgroup> Bs: array<Tile, 512>; // [BK][BN / 4]

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wg.y * BM;
    let col0 = wg.x * BN;
    let k4 = p.K >> 2u;
    //@ACC var acc_R_C = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < p.K; k0 += BK) {
        // the tiles: 128 rows × 16 k of X (threads 0…127) and of W (128…255), stored transposed ([k][row / 4] vec4):
        // a thread reads four rows' four k (rows 4q … 4q + 3, k 4kq … 4kq + 3) and writes four whole vec4 (one per k)
        for (var t = li & 127u; t < 8u * BK; t += 128u) {
        let q = t / (BK / 4u);
        let kq = (t % (BK / 4u)) * 4u;
        let gk = k0 + kq;
        var v0 = vec4<f32>(0.0);
        var v1 = vec4<f32>(0.0);
        var v2 = vec4<f32>(0.0);
        var v3 = vec4<f32>(0.0);
        if (li < 128u) {
            let gr = row0 + q * 4u;
            if (gk < p.K) {
                if (gr < p.M) { v0 = load_x4(gr * k4 + (gk >> 2u)); }
                if (gr + 1u < p.M) { v1 = load_x4((gr + 1u) * k4 + (gk >> 2u)); }
                if (gr + 2u < p.M) { v2 = load_x4((gr + 2u) * k4 + (gk >> 2u)); }
                if (gr + 3u < p.M) { v3 = load_x4((gr + 3u) * k4 + (gk >> 2u)); }
            }
            As[(kq + 0u) * 32u + q] = Tile(vec4<f32>(v0.x, v1.x, v2.x, v3.x));
            As[(kq + 1u) * 32u + q] = Tile(vec4<f32>(v0.y, v1.y, v2.y, v3.y));
            As[(kq + 2u) * 32u + q] = Tile(vec4<f32>(v0.z, v1.z, v2.z, v3.z));
            As[(kq + 3u) * 32u + q] = Tile(vec4<f32>(v0.w, v1.w, v2.w, v3.w));
        } else {
            let gn = col0 + q * 4u;
            if (gk < p.K) {
                if (gn < p.N) { v0 = load_w4(gn * p.K + gk); }
                if (gn + 1u < p.N) { v1 = load_w4((gn + 1u) * p.K + gk); }
                if (gn + 2u < p.N) { v2 = load_w4((gn + 2u) * p.K + gk); }
                if (gn + 3u < p.N) { v3 = load_w4((gn + 3u) * p.K + gk); }
            }
            Bs[(kq + 0u) * 32u + q] = Tile(vec4<f32>(v0.x, v1.x, v2.x, v3.x));
            Bs[(kq + 1u) * 32u + q] = Tile(vec4<f32>(v0.y, v1.y, v2.y, v3.y));
            Bs[(kq + 2u) * 32u + q] = Tile(vec4<f32>(v0.z, v1.z, v2.z, v3.z));
            Bs[(kq + 3u) * 32u + q] = Tile(vec4<f32>(v0.w, v1.w, v2.w, v3.w));
        }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk++) {
            let a0 = vec4<f32>(As[kk * 32u + lid.y]);
            let a1 = vec4<f32>(As[kk * 32u + 16u + lid.y]);
            let b0 = vec4<f32>(Bs[kk * 32u + lid.x]);
            let b1 = vec4<f32>(Bs[kk * 32u + 16u + lid.x]);
            //@ACC acc_R_C = fma(vec4<f32>(AROW), BCOL, acc_R_C);
        }
        workgroupBarrier();
    }
    //@ACC store4(OUTROW, col0 + OUTCOL + lid.x * 4u, acc_R_C);
}

// one thread's 4 outputs of row r: columns c … c + 3
fn store4(r: u32, c: u32, v: vec4<f32>) {
    if (r >= p.M) {
        return;
    }
    store(r, c, v.x);
    store(r, c + 1u, v.y);
    store(r, c + 2u, v.z);
    store(r, c + 3u, v.w);
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
