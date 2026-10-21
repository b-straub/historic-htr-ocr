// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// bf16 weights → f16 in place, once at load, times `scale` (a power of two: exact — 1, or the down projection's
// FFN_DOWN_SCALE, vision.rs) (the f16 compute paths: their matmuls read f16 directly — the value each
// of them rounded every tile before, f16(bf16), now rounded once; same buffer, same size). Two values per u32 word.
// A word holding a value whose magnitude exceeds f16's largest (65504) is counted in over[0]: the load refuses such a model on the
// f16 paths (gpu.rs Gpu::weights_to_f16) instead of computing with inf.
enable f16;

struct Params {
    n: u32, // words
    scale: f32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> over: array<atomic<u32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var i = gid.x; i < p.n; i += nwg.x * 256u) {
        let w = W[i];
        let v = vec2<f32>(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xffff0000u)) * p.scale;
        if (any(abs(v) > vec2<f32>(65504.0))) {
            atomicAdd(&over[0], 1u);
        }
        W[i] = pack2x16float(v);
    }
}
