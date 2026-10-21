// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The vision encoder's input on the GPU: the baked image's RGB bytes → normalised patches, im2col'd for the patch
// conv (clip_image_f32 from_u8 / normalize, then the conv's patch layout c, ky, kx):
//   P[t][c·p·p + ky·p + kx] = lut[c·256 + rgb[((ty·p + ky)·w + tx·p + kx)·3 + c]],   t = ty·px + tx
// lut[c·256 + v] = (v / 255 − mean[c]) / std[c], computed on the CPU (vision.rs) — the same f32 values the CPU
// im2col produced, without the page's patches (≈ 18 MB on a full page) in CPU / wasm memory.

struct Params {
    w: u32,  // image width in pixels
    px: u32, // patches per row
    n: u32,  // patches
    side: u32, // the patch side in pixels
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> rgb: array<u32>; // the RGB bytes, four per word (little-endian)
@group(0) @binding(2) var<storage, read> lut: array<f32>;
@group(0) @binding(3) var<storage, read_write> P: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let pp = p.side * p.side;
    let total = p.n * 3u * pp;
    for (var i = gid.x; i < total; i += nwg.x * 256u) {
        let t = i / (3u * pp);
        let r = i % (3u * pp);
        let c = r / pp;
        let ky = (r % pp) / p.side;
        let kx = r % p.side;
        let x = (t % p.px) * p.side + kx;
        let y = (t / p.px) * p.side + ky;
        let b = (y * p.w + x) * 3u + c;
        let v = (rgb[b >> 2u] >> ((b & 3u) * 8u)) & 0xffu;
        P[i] = lut[c * 256u + v];
    }
}
