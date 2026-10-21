// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The patch merger's unfold (pixtral.cpp: ggml_im2col with a 2×2 kernel, stride 2 over the [x, y, n_embd] grid):
// merged token o = oy·(px/2) + ox gets Z[o][c·4 + ky·2 + kx] = X[(2·oy + ky)·px + 2·ox + kx][c] — channel-major,
// then kernel row, then kernel column (torch unfold's order).

struct Params {
    px: u32,  // patches per row (even)
    py: u32,  // patch rows (even)
    dim: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Z: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let ox_n = p.px / 2u;
    let total = (p.px / 2u) * (p.py / 2u) * p.dim * 4u;
    for (var i = gid.x; i < total; i += nwg.x * 256u) {
        let kx = i % 2u;
        let ky = (i / 2u) % 2u;
        let c = (i / 4u) % p.dim;
        let o = i / (4u * p.dim);
        let ox = o % ox_n;
        let oy = o / ox_n;
        Z[i] = X[((2u * oy + ky) * p.px + 2u * ox + kx) * p.dim + c];
    }
}
