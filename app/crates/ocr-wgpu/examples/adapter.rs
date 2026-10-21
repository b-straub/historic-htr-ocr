// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Is there a GPU adapter wgpu can compute on? Prints it and exits 0; exits 3 when there is none (app/ci.sh: the GPU
// tests of ocr-wgpu run only where this passes, and the CI log says so when they are skipped).
//   cargo run -p ocr-wgpu --example adapter
use ocr_wgpu::gpu::{Compute, Gpu};

fn main() {
    match pollster::block_on(Gpu::with_compute(Some(Compute::F32))) {
        Ok(gpu) => println!("GPU adapter: {} ({:?})", gpu.info.name, gpu.info.backend),
        Err(e) => {
            eprintln!("no GPU adapter: {e:#}");
            std::process::exit(3);
        }
    }
}
