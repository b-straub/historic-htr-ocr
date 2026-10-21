// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// cut a rectangle out of a baked page into a new PNG: the small crops of the fast dev loop (sizes on the 28 grid)
//   cargo run --release -p ocr-wgpu --example cut -- <src.png> <x> <y> <w> <h> <out.png>
use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 7 {
        bail!("usage: cut <src.png> <x> <y> <w> <h> <out.png>");
    }
    let n = |i: usize| -> Result<u32> { a[i].parse().with_context(|| format!("argument {i} is not a number")) };
    let (x, y, w, h) = (n(2)?, n(3)?, n(4)?, n(5)?);
    if w % 28 != 0 || h % 28 != 0 {
        bail!("{w}x{h} is not on the 28 px grid");
    }
    let img = image::open(&a[1])?.to_rgb8();
    if x + w > img.width() || y + h > img.height() {
        bail!("{x},{y} {w}x{h} leaves the {}x{} image", img.width(), img.height());
    }
    image::imageops::crop_imm(&img, x, y, w, h).to_image().save(&a[6])?;
    Ok(())
}
