// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// the built-in auto-crop on scans → one line per image in `cropprobe --rect`'s format, so the two can be diffed:
//   autocrop [--keep-spread] [--single-leaf] [--page-side left|right] <images…>
//   <name>: crop <w>×<h> at <x>,<y>
// --single-leaf sets keep_spread too (cropprobe: keepSpread = isSpread || singleLeaf). cropprobe has no --page-side.
use anyhow::{Context, Result, bail};
use ocr_core::autocrop::{AutoCropOptions, PageSide, auto_crop};

fn main() -> Result<()> {
    let mut opts = AutoCropOptions::default();
    let mut paths = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--keep-spread" => opts.keep_spread = true,
            "--single-leaf" => {
                opts.single_leaf = true;
                opts.keep_spread = true;
            }
            "--page-side" => {
                opts.page_side = match args.next().as_deref() {
                    Some("left") => Some(PageSide::Left),
                    Some("right") => Some(PageSide::Right),
                    other => bail!("--page-side needs left | right, got {other:?}"),
                }
            }
            _ => paths.push(a),
        }
    }
    if paths.is_empty() {
        bail!("usage: autocrop [--keep-spread] [--single-leaf] [--page-side left|right] <images…>");
    }
    for p in &paths {
        let img = ocr_core::imageops::load(std::path::Path::new(p)).with_context(|| format!("{p}: cannot load"))?.to_rgb8();
        let r = auto_crop(&img, &opts)?;
        let name = std::path::Path::new(p).file_name().map_or(p.clone(), |n| n.to_string_lossy().into_owned());
        println!("{name}: crop {}×{} at {},{}", r.w, r.h, r.x, r.y);
    }
    Ok(())
}
