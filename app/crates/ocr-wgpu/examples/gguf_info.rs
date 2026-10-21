// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// print a GGUF's metadata + tensor table (llama.cpp's gguf-dump, in Rust): cross-check of the reader
//   cargo run --release -p ocr-wgpu --example gguf_info -- <file.gguf> [--all-tensors]
use std::path::PathBuf;

use anyhow::{Context, Result};
use ocr_wgpu::gguf::Gguf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(args.get(1).context("usage: gguf_info <file.gguf> [--all-tensors]")?);
    let all = args.iter().any(|a| a == "--all-tensors");
    let g = Gguf::open(&path)?;
    println!("{}: GGUF v{}, {} KVs, {} tensors, alignment {}, data at {}", path.display(), g.version, g.metadata.len(), g.tensors.len(), g.alignment, g.data_offset);
    for (k, v) in &g.metadata {
        println!("  {k} = {v}");
    }
    let mut counts = std::collections::BTreeMap::new();
    for t in &g.tensors {
        *counts.entry(format!("{:?}", t.ty)).or_insert(0usize) += 1;
    }
    println!("types: {counts:?}");
    for t in &g.tensors {
        // block 0 + the non-block tensors; --all-tensors for every one
        let blk = t.name.split('.').find_map(|p| p.parse::<u32>().ok());
        if all || blk.is_none_or(|b| b == 0) || !t.name.contains("blk.") {
            println!("  {:<40} {:<5} {:?}", t.name, format!("{:?}", t.ty), t.shape);
        }
    }
    Ok(())
}
