// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// render one markup (stdin) → HTML (stdout): the editor round-trip check (ui/editor.js) feeds on it
use std::io::Read;

fn main() -> anyhow::Result<()> {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s)?;
    print!("{}", ocr_core::render::Marks::profile()?.html(&s));
    Ok(())
}
