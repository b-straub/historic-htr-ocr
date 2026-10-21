// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// OCR Lite core: everything the app does that is neither UI nor the engine.
//   autocrop the built-in auto crop (find the paper on the scan: decision + safety margin)
//   presets  re-exported from ocr-decode (THE decoding presets; the decoding step lives there too, used by ocr-wgpu)
//   imageops cheap image operations (rotate, contrast, pencil, crop) + the bake (≤ 1540, white pad to 28)
//   render   the tagged markup → HTML (the marks from app/markup.json, compiled in)
// Builds natively and for wasm32 (the browser build).

pub mod autocrop;
pub mod imageops;
pub use ocr_decode::presets;
pub mod render;
