// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// ocr-decode: the engine-independent part of a read (used by ocr-wgpu; pure Rust, no GPU — builds for the browser too).
//   prompt   the text around the image (chat template prefix / suffix, the image pad)
//   presets  THE decoding presets (app/presets.json, compiled in) + the user's own presets
//   sampling the decoding step on one position's logits (rp → fp → greedy / T + top-p) + the read loop around it

pub mod presets;
pub mod prompt;
pub mod sampling;
