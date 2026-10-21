// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// ocr-wgpu: the purpose-built LightOnOCR engine on wgpu (steps 1–3 of the port: GGUF, vision encoder, text decoder).
//   gguf      = the ocr-gguf crate: GGUF reader (header, metadata, tensor table, mapped data) — no llama.cpp
//   gpu       wgpu device + buffers + the WGSL kernels (matmul, norms, rope, attention, …) as dispatchable ops
//   vision    the mmproj (LightOnOCR = Pixtral graph of llama.cpp's clip.cpp) on the GPU
//   tokenizer the text model's byte-level BPE (gpt2 / qwen2 pre-tokenizer) from the GGUF metadata
//   text      the text decoder (llama.cpp's Qwen3 graph): q8_0 weights, f16 KV cache, prefill + decode step
//   engine    image → read: the prompt exactly as ocr-core's llama.cpp + mtmd engine builds it, ocr-decode's read loop

pub use ocr_gguf as gguf;
/// the wgpu this engine is built on (the browser build keeps a device handle for its lost-device selftest)
pub use wgpu;
pub mod engine;
pub mod gpu;
pub mod text;
pub mod tokenizer;
pub mod vision;
