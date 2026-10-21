// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The prompt around one image: the chat template of one image-only user message (the
// GGUF's tokenizer.chat_template with its system turn), the image where <|image_pad|> stands. The image itself is
// wrapped the way llama.cpp's mtmd does it for the projector (LIGHTONOCR: <|im_start|> … <|im_end|>, mtmd.cpp) —
// ocr-wgpu builds it (its engine.rs).

pub const PREFIX: &str = "<|im_start|>system<|im_end|>\n<|im_start|>user\n";
pub const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// the image's stand-in token: the repetition window starts on one per image token (as hf/window_rep_penalty.py)
pub const IMAGE_PAD: &str = "<|image_pad|>";
