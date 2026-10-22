// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The browser build's configuration: where the model comes from and which sources may store it. The one place to
// change when the model changes (app/README.md "The model").
//
// The model: two GGUF files on Hugging Face, stored in this browser under MODEL_FILES (text-q8_0.gguf +
// mmproj-bf16.gguf). Hugging Face `resolve` URLs answer a browser fetch with CORS (a redirect to its CDN, both with
// Access-Control-Allow-Origin), so the worker streams the files straight into the browser's storage (OPFS).
//
// The model: LightOnOCR-3-1B · Historic German Handwriting, whose repo names its files as stored (remoteFiles null).
// Another GGUF repo of the same architecture works too: set its name, repo and, when its file names differ, remoteFiles.
export const MODEL = {
  /// shown in the setup and Settings
  name: "LightOnOCR-3-1B · Historic German Handwriting",
  /// the Hugging Face model repo (the files are read from its resolve/main/)
  repo: "https://huggingface.co/b-straub/LightOnOCR-3-1B-Historic-German-Handwriting",
  /// the repo's file names when they differ from MODEL_FILES ({ stored name: name in the repo }), else null
  remoteFiles: null,
  /// the files' MD5 as released (the repo's MD5SUMS) — the model check (Settings → Verify model, and by itself after a
  /// crash: iOS ending the app in the middle of a read left the stored model damaged)
  md5: { "text-q8_0.gguf": "5b0d07481b0d65e5a0e2d317d78cc410", "mmproj-bf16.gguf": "01481f8c70a20012169830adcfbb75c9" },
};

/// the folder the model files are downloaded from
export const MODEL_BASE = `${MODEL.repo}/resolve/main/`;
/// for texts: "downloaded once from …"
export const MODEL_FROM = "Hugging Face";

/// where a model may come from — ONE switch for the page (the setup's and Settings' controls, ?model=) and the engine
/// worker (told at its start: its commands refuse a source not listed):
///   hub     the model's Hugging Face repo (MODEL_BASE)
///   upload  the user's own two files (setup "I have the files", Settings "Upload model files…") — off
///   url     a download from another folder URL (Settings) — off
///   stream  ?model=<folder>: the model read by HTTP range requests, never stored — off
/// A model stored earlier from a source not listed is not used: the setup downloads the hub's.
export const MODEL_SOURCES = ["hub"];

/// the project's links (About, setup)
export const LINKS = {
  card: MODEL.repo,
  repo: "https://github.com/b-straub/historic-htr-ocr",
  site: "https://b-straub.github.io/historic-htr-ocr/",
};
