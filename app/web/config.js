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
// Another GGUF repo of the same architecture works too: set its base, repo, releases and, when its file names differ,
// remoteFiles. The user-facing name is translated (ui/lang: model.name); base is the technical one, never translated.
//
// Releases are named by DATE (no v1 / v2), newest first: the first is the current one — the one downloaded; Settings
// offers it where an older one is stored. A release's md5 = its files' MD5 (the repo's MD5SUMS): the model check
// (Settings → Verify, and by itself after a crash: iOS ending the app in the middle of a read left the stored model
// damaged), and the version of a model stored before the version was recorded.
export const MODEL = {
  /// the technical name (the small print in Settings, About, the status line)
  base: "LightOnOCR-3-1B",
  /// the Hugging Face model repo
  repo: "https://huggingface.co/b-straub/LightOnOCR-3-1B-Historic-German-Handwriting",
  /// the repo's file names when they differ from MODEL_FILES ({ stored name: name in the repo }), else null
  remoteFiles: null,
  /// { version, revision (the repo's git revision the files are read from: resolve/<revision>/), md5 }, newest first.
  /// revision = the repo's tag model-<version> (made by hand on Hugging Face for each release; the repo's v1.x.y tags
  /// are app versions, set by the release job — not model releases)
  releases: [
    {
      version: "2026.10",
      revision: "model-2026.10",
      md5: { "text-q8_0.gguf": "5b0d07481b0d65e5a0e2d317d78cc410", "mmproj-bf16.gguf": "01481f8c70a20012169830adcfbb75c9" },
    },
  ],
};
/// the release downloaded now
export const CURRENT = MODEL.releases[0];

/// the folder the model files are downloaded from (a test build: web/dist.py --model-base replaces it with a folder on
/// the app's own web space, and MODEL_FROM with "this server (test build)")
export const MODEL_BASE = `${MODEL.repo}/resolve/${CURRENT.revision}/`;
/// for texts: "downloaded once from …"
export const MODEL_FROM = "Hugging Face";
/// a test build (web/dist.py --model-base sets it): the model comes from MODEL_BASE on the app's own web space — the
/// texts that name Hugging Face as the source say so
export const MODEL_TEST = false;

/// where a model may come from — ONE switch for the page (the setup's and Settings' controls, ?model=) and the engine
/// worker (told at its start: its commands refuse a source not listed):
///   hub     the model's Hugging Face repo (MODEL_BASE)
///   upload  the user's own two files (setup "I have the files", Settings "Upload model files…") — stored only when
///           their MD5s are a release's (worker.js upload): the same model as the hub's, got another way
///   url     a download from another folder URL (Settings) — off
///   stream  ?model=<folder>: the model read by HTTP range requests, never stored — off
/// A model stored earlier from a source not listed is not used: the setup downloads the hub's.
export const MODEL_SOURCES = ["hub", "upload"];

/// the project's links (About, setup)
export const LINKS = {
  card: MODEL.repo,
  repo: "https://github.com/b-straub/historic-htr-ocr",
  site: "https://b-straub.github.io/historic-htr-ocr/",
};
