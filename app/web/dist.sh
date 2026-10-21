#!/bin/sh
# OCR Lite — Copyright (c) 2026 Bernhard Straub
# SPDX-License-Identifier: MIT
#
# OCR Lite's deployable web build → app/dist/ (generated, not in git):
#   1. web/build.sh — the wasm (ocr-web → web/pkg/)
#   2. web/dist.py  — the flat static site: hashed JS / CSS / WASM, index.html, sw.js, manifest (no model: the app
#                     downloads it from Hugging Face)
#   web/dist.sh [--no-wasm] [--out DIR]
set -eu
cd "$(dirname "$0")"
if [ "${1:-}" = "--no-wasm" ]; then
  shift
else
  ./build.sh
fi
exec python3 -I dist.py "$@"
