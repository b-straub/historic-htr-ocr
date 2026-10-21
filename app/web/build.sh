#!/bin/sh
# OCR Lite — Copyright (c) 2026 Bernhard Straub
# SPDX-License-Identifier: MIT
#
# OCR Lite's browser build: crates/ocr-web → wasm32 → wasm-bindgen (--target web: plain ES modules, no bundler) →
# web/pkg/ (generated, not in git). The wasm-bindgen CLI must be the exact version of the crate in Cargo.lock.
#   web/build.sh && web/serve.py          then open http://127.0.0.1:8765/
set -eu
cd "$(dirname "$0")/.."
want=$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | sed -n 's/^version = "\(.*\)"$/\1/p')
have=$(wasm-bindgen --version 2>/dev/null | awk '{print $2}' || true)
if [ "$have" != "$want" ]; then
  echo "wasm-bindgen CLI '${have:-missing}', the crate is $want — install it: cargo install wasm-bindgen-cli --version '=$want' --locked" >&2
  exit 1
fi
# no build-machine paths in the wasm (panic locations of the dependencies and std name their source files): ~/.cargo,
# the toolchain and this workspace are remapped
sysroot=$(rustc --print sysroot)
export RUSTFLAGS="--remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo --remap-path-prefix=$sysroot=/rustc --remap-path-prefix=$(pwd)=."
cargo build --release --target wasm32-unknown-unknown -p ocr-web
wasm-bindgen --target web --no-typescript --out-dir web/pkg target/wasm32-unknown-unknown/release/ocr_web.wasm
ls -l web/pkg
