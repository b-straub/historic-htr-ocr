// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! The layout tools in pure Rust — the desktop GT tools' region geometry, ported: polygons in crop px (y down).
//! `geom`: basics, simplify outward, node editing, the knife (clip cut + split), straighten; `shape`: the raster steps
//! (merge, enclosing hull, snap to the border, clear of others, straight sides, subtract); `raster`: the binary raster
//! under them; `ink`: snap to ink and the line measures (the desktop tools' ink geometry; work in progress there too);
//! `crop`: the text crop (the page's main text block — OCR Lite's Auto).
//! No image, browser or platform dependency: the editor reaches it through ocr-web, a native app can link it.

pub mod crop;
pub mod geom;
pub mod ink;
pub mod order;
pub mod raster;
pub mod scale;
pub mod shape;
pub mod tools;

pub use geom::{Failed, Pt, Rect};

#[cfg(all(target_arch = "wasm32", feature = "web"))]
mod web;
