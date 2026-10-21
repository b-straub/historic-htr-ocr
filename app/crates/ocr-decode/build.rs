// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// THE decoding presets (app/presets.json) compiled in as DATA ONLY: the file's prose (`why`, `fields`, `names`,
// `seedRule`) stays out of the binaries; what presets.rs reads (`cliDefault` + `presets`) is written to
// OUT_DIR/presets.json unchanged. No fallback: a file without them fails the build.
use std::{env, fs, path::Path};

fn main() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../presets.json");
    println!("cargo:rerun-if-changed={}", src.display());
    let text = fs::read_to_string(&src).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
    let file: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
    let mut data = serde_json::Map::new();
    for key in ["cliDefault", "presets"] {
        match file.get(key) {
            Some(v) => {
                data.insert(key.to_string(), v.clone());
            }
            None => panic!("{}: no `{key}`", src.display()),
        }
    }
    let out = Path::new(&env::var("OUT_DIR").expect("OUT_DIR")).join("presets.json");
    fs::write(&out, serde_json::to_string_pretty(&data).expect("presets → JSON"))
        .unwrap_or_else(|e| panic!("{}: {e}", out.display()));
}
