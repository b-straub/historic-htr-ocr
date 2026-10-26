// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The ink tools on one image, for parity with the desktop tools' Python: the image (any format the app opens) scaled
// to a long edge of `--longest` px (the bake's scale) and turned grey — written as `--grey <png>` so the reference reads
// the same pixels — then per polygon (a JSON file [[[x, y], …], …] in that grey's px) the wrap and the line measures as
// JSON lines.
//   cargo run --release -p ocr-layout --example inkprobe -- <image> --longest 1540 --grey g.png --polys p.json
use std::path::PathBuf;

use ocr_layout::Pt;
use ocr_layout::ink::{self, img::Gray};

fn arg(args: &[String], name: &str) -> String {
    let i = args.iter().position(|a| a == name).unwrap_or_else(|| panic!("missing {name}"));
    args[i + 1].clone()
}

fn json_poly(p: &[Pt]) -> String {
    let pts: Vec<String> = p.iter().map(|q| format!("[{:.2},{:.2}]", q.x, q.y)).collect();
    format!("[{}]", pts.join(","))
}

fn opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "null".into())
}

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(","))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let longest: f64 = arg(&args, "--longest").parse().expect("--longest: a number");
    let img = ocr_core::imageops::load(&PathBuf::from(&args[1])).expect("image").to_luma8();
    let s = (longest / img.width().max(img.height()) as f64).min(1.0);
    let (w, h) = (((img.width() as f64) * s).round() as u32, ((img.height() as f64) * s).round() as u32);
    let small = image::imageops::resize(&img, w, h, image::imageops::FilterType::Lanczos3);
    small.save(arg(&args, "--grey")).expect("grey png");
    let g = Gray { w: w as usize, h: h as usize, px: small.into_raw() };
    let t0 = std::time::Instant::now();
    let pg = ink::page_from_gray(g).expect("page");
    let text = ink::page::text_blobs(&pg);
    eprintln!("page {} ms, pitch {:.2}, x-height {:.2} (image px)", t0.elapsed().as_millis(), pg.pitch / pg.f, pg.xh / pg.f);
    let polys: Vec<Vec<[f64; 2]>> = {
        let raw = std::fs::read_to_string(arg(&args, "--polys")).expect("polys");
        parse(&raw)
    };
    for p in polys {
        let poly: Vec<Pt> = p.iter().map(|q| Pt::new(q[0], q[1])).collect();
        let wr = ink::wrap(&pg, &text, &poly);
        let m = ink::measure(&pg, std::slice::from_ref(&poly));
        let pieces: Vec<String> = wr.pieces.iter().map(|q| json_poly(q)).collect();
        println!(
            "{{\"pitch\":{:.3},\"blobs\":{},\"pieces\":[{}],\"m\":{{\"pitch\":{},\"core\":{},\"widePitch\":{},\"wideCore\":{},\"extent\":{:.3},\"bandPitch\":{},\"bandCore\":{}}}}}",
            wr.pitch, wr.blobs, pieces.join(","), opt(m.pitch), list(&m.core), opt(m.wide_pitch), list(&m.wide_core), m.extent, opt(m.band_pitch), list(&m.band_core)
        );
    }
}

/// [[[x, y], …], …] — numbers only
fn parse(s: &str) -> Vec<Vec<[f64; 2]>> {
    let nums: Vec<char> = s.chars().collect();
    let mut out: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut depth = 0;
    let mut cur: Vec<f64> = Vec::new();
    let mut tok = String::new();
    for c in nums {
        match c {
            '[' => {
                depth += 1;
                if depth == 2 {
                    out.push(Vec::new());
                }
            }
            ']' | ',' => {
                if !tok.trim().is_empty() {
                    cur.push(tok.trim().parse().expect("a number"));
                }
                tok.clear();
                if c == ']' {
                    if depth == 3 {
                        let last = out.len() - 1;
                        out[last].push([cur[0], cur[1]]);
                        cur.clear();
                    }
                    depth -= 1;
                }
            }
            _ => tok.push(c),
        }
    }
    out
}
