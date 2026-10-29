// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! THE TEXT CROP: the page's main text block from its text ink — the crop OCR Lite's Auto proposes. On a photo of a
//! sheet against a darker background only the writing on the sheet counts (seals, colour charts, rulers beside it lie
//! outside; see sheet). The text blobs (ink::page::text_blobs) summed per column: the runs of ink, the heaviest run is the main block; every other run on
//! the page joins it (headings, margin columns, notes — however far) unless it is FOREIGN or TINY:
//!   • foreign: a neighbour page (≥ EDGE_CUT of its blobs cut by the image edge — its lines run off the scan) or a
//!     book edge (≥ TALL_INK of its ink in tall thin strokes — the leaves' edges); a foreign run ends the block on its
//!     side
//!   • tiny: fewer than TINY_BLOBS of the main run's blobs (≥ 3) or under TINY_MASS of its ink — a stain, bleed-through,
//!     a speck; skipped
//! The block's ends are trimmed while their ink is mostly tall strokes (a book edge grown into the text). Then the rows
//! inside the block's columns, the same way without the foreign / blob rules (a one-word heading or a date line stays),
//! and PAD pitch around. A 200-page sample over the training corpora: as large as the sealed training crops (median),
//! 86 of 124 pages fit the GT text (≤ 1 line pitch of it cut, ≤ 2 pitches beyond it; paper crop: 57); photographed charters with seals and colour
//! charts: 28 of 44 (paper crop: 0).

use crate::ink::{self, Page, img};
use crate::raster::Mask;

/// a run: the profile above RUN_LEVEL of its peak, gaps under RUN_JOIN pitch joined (rows: ROW_JOIN)
const RUN_LEVEL: f64 = 0.03;
const RUN_JOIN: f64 = 0.25;
const ROW_JOIN: f64 = 0.5;
/// the profiles are blurred with PROFILE_SIGMA pitch
const PROFILE_SIGMA: f64 = 0.25;
/// a neighbour page: this share of a run's blobs (and ≥ EDGE_CUT_MIN) touch the image's left / right edge
const EDGE_CUT: f64 = 0.05;
const EDGE_CUT_MIN: usize = 3;
/// a book edge: this share of a run's ink in tall strokes (height ≥ TALL × width)
const TALL_INK: f64 = 0.5;
const TALL: usize = 3;
/// a tiny run: blobs < max(3, main run's blobs / TINY_BLOBS_DIV), or ink < TINY_MASS of the main run's
const TINY_BLOBS_DIV: usize = 20;
const TINY_MASS: f64 = 0.005;
/// the margin around the block (pitch)
const PAD: f64 = 0.5;
/// the sheet: blur sigma (px), closing and opening (pitch), the image share above which there is no background, how
/// much denser the text ink must be on it than off it
const SHEET_BLUR: f64 = 3.0;
const SHEET_CLOSE: f64 = 0.5;
const SHEET_OPEN: f64 = 3.0;
const SHEET_ALL: f64 = 0.97;
const SHEET_DENSER: f64 = 1.5;
/// the writing counts from this far inside the sheet's edge (pitch) — its torn edge and the specks along it do not
const SHEET_EDGE: f64 = 0.5;
/// the colour phase: a sheet fills this much of its box, and covers this much of the image
const SHEET_FILL: f64 = 0.9;
const SHEET_MIN: f64 = 0.45;

/// a run of a profile: [start, end) and its ink
struct Run {
    s: usize,
    e: usize,
    mass: f64,
}

/// the runs of `p` above `level`, gaps under `join` joined
fn runs(p: &[f64], level: f64, join: usize) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    let mut i = 0;
    while i < p.len() {
        if p[i] <= level {
            i += 1;
            continue;
        }
        let s = i;
        let mut mass = 0.0;
        while i < p.len() && p[i] > level {
            mass += p[i];
            i += 1;
        }
        match out.last_mut() {
            Some(l) if s - l.e < join => {
                l.e = i;
                l.mass += mass;
            }
            _ => out.push(Run { s, e: i, mass }),
        }
    }
    out
}

/// the main block of `r`: the heaviest run, grown by its neighbours on each side — a tiny one skipped, a foreign one
/// ends that side → [start, end)
fn block(r: &[Run], blobs: &[usize], foreign: &[bool]) -> (usize, usize) {
    let k = (0..r.len()).max_by(|&a, &b| r[a].mass.total_cmp(&r[b].mass)).expect("block: at least one run");
    let tiny = |j: usize| blobs[j] < 3.max(blobs[k] / TINY_BLOBS_DIV) || r[j].mass < TINY_MASS * r[k].mass;
    let (mut s, mut e) = (r[k].s, r[k].e);
    for j in (0..k).rev() {
        if foreign[j] {
            break;
        }
        if !tiny(j) {
            s = r[j].s;
        }
    }
    for j in k + 1..r.len() {
        if foreign[j] {
            break;
        }
        if !tiny(j) {
            e = r[j].e;
        }
    }
    (s, e)
}

/// the largest region brighter than `t` (closed, holes filled), its thin attachments cut off (a seal's tag or cord, a
/// ruler touching it: opened, the largest piece again) → (mask, its share of the image)
fn bright_region(pg: &Page, blur: &img::Gray, t: u8) -> Option<(Mask, f64)> {
    let (w, h) = (pg.w(), pg.h());
    let bright = Mask { w, h, px: blur.px.iter().map(|&v| v > t).collect() };
    let k = (SHEET_CLOSE * pg.pitch) as usize | 1;
    let (n, lab, st) = img::components(&img::close_rect(&bright, k, k));
    let big = (1..n).max_by_key(|&c| st[c].area)?;
    let filled = img::fill_holes(&Mask { w, h, px: lab.iter().map(|&l| l == big as u32).collect() });
    let o = (SHEET_OPEN * pg.pitch) as usize | 1;
    let (n, lab, st) = img::components(&img::open_rect(&filled, o, o));
    let big = (1..n).max_by_key(|&c| st[c].area)?;
    let share = st[big].area as f64 / (w * h) as f64;
    Some((Mask { w, h, px: lab.iter().map(|&l| l == big as u32).collect() }, share))
}

/// the sheet in one feature image (`feat`, brighter = sheet): the largest bright region (Otsu on it blurred); when that
/// is (almost) the whole image, the split was dark objects against the rest (seals on a grey background): Otsu again on
/// the bright part. None = no such region (it covers the whole image), or the text ink is not denser on it than off it
/// (a bright background, a dark page)
fn sheet_in(pg: &Page, text: &[bool], feat: &img::Gray) -> Option<Mask> {
    let (w, h) = (pg.w(), pg.h());
    let blur = img::blur(feat, SHEET_BLUR);
    let t = img::otsu(&blur);
    let (mut m, mut share) = bright_region(pg, &blur, t)?;
    if share >= SHEET_ALL {
        let px: Vec<u8> = blur.px.iter().copied().filter(|&v| v > t).collect();
        let t2 = img::otsu(&img::Gray { w: px.len(), h: 1, px });
        (m, share) = bright_region(pg, &blur, t2)?;
    }
    if share >= SHEET_ALL {
        return None;
    }
    let (mut on, mut off) = (0usize, 0usize);
    for i in 0..w * h {
        let l = pg.lab[i] as usize;
        if l > 0 && text[l] {
            if m.px[i] {
                on += 1;
            } else {
                off += 1;
            }
        }
    }
    let (den_on, den_off) = (on as f64 / share, off as f64 / (1.0 - share));
    if den_on < SHEET_DENSER * den_off {
        return None;
    }
    Some(m)
}

/// a sheet lying inside the photo: a rectangle (it fills ≥ SHEET_FILL of its box) of ≥ SHEET_MIN of the image, the
/// background on all four sides — not a tinted patch of a paper that fills the scan
fn framed(pg: &Page, m: &Mask) -> bool {
    let (w, h) = (m.w, m.h);
    let (mut x0, mut y0, mut x1, mut y1, mut area) = (w, h, 0, 0, 0usize);
    for i in 0..w * h {
        if m.px[i] {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            area += 1;
        }
    }
    let e = (SHEET_CLOSE * pg.pitch) as usize;
    let inside = x0 > e && y0 > e && x1 + e < w - 1 && y1 + e < h - 1;
    inside && area as f64 >= SHEET_FILL * ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64 && area as f64 >= SHEET_MIN * (w * h) as f64
}

/// THE SHEET on a photo: the parchment or paper against its background — seals, colour charts and rulers beside it lie
/// outside. First by brightness (a sheet on a darker background); then by colour (`chroma`, the RGB spread at the
/// page's work px: a tinted parchment on a neutral grey background as bright as it), only a framed one
fn sheet(pg: &Page, text: &[bool], chroma: &img::Gray) -> Option<Mask> {
    sheet_in(pg, text, &pg.grey).or_else(|| sheet_in(pg, text, chroma).filter(|m| framed(pg, m)))
}

/// the text crop of an analysed page → [x0, y0, x1, y1] in the page's image px; None = no text ink
pub fn text_block(pg: &Page, text: &[bool], chroma: &img::Gray) -> Option<[f64; 4]> {
    let (w, h) = (pg.w(), pg.h());
    // only the writing on the sheet
    let text: Vec<bool> = match sheet(pg, text, chroma) {
        Some(m) => {
            let e = (SHEET_EDGE * pg.pitch) as usize | 1;
            let m = img::erode_rect(&m, e, e);
            let keep: Vec<bool> = (0..pg.n).map(|c| text[c] && m.px[(pg.cy[c] as usize).min(h - 1) * w + (pg.cx[c] as usize).min(w - 1)]).collect();
            keep
        }
        None => text.to_vec(),
    };
    let text = &text[..];
    let is_text = |i: usize| pg.lab[i] > 0 && text[pg.lab[i] as usize];
    let tall = |c: usize| pg.st[c].h >= TALL * pg.st[c].w;
    let sigma = PROFILE_SIGMA * pg.pitch;
    // the columns: all text ink and the tall strokes' share
    let (mut col, mut col_tall) = (vec![0f64; w], vec![0f64; w]);
    for i in 0..w * h {
        if is_text(i) {
            col[i % w] += 1.0;
            if tall(pg.lab[i] as usize) {
                col_tall[i % w] += 1.0;
            }
        }
    }
    let profile = img::blur_1d(&col, sigma);
    let peak = profile.iter().cloned().fold(0.0, f64::max);
    if peak <= 0.0 {
        return None;
    }
    let cr = runs(&profile, RUN_LEVEL * peak, (RUN_JOIN * pg.pitch) as usize);
    let in_run = |q: &Run| -> Vec<usize> { (1..pg.n).filter(|&c| text[c] && pg.cx[c] >= q.s as f64 && pg.cx[c] < q.e as f64).collect() };
    let mut blobs = Vec::new();
    let mut foreign = Vec::new();
    for q in &cr {
        let cs = in_run(q);
        let cut = cs.iter().filter(|&&c| pg.st[c].x <= 1 || pg.st[c].x + pg.st[c].w + 1 >= w).count();
        let ink: usize = cs.iter().map(|&c| pg.st[c].area).sum();
        let tall_ink: usize = cs.iter().filter(|&&c| tall(c)).map(|&c| pg.st[c].area).sum();
        foreign.push((cut >= EDGE_CUT_MIN && cut as f64 >= EDGE_CUT * cs.len() as f64) || (ink > 0 && tall_ink as f64 >= TALL_INK * ink as f64));
        blobs.push(cs.len());
    }
    let (mut x0, mut x1) = block(&cr, &blobs, &foreign);
    // the ends trimmed while their ink is mostly tall strokes
    let (ct, ca) = (img::blur_1d(&col_tall, pg.pitch), img::blur_1d(&col, pg.pitch));
    while x1 > x0 + 1 && ct[x1 - 1] >= TALL_INK * ca[x1 - 1] {
        x1 -= 1;
    }
    while x0 + 1 < x1 && ct[x0] >= TALL_INK * ca[x0] {
        x0 += 1;
    }
    // the rows inside the block's columns
    let mut row = vec![0f64; h];
    for y in 0..h {
        for x in x0..x1 {
            if is_text(y * w + x) {
                row[y] += 1.0;
            }
        }
    }
    let row = img::blur_1d(&row, sigma);
    let rpeak = row.iter().cloned().fold(0.0, f64::max);
    if rpeak <= 0.0 {
        return None;
    }
    let rr = runs(&row, RUN_LEVEL * rpeak, (ROW_JOIN * pg.pitch) as usize);
    let (y0, y1) = block(&rr, &vec![3; rr.len()], &vec![false; rr.len()]);
    let pad = PAD * pg.pitch;
    Some([
        (x0 as f64 - pad).max(0.0) / pg.f,
        (y0 as f64 - pad).max(0.0) / pg.f,
        (x1 as f64 + pad).min(w as f64) / pg.f,
        (y1 as f64 + pad).min(h as f64) / pg.f,
    ])
}

/// the text crop of an image (`rgb`, w × h — OCR Lite passes the page at its bake scale) → [x0, y0, x1, y1] in its px;
/// Err = no text ink found
pub fn text_crop(rgb: &[u8], w: usize, h: usize) -> Result<[f64; 4], ink::page::NoInk> {
    let pg = ink::page_from_gray(img::luma_pil_rgb(rgb, w, h))?;
    let text = ink::page::text_blobs(&pg);
    let spread = img::Gray { w, h, px: rgb.chunks_exact(3).map(|p| p.iter().max().expect("rgb") - p.iter().min().expect("rgb")).collect() };
    let chroma = if (pg.w(), pg.h()) == (w, h) { spread } else { img::resize_area(&spread, pg.w(), pg.h()) };
    text_block(&pg, &text, &chroma).ok_or_else(|| ink::page::NoInk("no text ink on the page".into()))
}
