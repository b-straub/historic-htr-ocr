// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! THE TEXT CROP: the page's main text block from its text ink — the crop OCR Lite's Auto proposes. On a photo of a
//! sheet against a darker background, or of a tinted paper on a white scanner bed, only the writing on the sheet counts
//! (seals, colour charts, rulers beside it lie outside; see sheet). The text blobs (ink::page::text_blobs) summed per column: the runs of ink, the heaviest run is the main block; every other run on
//! the page joins it (headings, margin columns, notes — however far) unless it is FOREIGN or TINY:
//!   • foreign: a neighbour page (≥ EDGE_CUT of its blobs cut by the image edge — its lines run off the scan) or a
//!     book edge (≥ TALL_INK of its ink in tall thin strokes — the leaves' edges); a foreign run ends the block on its
//!     side
//!   • tiny: fewer than TINY_BLOBS of the main run's blobs (≥ 3) or under TINY_MASS of its ink — a stain, bleed-through,
//!     a speck; skipped
//! The block's ends are trimmed while their ink is mostly tall strokes (a book edge grown into the text). Then the rows
//! inside the block's columns, the same way without the foreign / blob rules (a one-word heading or a date line stays),
//! and PAD pitch around. A 200-page sample over the training corpora: as large as the sealed training crops (median),
//! 88 of 124 pages fit the GT text (≤ 1 line pitch of it cut, ≤ 2 pitches beyond it; paper crop: 57); photographed charters with seals and colour
//! charts: 30 of 44 (paper crop: 0); letters on a white scanner bed with a colour chart and a ruler beside them: 4 of 4.

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
/// a run of drawn lines below or above the text: this share of its ink in tall strokes of ≥ STRAP pitch (the
/// training sample's text runs: ≤ 0.17; a charter's plica with nine seal straps: 0.25) — UNMEASURED beyond them
const STRAP_INK: f64 = 0.2;
const STRAP: f64 = 2.0;
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
/// a framed tinted sheet needs only this share of the text blobs (the rest may be a chart's letters, a ruler's figures)
const SHEET_TINT_MOST: f64 = 0.25;
/// the writing counts from this far inside the sheet's edge (pitch) — its torn edge and the specks along it do not
const SHEET_EDGE: f64 = 0.5;
/// the colour phase: a sheet fills this much of its box, and covers this much of the image
const SHEET_FILL: f64 = 0.9;
const SHEET_MIN: f64 = 0.45;
/// the tint (RGB spread) of a paper or parchment: above TINT_MAX it is a printed colour (a colour chart's patches, a
/// yellow ruler) — counted as background
const TINT_MAX: u8 = 80;

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

/// a sheet candidate: its mask, how much denser the text is on it than off it — by ink and by blobs (a colour chart's
/// patches are dense ink but few blobs) — and the share of the text blobs on it
struct Candidate {
    m: Mask,
    denser: f64,
    blobs_denser: f64,
    most: f64,
}

/// the sheet candidate in one feature image (`feat`, brighter = sheet): the largest bright region (Otsu on it
/// blurred); when that is (almost) the whole image, the split was dark objects against the rest (seals on a grey
/// background): Otsu again on the bright part. None = no such region (it covers the whole image)
fn sheet_in(pg: &Page, text: &[bool], feat: &img::Gray) -> Option<Candidate> {
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
    let (mut b_on, mut b_all) = (0usize, 0usize);
    for c in 1..pg.n {
        if text[c] {
            b_all += 1;
            if m.px[(pg.cy[c] as usize).min(h - 1) * w + (pg.cx[c] as usize).min(w - 1)] {
                b_on += 1;
            }
        }
    }
    let most = b_on as f64 / b_all.max(1) as f64;
    let blobs_denser = (most / share) / ((1.0 - most) / (1.0 - share)).max(f64::MIN_POSITIVE);
    Some(Candidate { m, denser: den_on / den_off.max(f64::MIN_POSITIVE), blobs_denser, most })
}

/// a sheet lying inside the photo: a rectangle (it fills ≥ SHEET_FILL of its box) of ≥ SHEET_MIN of the image, the
/// background on three sides at least (a page may run off one edge of the scan) — not a tinted patch of a paper that
/// fills the scan
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
    let free = [x0 > e, y0 > e, x1 + e < w - 1, y1 + e < h - 1].iter().filter(|&&f| f).count();
    free >= 3 && area as f64 >= SHEET_FILL * ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64 && area as f64 >= SHEET_MIN * (w * h) as f64
}

/// THE SHEET on a photo: the parchment or paper against its background — seals, colour charts and rulers beside it lie
/// outside. First by brightness (a sheet on a darker background): the text denser on it, by ink and by blobs (not a
/// bright background around a dark page, not a colour chart). Then by its tint (`chroma`, the RGB spread at the page's
/// work px, printed colours above TINT_MAX left out): a tinted paper on a white scanner bed or a neutral grey background
/// as bright as it, only a framed one — there a chart's patches and a ruler's ticks off it may be the denser ink
fn sheet(pg: &Page, text: &[bool], chroma: &img::Gray) -> Option<Mask> {
    let by_light = sheet_in(pg, text, &pg.grey).filter(|c| c.denser >= SHEET_DENSER && c.blobs_denser >= SHEET_DENSER);
    let by_tint = || {
        let tint = img::Gray { w: chroma.w, h: chroma.h, px: chroma.px.iter().map(|&v| if v > TINT_MAX { 0 } else { v }).collect() };
        sheet_in(pg, text, &tint).filter(|c| c.most >= SHEET_TINT_MOST && framed(pg, &c.m))
    };
    by_light.or_else(by_tint).map(|c| c.m)
}

/// the text crop of an analysed page → [x0, y0, x1, y1] in the page's image px; None = no text ink
pub fn text_block(pg: &Page, text: &[bool], chroma: &img::Gray) -> Option<[f64; 4]> {
    text_block_in(pg, text, chroma, 0, pg.w())
}

/// the text crop of the page's columns [cx0, cx1) (work px: the blobs centred there count, the block ends there)
fn text_block_in(pg: &Page, text: &[bool], chroma: &img::Gray, cx0: usize, cx1: usize) -> Option<[f64; 4]> {
    let (w, h) = (pg.w(), pg.h());
    let inside: Vec<bool> = (0..pg.n).map(|c| text[c] && pg.cx[c] >= cx0 as f64 && pg.cx[c] < cx1 as f64).collect();
    let text = &inside[..];
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
    x0 = x0.max(cx0);
    x1 = x1.min(cx1);
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
    // a run of lines drawn rather than written ends the block on its side: ≥ STRAP_INK of its ink in tall strokes of
    // ≥ STRAP pitch (a charter's plica with the seal straps' edges; an ascender is shorter)
    let strap = |c: usize| tall(c) && pg.st[c].h as f64 >= STRAP * pg.pitch;
    let rforeign: Vec<bool> = rr
        .iter()
        .map(|q| {
            let cs: Vec<usize> = (1..pg.n).filter(|&c| text[c] && pg.cy[c] >= q.s as f64 && pg.cy[c] < q.e as f64 && pg.cx[c] >= x0 as f64 && pg.cx[c] < x1 as f64).collect();
            let ink: usize = cs.iter().map(|&c| pg.st[c].area).sum();
            let drawn: usize = cs.iter().filter(|&&c| strap(c)).map(|&c| pg.st[c].area).sum();
            ink > 0 && drawn as f64 >= STRAP_INK * ink as f64
        })
        .collect();
    let (y0, y1) = block(&rr, &vec![3; rr.len()], &rforeign);
    let pad = PAD * pg.pitch;
    Some([
        (x0 as f64 - pad).max(cx0 as f64) / pg.f,
        (y0 as f64 - pad).max(0.0) / pg.f,
        (x1 as f64 + pad).min(cx1 as f64) / pg.f,
        (y1 as f64 + pad).min(h as f64) / pg.f,
    ])
}

/// which page of the image the crop keeps: Auto (the main text block), the left or right page of a spread or the leaf
/// beside a neighbour leaf (the side hint: the gutter is searched on the other side), both pages of a spread
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Auto,
    Left,
    Right,
    Double,
}

/// a text crop: [x0, y0, x1, y1] in the image's px and the gutter's x (image px): for Left / Right / Double the one
/// split at — None = no gutter found (the crop is then Auto's); for Auto the spread's when `spread` (its box stays
/// Auto's: the UI offers Left / Right)
pub struct Crop {
    pub b: [f64; 4],
    pub gutter: Option<f64>,
    pub spread: bool,
}

/// Auto's spread hint: the middle gutter (Double's search) with text on both sides — the lighter side at least
/// SPREAD_SIDE of the heavier side's text ink — and the pages' inner margins around it: within SPREAD_MARGIN pitch
/// of it the text profile falls under SPREAD_GAP of its median over the written columns (a charter's fold dips too,
/// but its lines run across it: 0.47 … 1.54 on 13 folded charters; bound spreads 0.01 … 0.23) — UNMEASURED beyond them
const SPREAD_SIDE: f64 = 0.25;
const SPREAD_MARGIN: f64 = 1.0;
const SPREAD_GAP: f64 = 0.35;

/// the middle gutter of a spread with writing on both pages (work px), or None
fn spread_gutter(pg: &Page, text: &[bool]) -> Option<usize> {
    let g = gutter(pg, Side::Double)?;
    let w = pg.w();
    let mut col = vec![0f64; w];
    for i in 0..w * pg.h() {
        let l = pg.lab[i] as usize;
        if l > 0 && text[l] {
            col[i % w] += 1.0;
        }
    }
    let col = img::blur_1d(&col, PROFILE_SIGMA * pg.pitch);
    let peak = col.iter().cloned().fold(0.0, f64::max);
    let m = (SPREAD_MARGIN * pg.pitch) as usize;
    let low = col[g.saturating_sub(m)..(g + m + 1).min(w)].iter().cloned().fold(f64::MAX, f64::min);
    let written: Vec<f64> = col.iter().cloned().filter(|&v| v > RUN_LEVEL * peak).collect();
    let typical = img::median(&written).unwrap_or(peak);
    if low > SPREAD_GAP * typical {
        return None;
    }
    let (mut l, mut r) = (0usize, 0usize);
    for c in 1..pg.n {
        if text[c] {
            if (pg.cx[c] as usize) < g {
                l += pg.st[c].area;
            } else {
                r += pg.st[c].area;
            }
        }
    }
    (l.min(r) as f64 >= SPREAD_SIDE * l.max(r) as f64 && l.max(r) > 0).then_some(g)
}

/// the gutter: the paper's brightness (the column's median grey over the middle rows — writing is too sparse to move
/// it) dips to GUTTER_DIP of both flanks (the brightest within GUTTER_FLANK pitch) — a spine's shadow, the dark line
/// between two leaves. No text-ink condition: the shadow itself is found as ink. The leaf benchmark (198 pages with
/// their side): neighbour-leaf leaks 52 → 22 (notebook leaves 30 → 1), no page cut — UNMEASURED beyond it
const GUTTER_DIP: f64 = 0.85;
const GUTTER_FLANK: f64 = 3.0;
/// the middle rows (a share of the height cut at the top and the bottom: a book's edge, a label strip)
const GUTTER_ROWS: f64 = 0.15;
/// the paper profile's blur (work px)
const GUTTER_BLUR: f64 = 1.0;
/// where a gutter may lie (shares of the width): Left / Right — on the facing side, the kept page at least
/// 1 − GUTTER_SIDE wide; Double — in the middle
const GUTTER_SIDE: f64 = 0.6;
const GUTTER_MID: (f64, f64) = (0.3, 0.7);

/// the gutter's column (work px) for `side`; None = nothing dips enough in its search range
fn gutter(pg: &Page, side: Side) -> Option<usize> {
    let (w, h) = (pg.w(), pg.h());
    let (r0, r1) = ((GUTTER_ROWS * h as f64) as usize, ((1.0 - GUTTER_ROWS) * h as f64) as usize);
    let mut paper = vec![0f64; w];
    let mut v = Vec::with_capacity(r1 - r0);
    for x in 0..w {
        v.clear();
        for y in r0..r1 {
            v.push(pg.grey.px[y * w + x] as f64);
        }
        paper[x] = img::median(&v).expect("rows");
    }
    // the line between two leaves is thin: barely blurred
    let paper = img::blur_1d(&paper, GUTTER_BLUR);
    let (s0, s1) = match side {
        Side::Right => (0.0, GUTTER_SIDE),
        Side::Left => (1.0 - GUTTER_SIDE, 1.0),
        Side::Double => GUTTER_MID,
        Side::Auto => return None,
    };
    // the flanks: from half a pitch to GUTTER_FLANK pitch beside the column, clipped to the image (a neighbour leaf's
    // strip at the edge may be narrower than that) — at least half a pitch of each
    let fl = (GUTTER_FLANK * pg.pitch) as usize;
    let near = ((0.5 * pg.pitch) as usize).max(1);
    let (x0, x1) = (((s0 * w as f64) as usize).max(2 * near), ((s1 * w as f64) as usize).min(w.saturating_sub(2 * near)));
    // the dips: runs of columns below GUTTER_DIP, each at its deepest column
    let mut dips: Vec<(usize, f64)> = Vec::new();
    let mut prev = false;
    for x in x0..x1 {
        let left = paper[x.saturating_sub(fl)..x - near].iter().cloned().fold(0.0, f64::max);
        let right = paper[x + near..(x + fl).min(w)].iter().cloned().fold(0.0, f64::max);
        let dip = paper[x] / left.min(right).max(1.0);
        let on = dip < GUTTER_DIP;
        if on {
            match dips.last_mut() {
                Some(d) if prev => {
                    if dip < d.1 {
                        *d = (x, dip);
                    }
                }
                _ => dips.push((x, dip)),
            }
        }
        prev = on;
    }
    // Left / Right: the dip nearest the kept page (beyond it lie the facing page's outer edge, a neighbour leaf, the
    // background); Double: the deepest in the middle
    match side {
        Side::Left => dips.first().map(|d| d.0),
        Side::Right => dips.last().map(|d| d.0),
        _ => dips.iter().min_by(|a, b| a.1.total_cmp(&b.1)).map(|d| d.0),
    }
}

/// the text crop of an image (`rgb`, w × h — OCR Lite passes the page at its bake scale) → [x0, y0, x1, y1] in its px;
/// Err = no text ink found
pub fn text_crop(rgb: &[u8], w: usize, h: usize) -> Result<[f64; 4], ink::page::NoInk> {
    text_crop_side(rgb, w, h, Side::Auto).map(|c| c.b)
}

/// the text crop for `side`: Left / Right — the text crop of the columns on that side of the gutter; Double — the two
/// sides' crops joined. No gutter: Auto's crop, `gutter` None
pub fn text_crop_side(rgb: &[u8], w: usize, h: usize, side: Side) -> Result<Crop, ink::page::NoInk> {
    let pg = ink::page_from_gray(img::luma_pil_rgb(rgb, w, h))?;
    let text = ink::page::text_blobs(&pg);
    let spread = img::Gray { w, h, px: rgb.chunks_exact(3).map(|p| p.iter().max().expect("rgb") - p.iter().min().expect("rgb")).collect() };
    let chroma = if (pg.w(), pg.h()) == (w, h) { spread } else { img::resize_area(&spread, pg.w(), pg.h()) };
    let none = || ink::page::NoInk("no text ink on the page".into());
    let auto = || text_block(&pg, &text, &chroma).ok_or_else(none);
    if side == Side::Auto {
        let g = spread_gutter(&pg, &text);
        return Ok(Crop { b: auto()?, gutter: g.map(|g| g as f64 / pg.f), spread: g.is_some() });
    }
    let g = match gutter(&pg, side) {
        Some(g) => g,
        None => return Ok(Crop { b: auto()?, gutter: None, spread: false }),
    };
    let left = || text_block_in(&pg, &text, &chroma, 0, g);
    let right = || text_block_in(&pg, &text, &chroma, g, pg.w());
    let b = match side {
        Side::Left => left().ok_or_else(none)?,
        Side::Right => right().ok_or_else(none)?,
        _ => match (left(), right()) {
            (Some(a), Some(b)) => [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])],
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => return Err(none()),
        },
    };
    Ok(Crop { b, gutter: Some(g as f64 / pg.f), spread: false })
}
