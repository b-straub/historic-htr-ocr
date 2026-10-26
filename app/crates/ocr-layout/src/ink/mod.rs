// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! The ink tools (the desktop tools' ink geometry, ported): SNAP TO INK — the text blobs a drawn shape holds (≥ half of
//! their ink inside; a sprinkle follows its nearest larger blob) → their hull (closed over word and line gaps, padded,
//! cut to the paper) as polygon pieces — and the LINE MEASURES of a region (pitch, wide pitch for a large hand, the
//! lines' own gaps, each line's core height, the ink's extent) for the per-region scale. All in the image's px (the
//! scaled crop the model reads); the analysis runs at WORK px.

pub mod img;
pub mod page;

use crate::geom::Pt;
use crate::raster::Mask;
use img::Gray;
pub use page::{NoInk, Page};

const HULL_PAD: f64 = 0.12;
const HULL_CLOSE: (f64, f64) = (2.0, 1.3);
const WRAP_IN: f64 = 0.5;
const WRAP_SPRINKLE: f64 = 0.15;
const WRAP_MIN_PIECE: f64 = 1.0;
const LINE_CORE: f64 = 0.5;
const BAND_SIGMA: f64 = 2.0;
const BAND_DIST: f64 = 5.0;

/// polygons (image px) filled into a WORK-px mask (vertices rounded; a pixel is in when its corner point is)
fn fill_polys(pg: &Page, polys: &[Vec<Pt>]) -> Mask {
    let mut m = Mask::new(pg.w(), pg.h());
    for p in polys {
        if p.len() >= 3 {
            // integer vertices; the raster tests pixel centres, so the polygon moves by half a pixel
            let q: Vec<Pt> = p.iter().map(|v| Pt::new((v.x * pg.f).round() + 0.5, (v.y * pg.f).round() + 0.5)).collect();
            m.fill(&q);
        }
    }
    m
}

/// binary CLOSING with a w × h ellipse via exact anisotropic distances, on the input's box + the element
fn smooth_close(m: &Mask, w: usize, h: usize) -> Mask {
    let (ax, ay) = (w as f64 / 2.0, ((h as f64 - 1.0) / 2.0).max(0.5));
    let set: Vec<usize> = (0..m.px.len()).filter(|&i| m.px[i]).collect();
    if set.is_empty() {
        return m.clone();
    }
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for &i in &set {
        let (x, y) = (i % m.w, i / m.w);
        x0 = x0.min(x);
        x1 = x1.max(x);
        y0 = y0.min(y);
        y1 = y1.max(y);
    }
    let pad = ax as usize + 2;
    let (bx0, by0) = (x0.saturating_sub(pad), y0.saturating_sub(pad));
    let (bx1, by1) = ((x1 + 1 + pad).min(m.w), (y1 + 1 + pad).min(m.h));
    let (bw, bh) = (bx1 - bx0, by1 - by0);
    let sub = Mask { w: bw, h: bh, px: (0..bw * bh).map(|i| m.px[(by0 + i / bw) * m.w + bx0 + i % bw]).collect() };
    let dil = |b: &Mask| {
        let d = img::edt_to_set(b, ax / ay);
        Mask { w: b.w, h: b.h, px: d.iter().map(|&v| v <= ax).collect() }
    };
    let d1 = dil(&sub);
    let inv = Mask { w: bw, h: bh, px: d1.px.iter().map(|&b| !b).collect() };
    let d2 = dil(&inv);
    let mut out = Mask::new(m.w, m.h);
    for i in 0..bw * bh {
        out.px[(by0 + i / bw) * m.w + bx0 + i % bw] = !d2.px[i];
    }
    out
}

/// the hull of a blob set: closed with HULL_CLOSE, padded HULL_PAD (an ellipse), cut to the paper
fn blob_hull(pg: &Page, sel: &[bool]) -> Mask {
    let pad = 1.max((HULL_PAD * pg.pitch) as usize);
    let m = Mask { w: pg.w(), h: pg.h(), px: pg.lab.iter().map(|&l| sel[l as usize]).collect() };
    let closed = smooth_close(&m, ((HULL_CLOSE.0 * pg.pitch) as usize) | 1, ((HULL_CLOSE.1 * pg.pitch) as usize) | 1);
    let mut out = img::dilate_ellipse(&closed, 2 * pad + 1, 2 * pad + 1);
    for i in 0..out.px.len() {
        if !pg.paper.px[i] {
            out.px[i] = false;
        }
    }
    out
}

/// a WORK-px mask's outer contours in image px, clamped to the image (Douglas–Peucker 1 px)
fn polygons(pg: &Page, m: &Mask) -> Vec<Vec<Pt>> {
    let (fw, fh) = (pg.full_w as f64, pg.full_h as f64);
    img::outlines(m, 1.0)
        .into_iter()
        .map(|c| c.into_iter().map(|(x, y)| Pt::new((x as f64 / pg.f).clamp(0.0, fw), (y as f64 / pg.f).clamp(0.0, fh))).collect())
        .collect()
}

/// snap to ink: what a drawn shape wraps
#[derive(Debug, Clone, PartialEq)]
pub struct Wrap {
    /// the line pitch (image px)
    pub pitch: f64,
    /// the text blobs taken
    pub blobs: usize,
    /// the hull's pieces (image px; several when the selection falls apart)
    pub pieces: Vec<Vec<Pt>>,
}

/// DRAW AND WRAP: the text blobs with ≥ WRAP_IN of their ink inside the drawn `polygon` (image px); a sprinkle (<
/// WRAP_SPRINKLE pitch²) goes with its nearest larger blob within a line pitch → their hull, specks (< WRAP_MIN_PIECE
/// pitch²) dropped
pub fn wrap(pg: &Page, text: &[bool], polygon: &[Pt]) -> Wrap {
    let m = fill_polys(pg, &[polygon.to_vec()]);
    let n = pg.n;
    let mut inside = vec![0f64; n];
    for i in 0..pg.lab.len() {
        let l = pg.lab[i] as usize;
        if l > 0 && m.px[i] {
            inside[l] += 1.0;
        }
    }
    let p2 = pg.pitch * pg.pitch;
    let area: Vec<f64> = pg.st.iter().map(|s| s.area as f64).collect();
    let small: Vec<bool> = area.iter().map(|&a| a < WRAP_SPRINKLE * p2).collect();
    let big: Vec<usize> = (0..n).filter(|&c| text[c] && !small[c]).collect();
    let mut sel: Vec<bool> = (0..n).map(|c| text[c] && !small[c] && inside[c] / area[c].max(1.0) >= WRAP_IN).collect();
    // a sprinkle follows its nearest larger blob when that lies within a line pitch; a sprinkle with none so near (the
    // separate letters of print — each smaller than a sprinkle) counts by its own share inside (a change to the desktop
    // tools' rule, where every sprinkle followed its nearest blob however far: a printed page wrapped nothing)
    for c in (0..n).filter(|&c| text[c] && small[c] && inside[c] > 0.0) {
        let d = |k: usize| (pg.cx[k] - pg.cx[c]).powi(2) + (pg.cy[k] - pg.cy[c]).powi(2);
        let mut nearest: Option<usize> = None;
        for &k in &big {
            if nearest.is_none_or(|b| d(k) < d(b)) {
                nearest = Some(k);
            }
        }
        sel[c] = match nearest {
            Some(b) if d(b) <= p2 => sel[b],
            _ => inside[c] / area[c].max(1.0) >= WRAP_IN,
        };
    }
    let blobs = sel.iter().filter(|&&s| s).count();
    let pitch = pg.pitch / pg.f;
    if blobs == 0 {
        return Wrap { pitch, blobs, pieces: Vec::new() };
    }
    let mut hull = blob_hull(pg, &sel);
    let (cn, cl, cs) = img::components(&hull);
    for k in 1..cn {
        if (cs[k].area as f64) < WRAP_MIN_PIECE * p2 {
            for i in 0..cl.len() {
                if cl[i] == k as u32 {
                    hull.px[i] = false;
                }
            }
        }
    }
    Wrap { pitch, blobs, pieces: polygons(pg, &hull) }
}

/// the page's ink inside `polys` (image px)
fn unit_ink(pg: &Page, polys: &[Vec<Pt>]) -> Mask {
    let m = fill_polys(pg, polys);
    Mask { w: pg.w(), h: pg.h(), px: (0..m.px.len()).map(|i| m.px[i] && pg.ink.px[i]).collect() }
}

/// the rows of the set pixels
fn rows(m: &Mask) -> Vec<usize> {
    (0..m.px.len()).filter(|&i| m.px[i]).map(|i| i / m.w).collect()
}

/// the row profile of `ys` over n rows, blurred with `sigma`
fn profile(ys: &[usize], n: usize, sigma: f64) -> Vec<f64> {
    let mut p = vec![0f64; n];
    for &y in ys {
        p[y.min(n - 1)] += 1.0;
    }
    img::blur_1d(&p, sigma)
}

/// local maxima of a profile ≥ floor, greedily ≥ dist apart (highest first) → sorted
fn peaks(p: &[f64], dist: f64, floor: f64) -> Vec<usize> {
    let mut cand: Vec<usize> = (1..p.len().saturating_sub(1)).filter(|&i| p[i] >= p[i - 1] && p[i] > p[i + 1] && p[i] >= floor).collect();
    cand.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    let mut out: Vec<usize> = Vec::new();
    for i in cand {
        if out.iter().all(|&q| (i as f64 - q as f64).abs() >= dist) {
            out.push(i);
        }
    }
    out.sort();
    out
}

/// the line pitch of a unit's own ink (image px); None = not measurable. `wide`: searched up to half the unit's height
/// (a large hand)
pub fn unit_pitch(pg: &Page, polys: &[Vec<Pt>], wide: bool) -> Option<f64> {
    let m = fill_polys(pg, polys);
    let (mut far, mut top) = (page::PITCH_FAR, page::PITCH_RANGE.1);
    if wide {
        let ys = rows(&m);
        let (lo, hi) = (ys.iter().min()?, ys.iter().max()?);
        let half = (hi - lo + 1) / 2;
        far = far.max(half);
        top = top.max(half);
    }
    let ink = Mask { w: m.w, h: m.h, px: (0..m.px.len()).map(|i| m.px[i] && pg.ink.px[i]).collect() };
    page::pitch(&ink, far, top).ok().map(|p| p / pg.f)
}

/// each text line's CORE height (image px): the unit's row profile (smoothed with its pitch), peaks ≥ 0.6 pitch apart,
/// each line's band between the minima to its neighbours, its core = the band's rows ≥ LINE_CORE of the prominence
pub fn unit_cores(pg: &Page, polys: &[Vec<Pt>], pitch: Option<f64>) -> Vec<f64> {
    let Some(pitch) = pitch else {
        return Vec::new();
    };
    let ys = rows(&unit_ink(pg, polys));
    if ys.is_empty() {
        return Vec::new();
    }
    let pw = pitch * pg.f;
    let prof = profile(&ys, pg.h(), (0.12 * pw).max(1.0));
    let top = prof.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pk = peaks(&prof, 0.6 * pw, 0.25 * top);
    if pk.is_empty() {
        return Vec::new();
    }
    let argmin = |a: usize, b: usize| {
        let mut k = a;
        for i in a..b {
            if prof[i] < prof[k] {
                k = i;
            }
        }
        k
    };
    let cuts: Vec<usize> = pk.windows(2).map(|w| argmin(w[0], w[1])).collect();
    let mut lo = vec![pk[0].saturating_sub(pw as usize)];
    lo.extend(&cuts);
    let mut hi = cuts.clone();
    hi.push((prof.len() - 1).min(pk[pk.len() - 1] + pw as usize));
    pk.iter()
        .zip(lo.iter().zip(&hi))
        .map(|(&k, (&a0, &b0))| {
            let base = prof[a0].max(prof[b0]);
            let thr = base + LINE_CORE * (prof[k] - base);
            let (mut a, mut b) = (k, k);
            while a > a0 && prof[a - 1] >= thr {
                a -= 1;
            }
            while b < b0 && prof[b + 1] >= thr {
                b += 1;
            }
            (b - a + 1) as f64 / pg.f
        })
        .collect()
}

/// the line pitch (image px) of a SMALL unit from its lines directly: the median gap between the row-profile peaks;
/// None = fewer than two lines
pub fn unit_band_pitch(pg: &Page, polys: &[Vec<Pt>]) -> Option<f64> {
    let ys = rows(&unit_ink(pg, polys));
    let (lo, hi) = (*ys.iter().min()?, *ys.iter().max()?);
    let sw = page::stroke_width(&pg.ink);
    let rel: Vec<usize> = ys.iter().map(|&y| y - lo).collect();
    let prof = profile(&rel, hi - lo + 1, BAND_SIGMA * sw);
    let top = prof.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pk = peaks(&prof, BAND_DIST * sw, 0.25 * top);
    if pk.len() < 2 {
        return None;
    }
    let gaps: Vec<f64> = pk.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    img::median(&gaps).map(|g| g / pg.f)
}

/// the height (image px) the unit's ink spans, its 1–99 % rows; 0 = no ink
pub fn unit_extent(pg: &Page, polys: &[Vec<Pt>]) -> f64 {
    let ys: Vec<f64> = rows(&unit_ink(pg, polys)).into_iter().map(|y| y as f64).collect();
    match (img::percentile(&ys, 99.0), img::percentile(&ys, 1.0)) {
        (Some(a), Some(b)) => (a - b) / pg.f,
        _ => 0.0,
    }
}

/// a region's line measures (image px) — what the per-region scale decides by
#[derive(Debug, Clone, PartialEq)]
pub struct Measures {
    pub pitch: Option<f64>,
    pub core: Vec<f64>,
    pub wide_pitch: Option<f64>,
    pub wide_core: Vec<f64>,
    pub extent: f64,
    /// the lines' own gaps — only where neither pitch measures
    pub band_pitch: Option<f64>,
    pub band_core: Vec<f64>,
}

pub fn measure(pg: &Page, polys: &[Vec<Pt>]) -> Measures {
    let pitch = unit_pitch(pg, polys, false);
    let wide_pitch = unit_pitch(pg, polys, true);
    let band_pitch = if pitch.is_none() && wide_pitch.is_none() { unit_band_pitch(pg, polys) } else { None };
    Measures {
        pitch,
        core: unit_cores(pg, polys, pitch),
        wide_pitch,
        wide_core: unit_cores(pg, polys, wide_pitch),
        extent: unit_extent(pg, polys),
        band_pitch,
        band_core: unit_cores(pg, polys, band_pitch),
    }
}

/// the page analysed from RGBA pixels (grey as PIL converts it)
pub fn page_from_rgba(rgba: &[u8], w: usize, h: usize) -> Result<Page, NoInk> {
    Page::new(&img::luma_pil(rgba, w, h))
}

/// the page analysed from grey pixels
pub fn page_from_gray(g: Gray) -> Result<Page, NoInk> {
    Page::new(&g)
}
