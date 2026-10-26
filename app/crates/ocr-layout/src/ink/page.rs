// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! The page's ink at WORK px (the desktop tools' ink geometry, ported): the grey scaled to a 1600 px long edge, the
//! paper (Otsu between scan background and paper; text at the paper's edge taken in), the ink (NICK, k = −0.1) inside
//! the eroded paper, its 8-connected components, the line PITCH (row-profile autocorrelation in 8 strips), the stroke
//! filter (no rules, no fused frames: their straight edge / rule runs cut out of dropped components), the x-height and
//! the components' ink centres; then the TEXT blobs (no drawings, no faint flat rule dashes).

use super::img::{self, Gray, Stat};
use crate::raster::Mask;

pub const WORK: f64 = 1600.0;
const PITCH_TOL: f64 = 0.12;
const PITCH_NEAR: f64 = 0.8;
const PITCH_HALF: f64 = 0.5;
const PITCH_SHARE: f64 = 0.5;
pub const PITCH_FAR: usize = 160;
const PITCH_VOTES: usize = 3;
pub const PITCH_RANGE: (usize, usize) = (12, 90);
const XH_AREA: f64 = 0.02;
const MIN_COMP: usize = 6;
const INK_MIN: usize = 4;
const VLINE: f64 = 3.0;
const VLINE_EDGE: f64 = 1.0;
const VLINE_W: f64 = 0.08;
const VLINE_IN: f64 = 4.0;
const PAPER_TEXT: f64 = 4.0;
const PAPER_GROW: usize = 6;
const PAPER_DARK: f64 = 0.35;
const HLINE: f64 = 4.0;
const BACKGROUND_MAX: f64 = 120.0;
const FLAT_H: f64 = 2.5;
const FLAT_W: f64 = 4.0;
const FLAT_FAINT: f64 = 0.55;
const DRAW_LEN: f64 = 3.0;
const DRAW_ANY: f64 = 6.0;
const DRAW_STRAIGHT: f64 = 0.06;
const DRAW_THICK: f64 = 1.4;

/// why the ink analysis cannot go on (a blank page, no periodic lines)
#[derive(Debug, Clone, PartialEq)]
pub struct NoInk(pub String);

impl std::fmt::Display for NoInk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NoInk {}

/// Python's round(): half to even
fn round_half_even(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - v.signum() } else { r }
}

/// the line pitch (WORK px) from the row ink profile's autocorrelation in 8 vertical strips: the lag most strips peak at
/// (± PITCH_TOL, up to `far`), the smallest with ≥ PITCH_NEAR of the best support, halved once when its half has ≥
/// PITCH_HALF of its support, refined to the median of those peaks; too few votes → the median of each strip's own
/// first peaks below `top`. Err = no periodic ink rows.
pub fn pitch(ink: &Mask, far: usize, top: usize) -> Result<f64, NoInk> {
    let (w, h) = (ink.w, ink.h);
    let sw = w / 8;
    let (mut peaks, mut own, mut inked): (Vec<Vec<usize>>, Vec<f64>, usize) = (Vec::new(), Vec::new(), 0);
    if sw > 0 {
        let mut x0 = 0;
        while x0 + sw <= w {
            let mut p: Vec<f64> = (0..h).map(|y| (x0..x0 + sw).filter(|&x| ink.px[y * w + x]).count() as f64).collect();
            x0 += sw;
            if p.iter().sum::<f64>() < 50.0 * PITCH_RANGE.0 as f64 {
                continue;
            }
            inked += 1;
            let mean = p.iter().sum::<f64>() / h as f64;
            for v in p.iter_mut() {
                *v -= mean;
            }
            let ac0: f64 = p.iter().map(|v| v * v).sum();
            let norm = if ac0 != 0.0 { ac0 } else { 1.0 };
            let end = far.min(h);
            if end <= PITCH_RANGE.0 {
                continue;
            }
            let seg: Vec<f64> = (PITCH_RANGE.0..end).map(|lag| (0..h - lag).map(|i| p[i] * p[i + lag]).sum::<f64>() / norm).collect();
            let k: Vec<usize> = (1..seg.len().saturating_sub(1)).filter(|&i| seg[i] >= seg[i - 1] && seg[i] >= seg[i + 1] && seg[i] > 0.1).collect();
            if !k.is_empty() {
                peaks.push(k.iter().map(|&i| PITCH_RANGE.0 + i).collect());
            }
            let k2: Vec<usize> = k.into_iter().filter(|&i| PITCH_RANGE.0 + i < top).collect();
            if !k2.is_empty() {
                let first = &k2[..k2.len().min(2)];
                // Python's max(key=…): the first of the highest
                let mut best = first[0];
                for &i in first {
                    if seg[i] > seg[best] {
                        best = i;
                    }
                }
                own.push((PITCH_RANGE.0 + best) as f64);
            }
        }
    }
    let mut lags: Vec<usize> = peaks.iter().flatten().copied().collect();
    lags.sort();
    lags.dedup();
    let support = |c: usize| peaks.iter().filter(|pk| pk.iter().any(|&l| (l as f64 - c as f64).abs() <= PITCH_TOL * c as f64)).count();
    let sup: Vec<usize> = lags.iter().map(|&c| support(c)).collect();
    let best = sup.iter().copied().max().unwrap_or(0);
    if (best as f64) < (PITCH_VOTES as f64).max(PITCH_SHARE * inked as f64) {
        return img::median(&own).ok_or_else(|| NoInk("no periodic ink rows — line pitch not measurable".into()));
    }
    let mut ci = (0..lags.len()).find(|&i| sup[i] as f64 >= PITCH_NEAR * best as f64).ok_or_else(|| NoInk("no lag".into()))?;
    let c = lags[ci] as f64;
    let half: Vec<usize> = (0..lags.len()).filter(|&i| (2.0 * lags[i] as f64 - c).abs() <= PITCH_TOL * c && sup[i] as f64 >= PITCH_HALF * sup[ci] as f64).collect();
    if !half.is_empty() {
        let mut b = half[0];
        for &i in &half {
            if sup[i] > sup[b] {
                b = i;
            }
        }
        ci = b;
    }
    let c = lags[ci] as f64;
    let near: Vec<f64> = peaks.iter().flatten().filter(|&&l| (l as f64 - c).abs() <= PITCH_TOL * c).map(|&l| l as f64).collect();
    img::median(&near).ok_or_else(|| NoInk("no peaks near the pitch".into()))
}

/// the paper: the largest bright component (Otsu between a dark scan background and the paper), its holes filled; all
/// paper when there is no dark background
fn paper_mask(grey: &Gray) -> Mask {
    let (w, h) = (grey.w, grey.h);
    let blur = img::blur(grey, 3.0);
    let t = img::otsu(&blur);
    let bright = Mask { w, h, px: blur.px.iter().map(|&v| v > t).collect() };
    let dark: Vec<f64> = blur.px.iter().filter(|&&v| v <= t).map(|&v| v as f64).collect();
    let all = Mask { w, h, px: vec![true; w * h] };
    if !dark.is_empty() && dark.iter().sum::<f64>() / dark.len() as f64 > BACKGROUND_MAX {
        return all;
    }
    let m = img::close_rect(&bright, 15, 15);
    let (n, lab, st) = img::components(&m);
    if n <= 1 {
        return all;
    }
    let mut k = 1;
    for i in 2..n {
        if st[i].area > st[k].area {
            k = i;
        }
    }
    img::fill_holes(&Mask { w, h, px: lab.iter().map(|&l| l == k as u32).collect() })
}

/// TEXT AT THE PAPER'S EDGE is paper: ink components of text size touching the paper join it, PAPER_GROW rounds; the
/// scan background (one huge component) stays out
fn paper_text(paper: &Mask, ink: &Mask, pitch: f64, grey: &Gray) -> Mask {
    let (w, h) = (paper.w, paper.h);
    let mut out = paper.clone();
    let mut outside = Mask { w, h, px: (0..w * h).map(|i| !paper.px[i] && ink.px[i]).collect() };
    let hl = img::open_rect(&outside, 3.max((HLINE * pitch) as usize), 1);
    let vl = img::open_rect(&outside, 1, 3.max((VLINE * pitch) as usize));
    let lines = img::dilate_rect(&Mask { w, h, px: (0..w * h).map(|i| hl.px[i] || vl.px[i]).collect() }, 3, 3);
    for i in 0..w * h {
        if lines.px[i] {
            outside.px[i] = false;
        }
    }
    let (n, lab, st) = img::components(&outside);
    if n <= 1 {
        return out;
    }
    let mut small: Vec<bool> = st
        .iter()
        .map(|s| {
            (s.w as f64) <= PAPER_TEXT * pitch
                && (s.h as f64) <= PAPER_TEXT * pitch
                && s.y > 0
                && s.y + s.h < h
                && (s.h as f64) >= 0.25 * pitch
                && s.w <= 8 * s.h
        })
        .collect();
    small[0] = false;
    let on: Vec<f64> = (0..w * h).filter(|&i| paper.px[i] && ink.px[i]).map(|i| grey.px[i] as f64).collect();
    let ref_grey = img::median(&on).unwrap_or(0.0);
    let mut light: Vec<bool> = grey.px.iter().map(|&g| g as f64 >= PAPER_DARK * ref_grey).collect();
    let el = (pitch as usize) | 1;
    for _ in 0..PAPER_GROW {
        let grown = img::dilate_rect(&out, 3, 3);
        let mut add = vec![false; n];
        let mut any = false;
        for i in 0..w * h {
            if grown.px[i] && !out.px[i] && small[lab[i] as usize] {
                add[lab[i] as usize] = true;
                any = true;
            }
        }
        if !any {
            break;
        }
        for (c, a) in add.iter().enumerate() {
            if *a {
                small[c] = false;
            }
        }
        let grow = Mask { w, h, px: lab.iter().map(|&l| add[l as usize]).collect() };
        let round = img::dilate_ellipse(&grow, el, el);
        for i in 0..w * h {
            if round.px[i] && light[i] {
                out.px[i] = true;
            }
            if grow.px[i] {
                out.px[i] = true;
                light[i] = true;
            }
        }
    }
    let added = Mask { w, h, px: (0..w * h).map(|i| out.px[i] && !paper.px[i]).collect() };
    if added.px.iter().any(|&b| b) {
        let k2 = 2 * pitch as usize + 1;
        let near = img::dilate_rect(&added, k2, k2);
        let closed = img::close_rect(&out, k2, k2);
        for i in 0..w * h {
            if near.px[i] && closed.px[i] && light[i] {
                out.px[i] = true;
            }
        }
    }
    out
}

/// per component: a STROKE (small: ≤ 2 line heights high, ≤ half the page wide, ≥ INK_MIN px)? background false
fn stroke_filter(st: &[Stat], lh: f64, width: usize) -> Vec<bool> {
    let mut big: Vec<bool> = st.iter().map(|s| s.area >= INK_MIN && (s.h as f64) <= 2.0 * lh && (s.w as f64) <= 0.5 * width as f64).collect();
    big[0] = false;
    big
}

/// the page's ink at WORK px
pub struct Page {
    /// the full image's size
    pub full_w: usize,
    pub full_h: usize,
    /// WORK px per full px
    pub f: f64,
    pub grey: Gray,
    pub paper: Mask,
    pub ink: Mask,
    pub pitch: f64,
    pub n: usize,
    pub lab: Vec<u32>,
    pub st: Vec<Stat>,
    pub keep: Vec<bool>,
    pub xh: f64,
    pub cx: Vec<f64>,
    pub cy: Vec<f64>,
}

impl Page {
    pub fn new(full: &Gray) -> Result<Page, NoInk> {
        let f = (WORK / full.w.max(full.h) as f64).min(1.0);
        let grey = if f < 1.0 {
            img::resize_area(full, round_half_even(full.w as f64 * f) as usize, round_half_even(full.h as f64 * f) as usize)
        } else {
            full.clone()
        };
        let (w, h) = (grey.w, grey.h);
        let paper0 = paper_mask(&grey);
        let mut ink = img::nick_ink(&grey, -0.1);
        let pitch = pitch(&Mask { w, h, px: (0..w * h).map(|i| paper0.px[i] && ink.px[i]).collect() }, PITCH_FAR, PITCH_RANGE.1)?;
        let paper1 = paper_text(&paper0, &ink, pitch, &grey);
        let er = 3.max((0.4 * pitch) as usize) | 1;
        let paper = img::erode_rect(&paper1, er, er);
        for i in 0..w * h {
            if !paper.px[i] {
                ink.px[i] = false;
            }
        }
        let (mut n, mut lab, mut st) = img::components(&ink);
        let keep_of = |st: &[Stat]| -> Vec<bool> { stroke_filter(st, 2.0 * pitch, w).iter().zip(st).map(|(&k, s)| k && s.area >= MIN_COMP).collect() };
        let mut keep = keep_of(&st);
        // EDGE LINES and RULES only in DROPPED components: the straight vertical edge run (near the image's sides; a
        // long one inside) and the long horizontal rule run cut out, the rest judged again
        let lw = 3.max((VLINE_W * pitch) as usize) | 1;
        let mut vl = img::open_rect(&img::dilate_rect(&ink, lw, 1), 1, 3.max((VLINE * pitch) as usize));
        let e = (VLINE_EDGE * pitch) as usize;
        let inner = img::open_rect(&vl, 1, 3.max((VLINE_IN * pitch) as usize));
        for y in 0..h {
            for x in e..w.saturating_sub(e) {
                vl.px[y * w + x] = inner.px[y * w + x];
            }
        }
        let vcut = img::dilate_rect(&vl, lw, 1);
        let hl = img::open_rect(&img::dilate_rect(&ink, 1, lw), 3.max((HLINE * pitch) as usize), 1);
        let hcut = img::dilate_rect(&hl, 1, lw);
        let cut: Vec<bool> = (0..w * h).map(|i| (vcut.px[i] || hcut.px[i]) && lab[i] > 0 && !keep[lab[i] as usize]).collect();
        if cut.iter().any(|&c| c) {
            for i in 0..w * h {
                if cut[i] {
                    ink.px[i] = false;
                }
            }
            (n, lab, st) = img::components(&ink);
            keep = keep_of(&st);
        }
        let hs: Vec<f64> = (0..n).filter(|&c| keep[c] && st[c].area as f64 >= XH_AREA * pitch * pitch).map(|c| st[c].h as f64).collect();
        let xh = img::median(&hs).ok_or_else(|| NoInk("no ink components".into()))?;
        let (mut sx, mut sy) = (vec![0f64; n], vec![0f64; n]);
        for y in 0..h {
            for x in 0..w {
                let l = lab[y * w + x] as usize;
                if keep[l] {
                    sx[l] += x as f64;
                    sy[l] += y as f64;
                }
            }
        }
        let cx = (0..n).map(|c| sx[c] / st[c].area.max(1) as f64).collect();
        let cy = (0..n).map(|c| sy[c] / st[c].area.max(1) as f64).collect();
        Ok(Page { full_w: full.w, full_h: full.h, f, grey, paper, ink, pitch, n, lab, st, keep, xh, cx, cy })
    }

    pub fn w(&self) -> usize {
        self.grey.w
    }

    pub fn h(&self) -> usize {
        self.grey.h
    }
}

/// the page's stroke width (px): 2 × the 75th percentile of the ink's distance to the paper (at least 1)
pub fn stroke_width(ink: &Mask) -> f64 {
    let d = img::dist_l2_3(ink);
    let on: Vec<f64> = (0..d.len()).filter(|&i| ink.px[i]).map(|i| d[i] as f64).collect();
    (2.0 * img::percentile(&on, 75.0).unwrap_or(0.0)).max(1.0)
}

/// a drawing stroke is STRAIGHT (or a gentle arc): its centre line (per column along its long axis) deviates ≤
/// DRAW_STRAIGHT x-heights from a quadratic fit
fn straight(pg: &Page, c: usize) -> bool {
    let s = pg.st[c];
    let w = pg.w();
    let on = |i: usize, j: usize| pg.lab[(s.y + i) * w + s.x + j] == c as u32; // row i, column j of the box
    let (along, across) = if s.h > s.w { (s.h, s.w) } else { (s.w, s.h) };
    let mut cols = Vec::new();
    let mut ys = Vec::new();
    for a in 0..along {
        let hits: Vec<f64> = (0..across).filter(|&b| if s.h > s.w { on(a, b) } else { on(b, a) }).map(|b| b as f64).collect();
        if !hits.is_empty() {
            cols.push(a as f64);
            ys.push(hits.iter().sum::<f64>() / hits.len() as f64);
        }
    }
    if cols.len() < 3 {
        return true;
    }
    // least squares y = a x² + b x + c (normal equations)
    let n = cols.len() as f64;
    let (mut s1, mut s2, mut s3, mut s4, mut t0, mut t1, mut t2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (&x, &y) in cols.iter().zip(&ys) {
        s1 += x;
        s2 += x * x;
        s3 += x * x * x;
        s4 += x * x * x * x;
        t0 += y;
        t1 += x * y;
        t2 += x * x * y;
    }
    let m = [[s4, s3, s2], [s3, s2, s1], [s2, s1, n]];
    let r = [t2, t1, t0];
    let det = |m: [[f64; 3]; 3]| m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let d = det(m);
    if d.abs() < 1e-12 {
        return true;
    }
    let col = |k: usize| {
        let mut q = m;
        for i in 0..3 {
            q[i][k] = r[i];
        }
        det(q) / d
    };
    let (a, b, cc) = (col(0), col(1), col(2));
    let res: Vec<f64> = cols.iter().zip(&ys).map(|(&x, &y)| y - (a * x * x + b * x + cc)).collect();
    let mean = res.iter().sum::<f64>() / n;
    let std = (res.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    std <= DRAW_STRAIGHT * pg.xh
}

/// the page's TEXT blobs (per component): kept strokes minus DRAWINGS (long thin strokes, straight or very long) minus
/// faint FLAT rule dashes
pub fn text_blobs(pg: &Page) -> Vec<bool> {
    let sw = stroke_width(&pg.ink);
    let n = pg.n;
    let long: Vec<f64> = pg.st.iter().map(|s| s.w.max(s.h) as f64).collect();
    let mut drawing: Vec<bool> = (0..n).map(|c| pg.keep[c] && long[c] >= DRAW_LEN * pg.pitch && pg.st[c].area as f64 <= DRAW_THICK * sw * long[c]).collect();
    for c in 0..n {
        if drawing[c] {
            drawing[c] = long[c] >= DRAW_ANY * pg.pitch || straight(pg, c);
        }
    }
    let mut flat: Vec<bool> = (0..n).map(|c| pg.keep[c] && pg.st[c].h as f64 <= FLAT_H * sw && pg.st[c].w as f64 >= FLAT_W * pg.st[c].h as f64).collect();
    if flat.iter().any(|&f| f) {
        let mut sum = vec![0f64; n];
        let mut paper_g = Vec::new();
        for i in 0..pg.lab.len() {
            let l = pg.lab[i] as usize;
            if l > 0 {
                sum[l] += pg.grey.px[i] as f64;
            } else {
                paper_g.push(pg.grey.px[i] as f64);
            }
        }
        let mean: Vec<f64> = (0..n).map(|c| sum[c] / pg.st[c].area.max(1) as f64).collect();
        let body: Vec<f64> = (0..n).filter(|&c| pg.keep[c] && !flat[c] && pg.st[c].area > 50).map(|c| mean[c]).collect();
        if let (Some(ink_g), Some(paper_g)) = (img::median(&body), img::median(&paper_g)) {
            for c in 0..n {
                if flat[c] {
                    flat[c] = mean[c] - ink_g >= FLAT_FAINT * (paper_g - ink_g).max(1.0);
                }
            }
        }
    }
    (0..n).map(|c| pg.keep[c] && !drawing[c] && !flat[c]).collect()
}
