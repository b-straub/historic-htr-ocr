// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The built-in auto crop: find the paper on a scan → a crop rect. A line-by-line port of a Swift / Core Graphics
// implementation (`cropToPaperWithRect` → `cropDecision(keepSpread:pageSide:singleLeaf:)` →
// `cropSafetyMargin(facing: facingSides(…))`, with `normalizeGreyscale` (the analysis) and `resizedPixels` (the
// ≤ 1100 px analysis downscale)): thresholds, loop bounds, integer / float rounding (Swift `Int(x)` truncates = Rust
// `as`, Swift `.rounded()` / `round` = Rust `round`). Not ported: a crop plan from stored layout data (ink-union clamp,
// fold-line cut — the fold cut needs a known page side and is not on this path), void painting, the debug overlay, and
// the verification-only metrics (edgeInk / outInk / band / island: the decision never reads them).
//
// Same pixels in, same rect out: on 154 scans pre-shrunk to ≤ 1100 px (no analysis resize) and colour-matched to sRGB,
// all 154 rects are identical to the Swift implementation's. Exact parity is impossible in three places — all are pixel
// differences, which the thresholds turn into edge shifts of a few analysis px (Swift against itself moves as much
// when the same scan is decoded slightly differently):
//   • the downscale: CoreImage CILanczosScaleTransform (clampedToExtent, rendered by a default CIContext = resampled
//     in LINEAR sRGB, written back as 8-bit sRGB) vs `image` Lanczos3 on linear f32 here — kernel taps, edge handling
//     (edge replication vs a truncated, renormalised kernel) and rounding differ by ±1–2 grey levels per pixel.
//     Resampling in gamma space instead was measured clearly worse (shifts up to 617 px vs 13)
//   • colour management: Swift draws every image into a DeviceRGB premultipliedLast CGContext (makeRGBAContext). Opaque
//     scans → premultiplying is a no-op; sRGB, untagged and Gray Gamma 2.2 scans land unchanged (measured); a scan with
//     another embedded ICC profile (scanner profiles, Adobe RGB) is colour-matched by Core Graphics —
//     here the decoded bytes are used as is (`image` ignores ICC profiles)
//   • decoding: the caller's JPEG decoder (`image` = zune-jpeg) vs Apple's differ by ±1 level in places. NSImage may
//     also apply an EXIF orientation; `image` does not (the caller must).

use anyhow::Result;
use image::{ImageBuffer, Rgb, RgbImage, imageops};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum PageSide {
    Left,
    Right,
}

/// The page's mode hints (picked by hand in the auto crop menu). Swift callers pass keepSpread = is_spread ||
/// single_leaf; single_leaf alone does not imply keep_spread here either.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AutoCropOptions {
    pub keep_spread: bool,
    pub page_side: Option<PageSide>,
    pub single_leaf: bool,
}

/// The crop in the input image's pixels (same fields as imageops::Op::Crop).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CropRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// cropToPaperWithRect: the decision's rect (original px) + the outward safety margin; the whole image when there is
/// no decision or it is narrower than 2 px.
pub fn auto_crop(img: &RgbImage, opts: &AutoCropOptions) -> Result<CropRect> {
    let (iw, ih) = (img.width() as i64, img.height() as i64);
    let whole = CropRect { x: 0, y: 0, w: img.width(), h: img.height() };
    let Some(d) = crop_decision(img, opts) else {
        return Ok(whole);
    };
    let r = d.rect();
    if r.2 - r.0 < 2 || r.3 - r.1 < 2 {
        return Ok(whole);
    }
    // CGRect.intersection(full)
    let r = (r.0.max(0), r.1.max(0), r.2.min(iw), r.3.min(ih));
    if r.2 <= r.0 || r.3 <= r.1 {
        anyhow::bail!("crop decision {r:?} lies outside the {iw}×{ih} image");
    }
    let m = crop_safety_margin(img, r, facing_sides(opts.page_side, &d));
    Ok(CropRect { x: m.0 as u32, y: m.1 as u32, w: (m.2 - m.0) as u32, h: (m.3 - m.1) as u32 })
}

/// ITU-R BT.601 luma, integer (ImageFilters.swift lumaOf)
#[inline(always)]
fn luma_of(p: &Rgb<u8>) -> u8 {
    ((p.0[0] as u32 * 77 + p.0[1] as u32 * 150 + p.0[2] as u32 * 29) >> 8).min(255) as u8
}

const DOC_BG_THR: f64 = 76.0;
const FOLD_DIP_MAX: f64 = 0.85;
const FOLD_BRIGHT_MIN: f64 = 0.92;

/// What the crop needs from one cropDecision run (CropDecision, without the overlay profiles / verification metrics).
struct Decision {
    scale: f64,
    x_lo: i64,
    x_hi: i64,
    y_lo: i64,
    y_hi: i64,
    gaps: Vec<(i64, i64)>,
    chosen_segment_idx: i64,
}

impl Decision {
    /// CropDecision.rect → (x0, y0, x1, y1) in original px: floor of the origin, ceil of the size
    fn rect(&self) -> (i64, i64, i64, i64) {
        let x = (self.x_lo as f64 / self.scale).floor() as i64;
        let y = (self.y_lo as f64 / self.scale).floor() as i64;
        let w = ((self.x_hi - self.x_lo + 1) as f64 / self.scale).ceil() as i64;
        let h = ((self.y_hi - self.y_lo + 1) as f64 / self.scale).ceil() as i64;
        (x, y, x + w, y + h)
    }
}

/// facingSides(pageSide:decision:boxed: false) → [L, R]
fn facing_sides(side: Option<PageSide>, d: &Decision) -> [bool; 2] {
    match side {
        Some(PageSide::Left) => return [false, true],
        Some(PageSide::Right) => return [true, false],
        None => {}
    }
    if d.gaps.is_empty() || d.chosen_segment_idx < 0 {
        return [false, false];
    }
    [d.chosen_segment_idx > 0, d.chosen_segment_idx < d.gaps.len() as i64]
}

/// NSImage.furnitureTrim — edge furniture (cm ruler, label) walking from `edge` toward `limit`
#[allow(clippy::too_many_arguments)]
fn furniture_trim(p: &[f64], mean: &[f64], edge: i64, limit: i64, step: i64, dark_thr: f64, mean_max: f64,
                  gap_min: i64, gap_max: i64, max_strip: i64) -> i64 {
    let mut i = edge;
    let mut first = true;
    let mut cut = edge;
    while (i - limit) * step < 0 && (i - edge).abs() <= max_strip {
        if p[i as usize] > dark_thr {
            i += step;
            continue;
        }
        let mut n = 0i64;
        let mut sum = 0.0;
        while (i - limit) * step < 0 && p[i as usize] <= dark_thr {
            sum += mean[i as usize];
            i += step;
            n += 1;
        }
        if n >= gap_min && (first || sum / n as f64 <= mean_max) {
            if !(n <= gap_max && (i - limit) * step < 0) {
                return cut;
            }
            cut = i;
        }
        first = false;
    }
    cut
}

/// NSImage.resizedPixels for the analysis downscale — see the header: Lanczos in linear light like a default CIContext.
fn resized(img: &RgbImage, w: u32, h: u32) -> RgbImage {
    let lin: Vec<f32> = (0..256).map(|v| srgb_to_linear(v as f32 / 255.0)).collect();
    let src: ImageBuffer<Rgb<f32>, Vec<f32>> =
        ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
            let p = img.get_pixel(x, y).0;
            Rgb([lin[p[0] as usize], lin[p[1] as usize], lin[p[2] as usize]])
        });
    let out = imageops::resize(&src, w, h, imageops::FilterType::Lanczos3);
    ImageBuffer::from_fn(w, h, |x, y| {
        let p = out.get_pixel(x, y).0;
        let enc = |v: f32| (linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0).round().clamp(0.0, 255.0) as u8;
        Rgb([enc(p[0]), enc(p[1]), enc(p[2])])
    })
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// GrayscaleNormalizer.swift normalizeGreyscale(tp: 245, tk: 15) on a luma plane: NICK mask → centre-weighted
/// medians of paper / ink → linear remap ink → 15, paper → 245. Returns the remapped grey (R = G = B in Swift).
fn normalize_greyscale(gray: &[u8], w: usize, h: usize) -> Vec<u8> {
    let (tp, tk) = (245.0f64, 15.0f64);
    let n = w * h;
    let iw = w + 1;
    let mut integ = vec![0.0f64; iw * (h + 1)];
    let mut integ2 = vec![0.0f64; iw * (h + 1)];
    for y in 0..h {
        let (mut rs, mut rs2) = (0.0f64, 0.0f64);
        let (in_row, out_row, prev_row) = (y * w, (y + 1) * iw, y * iw);
        for x in 0..w {
            let v = gray[in_row + x] as f64;
            rs += v;
            rs2 += v * v;
            integ[out_row + x + 1] = integ[prev_row + x + 1] + rs;
            integ2[out_row + x + 1] = integ2[prev_row + x + 1] + rs2;
        }
    }
    let win_r = 10.max(40.min(w.min(h) / 12)) as i64;
    let nick_k = -0.10;
    let (wi, hi_) = (w as i64, h as i64);
    let mut is_ink = vec![false; n];
    for y in 0..hi_ {
        let (y0, y1) = (0.max(y - win_r), (hi_ - 1).min(y + win_r));
        for x in 0..wi {
            let (x0, x1) = (0.max(x - win_r), (wi - 1).min(x + win_r));
            let area = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
            let (y0u, y1u, x0u, x1u) = (y0 as usize, y1 as usize, x0 as usize, x1 as usize);
            let (tl, tr) = (y0u * iw + x0u, y0u * iw + (x1u + 1));
            let (bl, br) = ((y1u + 1) * iw + x0u, (y1u + 1) * iw + (x1u + 1));
            let s = integ[br] - integ[tr] - integ[bl] + integ[tl];
            let s2 = integ2[br] - integ2[tr] - integ2[bl] + integ2[tl];
            let m = s / area;
            let variance = 0f64.max(s2 / area - m * m);
            let term = (variance + m * m * (area - 1.0) / area).sqrt();
            let idx = y as usize * w + x as usize;
            is_ink[idx] = (gray[idx] as f64) < m + nick_k * term;
        }
    }
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let mut ink_h = [0.0f64; 256];
    let mut paper_h = [0.0f64; 256];
    for y in 0..h {
        let dy = (y as f64 - cy) / 1f64.max(cy);
        for x in 0..w {
            let dx = (x as f64 - cx) / 1f64.max(cx);
            let wgt = 0f64.max(1.0 - (dx * dx + dy * dy).sqrt());
            let p = y * w + x;
            if is_ink[p] {
                ink_h[gray[p] as usize] += wgt;
            } else {
                paper_h[gray[p] as usize] += wgt;
            }
        }
    }
    let ink_level = weighted_median(&cut_dark_spike(&ink_h), tk);
    let paper_level = weighted_median(&paper_h, tp);
    let denom = 1f64.max(paper_level - ink_level);
    gray.iter()
        .map(|&g| {
            let v = (g as f64 - ink_level) / denom * (tp - tk) + tk;
            v.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// GrayscaleNormalizer cutDarkSpike, as Swift EXECUTES it: `(0..<peak).min(by: { hist[$0] > hist[$1] })` is the
/// FIRST MAXIMUM below the peak (the comment there says argmin — see the report), the peak = the first maximum.
fn cut_dark_spike(hist: &[f64; 256]) -> [f64; 256] {
    let mut peak = 0usize;
    for i in 1..256 {
        if hist[i] > hist[peak] {
            peak = i;
        }
    }
    if peak == 0 {
        return *hist;
    }
    let mut valley = 0usize;
    for i in 1..peak {
        if hist[i] > hist[valley] {
            valley = i;
        }
    }
    let prefix: f64 = hist[0..=valley].iter().sum();
    if !(prefix > 0.0 && hist[valley] < 0.3 * hist[peak]) {
        return *hist;
    }
    let mut r = *hist;
    for v in r.iter_mut().take(valley + 1) {
        *v = 0.0;
    }
    r
}

/// first bin where the cumulative weight reaches half the total; `default` for an empty histogram
fn weighted_median(hist: &[f64; 256], default: f64) -> f64 {
    let total: f64 = hist.iter().sum();
    if total <= 0.0 {
        return default;
    }
    let half = total / 2.0;
    let mut cum = 0.0;
    for (i, w) in hist.iter().enumerate() {
        cum += w;
        if cum >= half {
            return i as f64;
        }
    }
    255.0
}

/// NSImage.cropDecision — the analysis on the ≤ 1100 px image; None = no decision (no ink at all, image < 2 px).
fn crop_decision(img: &RgbImage, opts: &AutoCropOptions) -> Option<Decision> {
    let keep_spread = opts.keep_spread;
    let single_leaf = opts.single_leaf;
    let side = opts.page_side;
    let (w0, h0) = (img.width() as f64, img.height() as f64);
    let max_dim = w0.max(h0);
    let target = 1100.0;
    let f = if max_dim > target { target / max_dim } else { 1.0 };
    let scaled;
    let base: &RgbImage = if f < 1.0 {
        let (sw, sh) = (1f64.max((w0 * f).round()), 1f64.max((h0 * f).round()));
        scaled = resized(img, sw as u32, sh as u32);
        &scaled
    } else {
        img
    };
    let (w, h) = (base.width() as usize, base.height() as usize);
    if w <= 1 || h <= 1 {
        return None;
    }
    let n = w * h;
    let (wi, hi) = (w as i64, h as i64);

    // ── 1. luma of the original (analysis size) and of the normalised grey
    let orig_gray: Vec<u8> = base.pixels().map(luma_of).collect();
    let gray = normalize_greyscale(&orig_gray, w, h);

    // ── 2. column brightness up front
    let h_d = h as f64;
    let mut paper_col = vec![0.0f64; w];
    let mut mean_col = vec![0.0f64; w];
    for x in 0..w {
        let (mut cnt, mut sum) = (0i64, 0i64);
        for y in 0..h {
            let v = orig_gray[y * w + x] as i64;
            if v > 140 {
                cnt += 1;
            }
            sum += v;
        }
        paper_col[x] = cnt as f64 / h_d;
        mean_col[x] = sum as f64 / h_d;
    }
    let bg_thr = DOC_BG_THR;

    // ── step 1a: wide uniformly-dark bands = mount
    let min_mount_w = wi / 5;
    let mount_paper_max = 0.05;
    let mut is_bg_col = vec![false; w];
    let mut band_start = 0usize;
    while band_start < w {
        if mean_col[band_start] <= bg_thr {
            let mut band_end = band_start;
            while band_end < w && mean_col[band_end] <= bg_thr {
                band_end += 1;
            }
            if (band_end - band_start) as i64 >= min_mount_w {
                let ps: f64 = paper_col[band_start..band_end].iter().sum();
                if ps / ((band_end - band_start) as f64) < mount_paper_max {
                    for c in is_bg_col.iter_mut().take(band_end).skip(band_start) {
                        *c = true;
                    }
                }
            }
            band_start = band_end;
        } else {
            band_start += 1;
        }
    }

    // ── 3. ink mask
    let mut is_ink: Vec<bool> = (0..n).map(|p| gray[p] < 128 && !is_bg_col[p % w]).collect();

    // ── 4. remove horizontal ruled lines
    let min_line_len = 20i64.max((w.max(h) as f64 * 0.22) as i64);
    for y in 0..h {
        let row = &mut is_ink[y * w..(y + 1) * w];
        let (mut max_run, mut run) = (0i64, 0i64);
        for &v in row.iter() {
            if v {
                run += 1;
                if run > max_run {
                    max_run = run;
                }
            } else {
                run = 0;
            }
        }
        if max_run >= min_line_len {
            row.fill(false);
        }
    }

    // ── 4. column ink density
    let mut col_ink = vec![0.0f64; w];
    for p in 0..n {
        if is_ink[p] {
            col_ink[p % w] += 1.0;
        }
    }
    for v in col_ink.iter_mut() {
        *v /= h_d;
    }

    // ── 5. box-smooth
    let sigma = 2i64.max(wi / 80);
    let col = box_smooth(&col_ink, sigma);

    // ── 6. ink extent within the main (non-mount) segment
    let col_max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !(col_max > 0.0) {
        return None;
    }
    let lo_thr = col_max * 0.02;
    let (mut seg_start, mut best_lo, mut best_hi, mut best_ink) = (0usize, -1i64, -1i64, -1.0f64);
    while seg_start < w {
        if is_bg_col[seg_start] {
            seg_start += 1;
            continue;
        }
        let mut seg_end = seg_start;
        while seg_end < w && !is_bg_col[seg_end] {
            seg_end += 1;
        }
        let (mut lo, mut hi_, mut sum) = (-1i64, -1i64, 0.0);
        for x in seg_start..seg_end {
            if col[x] > lo_thr {
                if lo < 0 {
                    lo = x as i64;
                }
                hi_ = x as i64;
                sum += col[x];
            }
        }
        if lo >= 0 && sum > best_ink {
            best_ink = sum;
            best_lo = lo;
            best_hi = hi_;
        }
        seg_start = seg_end;
    }
    if best_lo < 0 {
        return None;
    }
    let (mut x_lo, mut x_hi) = (best_lo, best_hi);
    let (mut original_x_lo, mut original_x_hi) = (best_lo, best_hi);

    let mut p_lo = mean_col.iter().position(|&v| v > bg_thr).map_or(0, |i| i as i64);
    let mut p_hi = mean_col.iter().rposition(|&v| v > bg_thr).map_or(wi - 1, |i| i as i64);
    let ruler_gap_w = 20i64.max(wi / 80);
    let max_strip_w = wi / 8;
    let black_thr = 0.15;
    p_hi = furniture_trim(&paper_col, &mean_col, p_hi, p_lo, -1, black_thr, bg_thr, ruler_gap_w, wi / 4, max_strip_w);
    p_lo = furniture_trim(&paper_col, &mean_col, p_lo, p_hi, 1, black_thr, bg_thr, ruler_gap_w, wi / 4, max_strip_w);
    x_lo = x_lo.max(p_lo);
    x_hi = x_hi.min(p_hi);

    let near_blank = ((x_hi - x_lo) as f64) < 0.25 * (p_hi - p_lo) as f64;
    if near_blank {
        x_lo = p_lo;
        x_hi = p_hi;
        original_x_lo = p_lo;
        original_x_hi = p_hi;
    }

    let mut leaf_without_mount = false;
    let mut leaf_rows: Option<(i64, i64)> = None;
    if single_leaf {
        let mut hist = [0i64; 256];
        let mut i = 0usize;
        while i < n {
            hist[orig_gray[i] as usize] += 1;
            i += 7;
        }
        let total: i64 = hist.iter().sum();
        let (mut acc, mut p95) = (0i64, 255i64);
        for v in 0..256 {
            acc += hist[v];
            if acc >= total * 95 / 100 {
                p95 = v as i64;
                break;
            }
        }
        let thr_v = 60i64.max(230i64.min((0.75 * p95 as f64) as i64)) as u8;
        let mut frame = [0i64; 256];
        let (fw, fh) = (2usize.max(w * 3 / 100), 2usize.max(h * 3 / 100));
        for y in 0..h {
            let row = y * w;
            if y < fh || y + fh >= h {
                for x in 0..w {
                    frame[orig_gray[row + x] as usize] += 1;
                }
            } else {
                for x in 0..fw {
                    frame[orig_gray[row + x] as usize] += 1;
                }
                for x in (w - fw)..w {
                    frame[orig_gray[row + x] as usize] += 1;
                }
            }
        }
        let frame_total: i64 = frame.iter().sum();
        let (mut facc, mut mount_p10) = (0i64, 255i64);
        for v in 0..256 {
            facc += frame[v];
            if facc >= frame_total / 10 {
                mount_p10 = v as i64;
                break;
            }
        }
        let has_mount = mount_p10 <= 60;
        let card_v = 255i64.min(mount_p10 + 30) as u8;
        let is_card = |bright: i64, card: i64, of: i64| -> bool {
            bright as f64 / of as f64 >= 0.05 || (has_mount && card as f64 / of as f64 >= 0.5)
        };
        let (mut rows_lo, mut rows_hi) = (-1i64, -1i64);
        let mut row_runs: Vec<(i64, i64)> = Vec::new();
        {
            let (mut run_start, mut best_row_len) = (-1i64, 0i64);
            for y in 0..hi {
                let (mut cnt, mut card) = (0i64, 0i64);
                let row = y as usize * w;
                for &v in &orig_gray[row..row + w] {
                    if v > thr_v {
                        cnt += 1;
                    }
                    if v > card_v {
                        card += 1;
                    }
                }
                if is_card(cnt, card, wi) {
                    if run_start < 0 {
                        run_start = y;
                    }
                    if y - run_start + 1 > best_row_len {
                        best_row_len = y - run_start + 1;
                        rows_lo = run_start;
                        rows_hi = y;
                    }
                } else {
                    if run_start >= 0 {
                        row_runs.push((run_start, y - 1));
                    }
                    run_start = -1;
                }
            }
            if run_start >= 0 {
                row_runs.push((run_start, hi - 1));
            }
        }
        let wide_rows: Vec<(i64, i64)> = row_runs.iter().cloned().filter(|r| r.1 - r.0 + 1 >= hi / 10).collect();
        if wide_rows.len() > 1 {
            rows_lo = wide_rows.iter().map(|r| r.0).min().unwrap_or(rows_lo);
            rows_hi = wide_rows.iter().map(|r| r.1).max().unwrap_or(rows_hi);
        }
        if rows_lo >= 0 && rows_hi > rows_lo {
            leaf_rows = Some((rows_lo, rows_hi));
        }
        let (mut b_lo, mut b_hi, mut best_len) = (p_lo, p_hi, 0i64);
        let mut runs: Vec<(i64, i64)> = Vec::new();
        if rows_lo >= 0 && rows_hi > rows_lo {
            let mut run_lo = -1i64;
            for x in 0..wi {
                let (mut cnt, mut card) = (0i64, 0i64);
                for y in rows_lo..=rows_hi {
                    let v = orig_gray[y as usize * w + x as usize];
                    if v > thr_v {
                        cnt += 1;
                    }
                    if v > card_v {
                        card += 1;
                    }
                }
                if is_card(cnt, card, rows_hi - rows_lo + 1) {
                    if run_lo < 0 {
                        run_lo = x;
                    }
                    if x - run_lo + 1 > best_len {
                        best_len = x - run_lo + 1;
                        b_lo = run_lo;
                        b_hi = x;
                    }
                } else {
                    if run_lo >= 0 {
                        runs.push((run_lo, x - 1));
                    }
                    run_lo = -1;
                }
            }
            if run_lo >= 0 {
                runs.push((run_lo, wi - 1));
            }
        }
        let wide: Vec<(i64, i64)> = runs.iter().cloned().filter(|r| r.1 - r.0 + 1 >= wi / 10).collect();
        if wide.len() > 1 {
            b_lo = wide.iter().map(|r| r.0).min().unwrap_or(b_lo);
            b_hi = wide.iter().map(|r| r.1).max().unwrap_or(b_hi);
        }
        if !has_mount {
            leaf_without_mount = true;
            b_lo = 0;
            b_hi = wi - 1;
        }
        x_lo = b_lo;
        x_hi = b_hi;
        original_x_lo = x_lo;
        original_x_hi = x_hi;
        p_lo = b_lo;
        p_hi = b_hi;
    }

    // ── edge-sliver trim (single leaf)
    if single_leaf && !near_blank {
        let span = p_hi - p_lo;
        if span > 40 {
            let mut vals: Vec<f64> = (p_lo..=p_hi).map(|x| mean_col[x as usize]).collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            let plateau = vals[vals.len() / 2];
            let fold_thr = plateau * FOLD_DIP_MAX;
            let bright_thr = plateau * FOLD_BRIGHT_MIN;
            let max_sliver_w = 8i64.max(span / 8);
            let r_lo = p_hi - max_sliver_w;
            if r_lo > p_lo {
                let mut xf = r_lo;
                for x in r_lo..=p_hi {
                    if mean_col[x as usize] < mean_col[xf as usize] {
                        xf = x;
                    }
                }
                let sliver_w = p_hi - xf;
                if mean_col[xf as usize] < fold_thr && sliver_w >= 2 && sliver_w <= max_sliver_w {
                    let strip = (xf + 1..=p_hi).fold(0.0, |a, x| a + mean_col[x as usize]) / sliver_w as f64;
                    if strip > bright_thr {
                        p_hi = xf;
                    }
                }
            }
            let l_hi = p_lo + max_sliver_w;
            if l_hi < p_hi {
                let mut xf = l_hi;
                for x in p_lo..=l_hi {
                    if mean_col[x as usize] < mean_col[xf as usize] {
                        xf = x;
                    }
                }
                let sliver_w = xf - p_lo;
                if mean_col[xf as usize] < fold_thr && sliver_w >= 2 && sliver_w <= max_sliver_w {
                    let strip = (p_lo..xf).fold(0.0, |a, x| a + mean_col[x as usize]) / sliver_w as f64;
                    if strip > bright_thr {
                        p_lo = xf;
                    }
                }
            }
            x_lo = x_lo.max(p_lo);
            x_hi = x_hi.min(p_hi);
        }
    }

    let mut gaps: Vec<(i64, i64)> = Vec::new();
    // the dip column of the tier that made a gap (spine / valley trough / fold) — as the Swift implementation's gapCentre
    let mut gap_centre = -1i64;
    let mut col_bright_sm: Vec<f64> = Vec::new();
    if !keep_spread && !near_blank {
        // ── 7. Falz-Finder: column brightness of the NORMALISED grey
        let mut col_bright = vec![0.0f64; w];
        for y in 0..h {
            for x in 0..w {
                col_bright[x] += gray[y * w + x] as f64;
            }
        }
        for v in col_bright.iter_mut() {
            *v /= h_d;
        }
        let b_sigma = 3i64.max(wi / 100);
        col_bright_sm = box_smooth(&col_bright, b_sigma);
        let cb = &col_bright_sm;

        let window = |x_lo: i64, x_hi: i64, p_lo: i64, p_hi: i64| -> (i64, i64) {
            match side {
                Some(PageSide::Left) => (x_lo + (x_hi - x_lo) * 65 / 100, (wi - 10).min(p_hi)),
                Some(PageSide::Right) => (10i64.max(p_lo), x_lo + (x_hi - x_lo) * 35 / 100),
                None => (x_lo + (x_hi - x_lo) / 10, x_hi - (x_hi - x_lo) / 10),
            }
        };
        let (spine_lo_w, spine_hi_w) = window(x_lo, x_hi, p_lo, p_hi);
        let (mut search_lo_d, mut search_hi_d) = (spine_lo_w, spine_hi_w);
        if spine_lo_w + 20 < spine_hi_w {
            let (mut best_idx, mut max_depth, mut best_spine_min, mut best_flank_b) = (-1i64, -1.0f64, 0.0f64, 0.0f64);
            for x in (spine_lo_w + 10)..=(spine_hi_w - 10) {
                let left_b = (spine_lo_w..x).fold(0.0, |a, j| a + cb[j as usize]) / (x - spine_lo_w) as f64;
                let right_b = (x + 1..=spine_hi_w).fold(0.0, |a, j| a + cb[j as usize]) / (spine_hi_w - x) as f64;
                let flank = left_b.min(right_b);
                if flank > 0.0 {
                    let val = cb[x as usize];
                    let depth = (flank - val) / flank;
                    if depth > max_depth {
                        max_depth = depth;
                        best_idx = x;
                        best_spine_min = val;
                        best_flank_b = flank;
                    }
                }
            }
            if best_idx != -1 {
                if best_spine_min < best_flank_b * 0.85 {
                    let spine_thr = best_flank_b * 0.90;
                    let (mut s_lo, mut s_hi) = (best_idx, best_idx);
                    while s_lo > spine_lo_w && cb[(s_lo - 1) as usize] < spine_thr {
                        s_lo -= 1;
                    }
                    while s_hi < spine_hi_w && cb[(s_hi + 1) as usize] < spine_thr {
                        s_hi += 1;
                    }
                    if s_hi - s_lo + 1 >= 2 {
                        gaps.push((s_lo, s_hi + 1));
                        gap_centre = best_idx;
                    }
                }
            }
        }

        // ── 7b. pale-binding valley on the smoothed ink density
        if gaps.is_empty() {
            let (search_lo, search_hi) = window(x_lo, x_hi, p_lo, p_hi);
            search_lo_d = search_lo;
            search_hi_d = search_hi;
            if search_lo + 20 < search_hi {
                let search_range = search_hi - search_lo;
                let banded = side.is_none();
                let left_band_hi = if banded { search_lo + (search_range as f64 * 0.28) as i64 } else { search_hi };
                let right_band_lo = if banded { search_hi - (search_range as f64 * 0.28) as i64 } else { search_lo };
                let (mut best_x, mut max_depth, mut best_valley_min, mut best_flanking) = (-1i64, -1.0f64, 0.0f64, 0.0f64);
                for x in (search_lo + 10)..=(search_hi - 10) {
                    if x > left_band_hi && x < right_band_lo {
                        continue;
                    }
                    let l_avg = (search_lo..x).fold(0.0, |a, j| a + col[j as usize]) / (x - search_lo) as f64;
                    let r_avg = (x + 1..=search_hi).fold(0.0, |a, j| a + col[j as usize]) / (search_hi - x) as f64;
                    let flank = l_avg.min(r_avg);
                    if flank > 0.05 {
                        let val = col[x as usize];
                        let depth = (flank - val) / flank;
                        if depth > max_depth {
                            max_depth = depth;
                            best_x = x;
                            best_valley_min = val;
                            best_flanking = flank;
                        }
                    }
                }
                if best_x != -1 {
                    let outer_flank_w = match side {
                        Some(PageSide::Right) => best_x - search_lo,
                        Some(PageSide::Left) => search_hi - best_x,
                        None => i64::MAX,
                    };
                    let min_flank_w = 50i64.max(wi / 18);
                    let keep_valley_w = match side {
                        Some(PageSide::Right) => (x_hi - best_x) as f64,
                        Some(PageSide::Left) => (best_x - x_lo) as f64,
                        None => (best_x - x_lo).max(x_hi - best_x) as f64,
                    };
                    if best_valley_min < best_flanking * 0.70
                        && (side.is_none() || outer_flank_w >= min_flank_w)
                        && keep_valley_w >= 0.55 * (x_hi - x_lo + 1) as f64
                    {
                        let t_thr = best_flanking * 0.80;
                        let (mut t_lo, mut t_hi) = (best_x, best_x);
                        while t_lo > search_lo && col[(t_lo - 1) as usize] < t_thr {
                            t_lo -= 1;
                        }
                        while t_hi < search_hi && col[(t_hi + 1) as usize] < t_thr {
                            t_hi += 1;
                        }
                        if t_hi - t_lo + 1 >= 2 {
                            gaps.push((t_lo, t_hi + 1));
                            gap_centre = best_x;
                        }
                    }
                }
            }
        }

        // ── 7c. shallow LOCAL fold (only with a page_side)
        if gaps.is_empty() && side.is_some() {
            let f = 20i64.max(wi / 20);
            let (lo, hi_) = (search_lo_d.max(f), search_hi_d.min(wi - f));
            let (mut best_x, mut max_depth, mut best_fl) = (-1i64, -1.0f64, 0.0f64);
            if lo + 2 < hi_ {
                for x in (lo + 1)..hi_ {
                    let l_max = (x - f..x).fold(0.0f64, |a, j| a.max(cb[j as usize]));
                    let r_max = (x..x + f).fold(0.0f64, |a, j| a.max(cb[j as usize]));
                    let fl = l_max.min(r_max);
                    if fl > 0.0 {
                        let depth = (fl - cb[x as usize]) / fl;
                        if depth > max_depth {
                            max_depth = depth;
                            best_x = x;
                            best_fl = fl;
                        }
                    }
                }
            }
            let mut fold_gates = false;
            if best_x >= 0 && max_depth >= 0.055 {
                let dip_ink_free = col_ink[best_x as usize] < 0.015;
                let discard_ink_cols = if side == Some(PageSide::Right) {
                    (lo..best_x).filter(|&j| col_ink[j as usize] > 0.02).count()
                } else {
                    (best_x + 1..=hi_).filter(|&j| col_ink[j as usize] > 0.02).count()
                };
                fold_gates = dip_ink_free && discard_ink_cols >= 15;
            }
            if best_x >= 0 && max_depth >= 0.055 && fold_gates {
                let keep_w = if side == Some(PageSide::Right) { (x_hi - best_x) as f64 } else { (best_x - x_lo) as f64 };
                if keep_w >= 0.55 * (x_hi - x_lo + 1) as f64 {
                    let thr = best_fl - (best_fl - cb[best_x as usize]) * 0.5;
                    let (mut f_lo, mut f_hi) = (best_x, best_x);
                    while f_lo > lo && cb[(f_lo - 1) as usize] < thr {
                        f_lo -= 1;
                    }
                    while f_hi < hi_ && cb[(f_hi + 1) as usize] < thr {
                        f_hi += 1;
                    }
                    if f_hi - f_lo + 1 >= 2 {
                        gaps.push((f_lo, f_hi + 1));
                        gap_centre = best_x;
                    }
                }
            }
        }
    }

    // densest segment between gaps (ink-weighted, cover columns colInk ≥ 0.70 excluded)
    let (mut seg_lo, mut seg_hi) = (x_lo, x_hi);
    let mut chosen_segment_idx = -1i64;
    if !gaps.is_empty() && !keep_spread {
        let starts: Vec<i64> = std::iter::once(x_lo).chain(gaps.iter().map(|g| g.1)).collect();
        let ends: Vec<i64> = gaps.iter().map(|g| g.0).chain(std::iter::once(x_hi + 1)).collect();
        let mut best_sum = -1.0f64;
        let mut segment_idx = 0i64;
        for (&s, &e) in starts.iter().zip(ends.iter()) {
            if s >= e {
                continue;
            }
            let sum = (s..e).fold(0.0, |a, x| a + if col_ink[x as usize] < 0.70 { col[x as usize] } else { 0.0 });
            if sum > best_sum {
                best_sum = sum;
                seg_lo = s;
                seg_hi = e - 1;
                chosen_segment_idx = segment_idx;
            }
            segment_idx += 1;
        }
    }

    // final X
    let col_margin = || 2i64.max((w.max(h) as f64 * 0.025) as i64);
    if keep_spread {
        let (mut lo, mut hi_) = (x_lo, x_hi);
        while lo > p_lo && mean_col[(lo - 1) as usize] > bg_thr {
            lo -= 1;
        }
        while hi_ < p_hi && mean_col[(hi_ + 1) as usize] > bg_thr {
            hi_ += 1;
        }
        x_lo = lo;
        x_hi = hi_;
    } else if !gaps.is_empty() && side == Some(PageSide::Left) {
        x_lo = original_x_lo.max(p_lo);
        x_hi = p_hi.min(gaps[0].0 + 8);
    } else if !gaps.is_empty() && side == Some(PageSide::Right) {
        x_lo = p_lo.max(gaps[gaps.len() - 1].1 - 8);
        x_hi = original_x_hi.min(p_hi);
    } else if chosen_segment_idx != -1 {
        let c = chosen_segment_idx as usize;
        x_lo = if c > 0 { p_lo.max(gaps[c - 1].1 - 8) } else { p_lo.max(seg_lo) };
        x_hi = if c < gaps.len() { p_hi.min(gaps[c].0 + 8) } else { p_hi.min(seg_hi) };
    } else if side == Some(PageSide::Left) {
        x_lo = original_x_lo.max(p_lo);
        x_hi = p_hi.min(x_hi + col_margin());
    } else if side == Some(PageSide::Right) {
        x_lo = p_lo.max(x_lo - col_margin());
        x_hi = original_x_hi.min(p_hi);
    } else {
        x_lo = p_lo.max(x_lo - col_margin());
        x_hi = p_hi.min(x_hi + col_margin());
    }

    // ink-safety clamp: stroke tails on bright paper inside the gap
    if let Some(&g) = gaps.first()
        && !keep_spread
        && !col_bright_sm.is_empty()
    {
        // the centre of the tier that MADE the gap (as the Swift implementation): the old falzFound / troughLo choice was also set by a
        // dip too narrow for a gap (a 1-column spine, then the valley tier makes the gap) and centred the walk outside it
        let c = gap_centre;
        if c >= 0 {
            let cb = &col_bright_sm;
            let paper_ref = cb[0.max(g.0 - 1) as usize].max(cb[(wi - 1).min(g.1) as usize]);
            let bright_thr = paper_ref * 0.97;
            let ink_thr = 0.008;
            if x_lo >= g.0 {
                let mut x = c + 1;
                while x < g.1 {
                    if col_ink[x as usize] > ink_thr && cb[x as usize] > bright_thr {
                        x_lo = x_lo.min(p_lo.max(x - 4));
                        break;
                    }
                    x += 1;
                }
            }
            if x_hi <= g.1 {
                let mut x = c - 1;
                while x > g.0 {
                    if col_ink[x as usize] > ink_thr && cb[x as usize] > bright_thr {
                        x_hi = x_hi.max(p_hi.min(x + 4));
                        break;
                    }
                    x -= 1;
                }
            }
        }
    }

    // ── 8. row brightness → Y trim
    let w_d = w as f64;
    let mut row_bright = vec![0.0f64; h];
    let mut row_mean = vec![0.0f64; h];
    for y in 0..h {
        let (mut cnt, mut sum) = (0i64, 0i64);
        for &v in &orig_gray[y * w..(y + 1) * w] {
            if v > 140 {
                cnt += 1;
            }
            sum += v as i64;
        }
        row_bright[y] = cnt as f64 / w_d;
        row_mean[y] = sum as f64 / w_d;
    }
    let pad = 6i64;
    let (mut y_lo, mut y_hi) = (0i64, hi - 1);
    if let (Some(rf), Some(rl)) =
        (row_mean.iter().position(|&v| v > bg_thr), row_mean.iter().rposition(|&v| v > bg_thr))
    {
        let (mut r_lo, mut r_hi) = (rf as i64, rl as i64);
        let ruler_gap_h = 20i64.max(hi / 80);
        let max_strip_h = hi / 8;
        let b_thr = 0.15;
        r_hi = furniture_trim(&row_bright, &row_mean, r_hi, r_lo, -1, b_thr, bg_thr, ruler_gap_h, i64::MAX, max_strip_h);
        r_lo = furniture_trim(&row_bright, &row_mean, r_lo, r_hi, 1, b_thr, bg_thr, ruler_gap_h, i64::MAX, max_strip_h);
        y_lo = 0.max(r_lo - pad);
        y_hi = (hi - 1).min(r_hi + pad);
    }
    if let Some((rl, rh)) = leaf_rows
        && !leaf_without_mount
    {
        y_lo = y_lo.min(0.max(rl - pad));
        y_hi = y_hi.max((hi - 1).min(rh + pad));
    }
    if leaf_without_mount {
        y_lo = 0;
        y_hi = hi - 1;
    }
    if y_hi - y_lo < (0.35 * h as f64) as i64 {
        y_lo = 0;
        y_hi = hi - 1;
    }

    Some(Decision { scale: f, x_lo, x_hi, y_lo, y_hi, gaps, chosen_segment_idx })
}

/// centred box mean over ±r, window clipped to the array
fn box_smooth(a: &[f64], r: i64) -> Vec<f64> {
    let n = a.len() as i64;
    (0..n)
        .map(|x| {
            let (lo, hi) = (0.max(x - r), (n - 1).min(x + r));
            let mut s = 0.0;
            for j in lo..=hi {
                s += a[j as usize];
            }
            s / (hi - lo + 1) as f64
        })
        .collect()
}

const CROP_MARGIN_FRAC: f64 = 0.01;
const CROP_INK_MARGIN_FRAC: f64 = 0.03;
const CROP_MARGIN_DIP: f64 = 0.90;
const CROP_MARGIN_INK: f64 = 0.65;
const CROP_MARGIN_REF: i64 = 10;
const CROP_MARGIN_PAPER_FRAC: f64 = 0.85;

/// NSImage.cropSafetyMargin on the FULL-RES image: `r` = (x0, y0, x1, y1) grown outward per side — a barrier (fold,
/// shadow, scanner border) is never crossed, strokes crossing the edge are kept whole up to 3 %, a blind 1 % margin on
/// every non-facing side. `facing` = [L, R].
fn crop_safety_margin(img: &RgbImage, r: (i64, i64, i64, i64), facing: [bool; 2]) -> (i64, i64, i64, i64) {
    let (big_w, big_h) = (img.width() as i64, img.height() as i64);
    let (x0, y0, x1, y1) = r;
    if x1 - x0 < 4 || y1 - y0 < 4 {
        return r;
    }
    let mut out = [0i64; 4];
    // L, R, T, B
    for (k, out_k) in out.iter_mut().enumerate() {
        let horizontal = k < 2;
        let dim = if horizontal { x1 - x0 } else { y1 - y0 };
        let cap = (CROP_INK_MARGIN_FRAC * dim as f64).round() as i64;
        let is_facing = k < 2 && facing[k];
        let blind = if is_facing { 0 } else { (CROP_MARGIN_FRAC * dim as f64).round() as i64 };
        let ref_n = CROP_MARGIN_REF.min(dim / 4);
        let room = match k {
            0 => x0,
            1 => big_w - x1,
            2 => y0,
            _ => big_h - y1,
        };
        let avail = cap.min(room);
        if avail <= 0 || ref_n < 2 {
            *out_k = 0;
            continue;
        }
        let n_t = ref_n + avail;
        let n_s = if horizontal { y1 - y0 } else { x1 - x0 };
        // oriented grey: g[ti * nS + s], ti = 0 … refN-1 inside (refN-1 = the last row/column inside), ≥ refN outside
        let mut g = vec![0u8; (n_t * n_s) as usize];
        for ti in 0..n_t {
            for s in 0..n_s {
                let (x, y) = match k {
                    0 => (x0 + ref_n - 1 - ti, y0 + s),
                    1 => (x1 - ref_n + ti, y0 + s),
                    2 => (x0 + s, y0 + ref_n - 1 - ti),
                    _ => (x0 + s, y1 - ref_n + ti),
                };
                g[(ti * n_s + s) as usize] = luma_of(img.get_pixel(x as u32, y as u32));
            }
        }
        let gi = |ti: i64, s: i64| g[(ti * n_s + s) as usize];
        // p75 tone per oriented row/column
        let mut p75 = vec![0.0f64; n_t as usize];
        for ti in 0..n_t {
            let mut hist = [0i64; 256];
            for s in 0..n_s {
                hist[gi(ti, s) as usize] += 1;
            }
            let mut acc = 0i64;
            let need = n_s * 25 / 100;
            for v in (0..256).rev() {
                acc += hist[v];
                if acc > need {
                    p75[ti as usize] = v as f64;
                    break;
                }
            }
        }
        let mut refs: Vec<f64> = p75[0..ref_n as usize].to_vec();
        refs.sort_by(|a, b| a.total_cmp(b));
        let reference = refs[refs.len() / 2];
        if reference <= DOC_BG_THR {
            *out_k = 0;
            continue;
        }
        // barrier: first dip below cropMarginDip·ref, descended to its darkest point
        let mut barrier = avail;
        let mut dip_start = avail;
        let mut t = 0i64;
        while t < avail {
            if p75[(t + ref_n) as usize] < CROP_MARGIN_DIP * reference {
                dip_start = t;
                while t + 1 < avail && p75[(t + 1 + ref_n) as usize] <= p75[(t + ref_n) as usize] {
                    t += 1;
                }
                barrier = t;
                break;
            }
            t += 1;
        }
        let n_ti = ref_n + barrier;
        let rr = 2 * CROP_MARGIN_REF;
        let mut dark_at = vec![0i64; (n_s + 1) as usize];
        for s in 0..n_s {
            let c = (0..ref_n).filter(|&ti| gi(ti, s) as f64 <= DOC_BG_THR).count() as i64;
            dark_at[(s + 1) as usize] = dark_at[s as usize] + c;
        }
        let on_paper: Vec<bool> = (0..n_s)
            .map(|s| {
                let (lo, hi) = (0.max(s - rr), (n_s - 1).min(s + rr));
                let dark = dark_at[(hi + 1) as usize] - dark_at[lo as usize];
                dark as f64 <= (1.0 - CROP_MARGIN_PAPER_FRAC) * ((hi - lo + 1) * ref_n) as f64
            })
            .collect();
        let mut ink = vec![false; (n_ti * n_s) as usize];
        for ti in 0..n_ti {
            let thr = CROP_MARGIN_INK * p75[ti as usize];
            for s in 0..n_s {
                if on_paper[s as usize] && (gi(ti, s) as f64) < thr {
                    ink[(ti * n_s + s) as usize] = true;
                }
            }
        }
        let mut seen = vec![false; (n_ti * n_s) as usize];
        let mut need_t = 0i64;
        let mut stack: Vec<i64> = Vec::new();
        for s in 0..n_s {
            let start = (ref_n - 1) * n_s + s;
            if !ink[start as usize] || seen[start as usize] {
                continue;
            }
            let mut max_ti = ref_n - 1;
            stack.push(start);
            seen[start as usize] = true;
            while let Some(p) = stack.pop() {
                let (ti, ss) = (p / n_s, p % n_s);
                if ti > max_ti {
                    max_ti = ti;
                }
                for dt in -2..=2 {
                    for ds in -2..=2 {
                        if dt == 0 && ds == 0 {
                            continue;
                        }
                        let (nt, ns) = (ti + dt, ss + ds);
                        if nt < 0 || nt >= n_ti || ns < 0 || ns >= n_s {
                            continue;
                        }
                        let q = (nt * n_s + ns) as usize;
                        if ink[q] && !seen[q] {
                            seen[q] = true;
                            stack.push(q as i64);
                        }
                    }
                }
            }
            // a component through the whole window (no barrier, not stopped by the image edge) is a structure
            if barrier == avail && avail == cap && max_ti == n_ti - 1 {
                continue;
            }
            need_t = need_t.max(max_ti - ref_n + 1);
        }
        if barrier < avail && need_t >= dip_start {
            need_t = barrier;
        }
        *out_k = need_t.max(blind.min(barrier));
    }
    (x0 - out[0], y0 - out[2], x1 + out[1], y1 + out[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAPER: Rgb<u8> = Rgb([235, 232, 225]);

    /// paper over [x0, x1) × [y0, y1) with text lines (words of jittered length, shifted per line so no column stays
    /// blank down the page — vertically aligned word gaps would read as a pale-binding valley)
    fn paint_page(img: &mut RgbImage, x0: u32, x1: u32, y0: u32, y1: u32, margin: u32) {
        for y in y0..y1 {
            for x in x0..x1 {
                img.put_pixel(x, y, PAPER);
            }
        }
        let mut line = 0u32;
        let mut y = y0 + margin;
        while y + 12 < y1 - margin {
            let mut x = x0 + margin + (line * 17) % 23;
            let mut word = 0u32;
            while x < x1 - margin {
                let len = (25 + (line * 7 + word * 13) % 30).min(x1 - margin - x);
                for xx in x..x + len {
                    for dy in 0..10 {
                        if (xx + dy) % 3 != 0 {
                            img.put_pixel(xx, y + dy, Rgb([40, 40, 50]));
                        }
                    }
                }
                x += len + 10 + (word * 5) % 7;
                word += 1;
            }
            y += 36;
            line += 1;
        }
    }

    fn page_in_border(w: u32, h: u32, border: u32) -> RgbImage {
        let mut img = RgbImage::from_pixel(w, h, Rgb([20, 20, 20]));
        paint_page(&mut img, border, w - border, border, h - border, 60);
        img
    }

    #[test]
    fn crops_inside_a_dark_border() {
        let img = page_in_border(900, 1200, 70);
        let r = auto_crop(&img, &AutoCropOptions::default()).unwrap();
        // the paper edge at 70 ± the safety margin (≤ 3 % into a border right at the edge), the text (130 … 770) kept
        assert!(r.x >= 70 - 30 && r.x <= 130, "{r:?}");
        assert!(r.y >= 70 - 40 && r.y <= 130, "{r:?}");
        assert!(r.x + r.w >= 770 && r.x + r.w <= 830 + 30, "{r:?}");
        assert!(r.y + r.h >= 1070 && r.y + r.h <= 1130 + 40, "{r:?}");
    }

    #[test]
    fn analysis_downscale_maps_back_to_full_res() {
        // 2200 px high → analysis at 1100 (scale ½); the rect comes back in the input's pixels
        let img = page_in_border(1650, 2200, 140);
        let r = auto_crop(&img, &AutoCropOptions::default()).unwrap();
        assert!(r.x >= 140 - 50 && r.x <= 200, "{r:?}");
        assert!(r.x + r.w >= 1450 && r.x + r.w <= 1510 + 50, "{r:?}");
        assert!(r.y >= 140 - 70 && r.y + r.h <= 2060 + 70, "{r:?}");
    }

    #[test]
    fn blank_image_is_kept_whole() {
        let img = RgbImage::from_pixel(300, 400, Rgb([240, 240, 240]));
        let r = auto_crop(&img, &AutoCropOptions::default()).unwrap();
        assert_eq!(r, CropRect { x: 0, y: 0, w: 300, h: 400 });
    }

    /// two pages side by side, a dark binding spine (600 … 640) between them
    fn spread() -> RgbImage {
        let mut img = RgbImage::from_pixel(1240, 1000, Rgb([70, 65, 60]));
        paint_page(&mut img, 0, 600, 0, 1000, 60);
        paint_page(&mut img, 640, 1240, 0, 1000, 60);
        img
    }

    #[test]
    fn spread_is_split_at_the_spine() {
        let r = auto_crop(&spread(), &AutoCropOptions::default()).unwrap();
        // one page: the cut sits 8 analysis px into the spine (+ the ink-aware margin on the facing side)
        assert!(r.x + r.w <= 660 || r.x >= 580, "split expected: {r:?}");
        assert!(r.w >= 500 && r.w <= 700, "{r:?}");
    }

    #[test]
    fn keep_spread_keeps_both_pages() {
        let r = auto_crop(&spread(), &AutoCropOptions { keep_spread: true, ..Default::default() }).unwrap();
        assert!(r.x <= 60 && r.x + r.w >= 1180, "whole spread expected: {r:?}");
    }

    #[test]
    fn page_side_right_drops_the_facing_sliver() {
        // a 140 px sliver of the facing page, spine 140 … 170, the right page 170 … 1000 (text from 230)
        let mut img = RgbImage::from_pixel(1000, 1100, Rgb([70, 65, 60]));
        paint_page(&mut img, 0, 140, 0, 1100, 20);
        paint_page(&mut img, 170, 1000, 0, 1100, 60);
        let opts = AutoCropOptions { page_side: Some(PageSide::Right), ..Default::default() };
        let r = auto_crop(&img, &opts).unwrap();
        assert!(r.x >= 130 && r.x <= 230, "right page expected: {r:?}");
        assert!(r.x + r.w >= 940, "{r:?}");
    }

    #[test]
    fn options_are_camel_case() {
        let o: AutoCropOptions =
            serde_json::from_str(r#"{"keepSpread":true,"pageSide":"left","singleLeaf":false}"#).unwrap();
        assert_eq!(o, AutoCropOptions { keep_spread: true, page_side: Some(PageSide::Left), single_leaf: false });
        let r = serde_json::to_string(&CropRect { x: 1, y: 2, w: 3, h: 4 }).unwrap();
        assert_eq!(r, r#"{"x":1,"y":2,"w":3,"h":4}"#);
    }
}
