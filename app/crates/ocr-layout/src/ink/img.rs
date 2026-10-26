// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! The image operations the ink analysis needs, as OpenCV / scipy compute them (the desktop tools' reference is
//! Python): area resize, Gaussian blur (reflect-101 border), Otsu, NICK binarisation, rectangle and ellipse
//! morphology (outside the image ignored), 8-connected components with stats, the 3 × 3 L2 chamfer distance, the exact
//! (anisotropic) Euclidean distance, outer contours + Douglas–Peucker.

use crate::raster::Mask;

/// an 8-bit grey image
#[derive(Clone, Debug, PartialEq)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

impl Gray {
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.px[y * self.w + x]
    }
}

/// RGBA → grey as PIL's convert("L"): (r·19595 + g·38470 + b·7471 + 2¹⁵) >> 16
pub fn luma_pil(rgba: &[u8], w: usize, h: usize) -> Gray {
    let px = rgba
        .chunks_exact(4)
        .map(|p| ((p[0] as u32 * 19595 + p[1] as u32 * 38470 + p[2] as u32 * 7471 + 0x8000) >> 16) as u8)
        .collect();
    Gray { w, h, px }
}

/// RGB → grey as PIL's convert("L")
pub fn luma_pil_rgb(rgb: &[u8], w: usize, h: usize) -> Gray {
    let px = rgb
        .chunks_exact(3)
        .map(|p| ((p[0] as u32 * 19595 + p[1] as u32 * 38470 + p[2] as u32 * 7471 + 0x8000) >> 16) as u8)
        .collect();
    Gray { w, h, px }
}

/// downscaled to nw × nh by pixel-area averaging (OpenCV INTER_AREA): every source pixel weighted by its overlap
pub fn resize_area(g: &Gray, nw: usize, nh: usize) -> Gray {
    let weights = |n_src: usize, n_dst: usize| -> Vec<Vec<(usize, f64)>> {
        let s = n_src as f64 / n_dst as f64;
        (0..n_dst)
            .map(|d| {
                let (a, b) = (d as f64 * s, (d + 1) as f64 * s);
                let mut out = Vec::new();
                let mut i = a.floor() as usize;
                while (i as f64) < b && i < n_src {
                    let lo = a.max(i as f64);
                    let hi = b.min(i as f64 + 1.0);
                    if hi > lo {
                        out.push((i, (hi - lo) / s));
                    }
                    i += 1;
                }
                out
            })
            .collect()
    };
    let (wx, wy) = (weights(g.w, nw), weights(g.h, nh));
    let mut tmp = vec![0f64; nw * g.h];
    for y in 0..g.h {
        for (x, ws) in wx.iter().enumerate() {
            tmp[y * nw + x] = ws.iter().map(|&(i, w)| w * g.px[y * g.w + i] as f64).sum();
        }
    }
    let mut px = vec![0u8; nw * nh];
    for (y, ws) in wy.iter().enumerate() {
        for x in 0..nw {
            let v: f64 = ws.iter().map(|&(i, w)| w * tmp[i * nw + x]).sum();
            px[y * nw + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    Gray { w: nw, h: nh, px }
}

/// a normalised Gaussian kernel of `ksize` taps (OpenCV getGaussianKernel)
fn gaussian_kernel(ksize: usize, sigma: f64) -> Vec<f64> {
    let c = (ksize as f64 - 1.0) / 2.0;
    let k: Vec<f64> = (0..ksize).map(|i| (-((i as f64 - c).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f64 = k.iter().sum();
    k.into_iter().map(|v| v / s).collect()
}

/// reflect-101 index into 0..n
fn reflect101(i: i64, n: usize) -> usize {
    let n = n as i64;
    if n == 1 {
        return 0;
    }
    let mut i = i;
    while i < 0 || i >= n {
        if i < 0 {
            i = -i;
        }
        if i >= n {
            i = 2 * n - 2 - i;
        }
    }
    i as usize
}

/// a 1-D profile blurred with `sigma` (OpenCV on float64: 8σ + 1 taps, reflect-101)
pub fn blur_1d(p: &[f64], sigma: f64) -> Vec<f64> {
    let ksize = ((sigma * 8.0 + 1.0).round() as usize) | 1;
    let k = gaussian_kernel(ksize, sigma);
    let r = (ksize / 2) as i64;
    (0..p.len())
        .map(|i| (0..ksize).map(|j| k[j] * p[reflect101(i as i64 + j as i64 - r, p.len())]).sum())
        .collect()
}

/// the image blurred with `sigma` (OpenCV on 8 bit: 6σ + 1 taps, reflect-101), rounded
pub fn blur(g: &Gray, sigma: f64) -> Gray {
    let ksize = ((sigma * 6.0 + 1.0).round() as usize) | 1;
    let k = gaussian_kernel(ksize, sigma);
    let r = (ksize / 2) as i64;
    let mut tmp = vec![0f64; g.w * g.h];
    for y in 0..g.h {
        for x in 0..g.w {
            tmp[y * g.w + x] = (0..ksize).map(|j| k[j] * g.px[y * g.w + reflect101(x as i64 + j as i64 - r, g.w)] as f64).sum();
        }
    }
    let mut px = vec![0u8; g.w * g.h];
    for y in 0..g.h {
        for x in 0..g.w {
            let v: f64 = (0..ksize).map(|j| k[j] * tmp[reflect101(y as i64 + j as i64 - r, g.h) * g.w + x]).sum();
            px[y * g.w + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    Gray { w: g.w, h: g.h, px }
}

/// Otsu's threshold: pixels > it are the bright class
pub fn otsu(g: &Gray) -> u8 {
    let mut hist = [0f64; 256];
    for &v in &g.px {
        hist[v as usize] += 1.0;
    }
    let n = g.px.len() as f64;
    let mu: f64 = (0..256).map(|i| i as f64 * hist[i]).sum::<f64>() / n;
    let (mut q1, mut mu1, mut best, mut t) = (0.0, 0.0, 0.0, 0u8);
    for i in 0..256 {
        let p = hist[i] / n;
        let q2 = q1 + p;
        if q2 > 0.0 && q2 < 1.0 {
            let m1 = (mu1 * q1 + i as f64 * p) / q2;
            let m2 = (mu - q2 * m1) / (1.0 - q2);
            let between = q2 * (1.0 - q2) * (m1 - m2) * (m1 - m2);
            if between > best {
                best = between;
                t = i as u8;
            }
            mu1 = m1;
        } else if q2 > 0.0 {
            mu1 = (mu1 * q1 + i as f64 * p) / q2;
        }
        q1 = q2;
    }
    t
}

/// NICK adaptive binarisation (k < 0 keeps fainter ink; the window radius scales with the image): ink = true where the
/// grey is ≤ mean + k · √(var + mean² · (area − 1) / area)
pub fn nick_ink(g: &Gray, k: f64) -> Mask {
    let (w, h) = (g.w, g.h);
    let r = 10.max(40.min(w.min(h) / 12)) as i64;
    let mut s1 = vec![0f64; (w + 1) * (h + 1)];
    let mut s2 = vec![0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let (mut a, mut b) = (0.0, 0.0);
        for x in 0..w {
            let v = g.px[y * w + x] as f64;
            a += v;
            b += v * v;
            s1[(y + 1) * (w + 1) + x + 1] = s1[y * (w + 1) + x + 1] + a;
            s2[(y + 1) * (w + 1) + x + 1] = s2[y * (w + 1) + x + 1] + b;
        }
    }
    let mut out = Mask::new(w, h);
    for y in 0..h {
        let (y0, y1) = ((y as i64 - r).clamp(0, h as i64 - 1) as usize, (y as i64 + r).clamp(0, h as i64 - 1) as usize);
        for x in 0..w {
            let (x0, x1) = ((x as i64 - r).clamp(0, w as i64 - 1) as usize, (x as i64 + r).clamp(0, w as i64 - 1) as usize);
            let bx = |s: &[f64]| s[(y1 + 1) * (w + 1) + x1 + 1] - s[y0 * (w + 1) + x1 + 1] - s[(y1 + 1) * (w + 1) + x0] + s[y0 * (w + 1) + x0];
            let area = ((y1 - y0 + 1) * (x1 - x0 + 1)) as f64;
            let mean = bx(&s1) / area;
            let var = (bx(&s2) / area - mean * mean).max(0.0);
            let t = mean + k * (var + mean * mean * (area - 1.0) / area).sqrt();
            out.px[y * w + x] = (g.px[y * w + x] as f64) <= t;
        }
    }
    out
}

// ── morphology (outside the image: ignored) ────────────────────────────────────────────────────────────────────────

/// a sliding max (dilate) or min (erode) of window `k` (anchor k / 2) along a line of `n` values read by `get`
fn slide(n: usize, k: usize, max: bool, get: impl Fn(usize) -> bool) -> Vec<bool> {
    let r = (k / 2) as i64;
    // running count of set values in the window: a max = any set, a min = all set (outside ignored)
    let mut pre = vec![0usize; n + 1];
    for i in 0..n {
        pre[i + 1] = pre[i] + get(i) as usize;
    }
    (0..n as i64)
        .map(|i| {
            let a = (i - r).max(0) as usize;
            let b = ((i - r + k as i64).min(n as i64)) as usize;
            let set = pre[b] - pre[a];
            if max { set > 0 } else { set == b - a }
        })
        .collect()
}

/// dilated (max = true) or eroded with a kw × kh rectangle
pub fn morph_rect(m: &Mask, kw: usize, kh: usize, max: bool) -> Mask {
    let (w, h) = (m.w, m.h);
    let mut tmp = Mask::new(w, h);
    for y in 0..h {
        let row = slide(w, kw, max, |x| m.px[y * w + x]);
        tmp.px[y * w..(y + 1) * w].copy_from_slice(&row);
    }
    let mut out = Mask::new(w, h);
    for x in 0..w {
        let col = slide(h, kh, max, |y| tmp.px[y * w + x]);
        for y in 0..h {
            out.px[y * w + x] = col[y];
        }
    }
    out
}

pub fn dilate_rect(m: &Mask, kw: usize, kh: usize) -> Mask {
    morph_rect(m, kw, kh, true)
}

pub fn erode_rect(m: &Mask, kw: usize, kh: usize) -> Mask {
    morph_rect(m, kw, kh, false)
}

pub fn open_rect(m: &Mask, kw: usize, kh: usize) -> Mask {
    dilate_rect(&erode_rect(m, kw, kh), kw, kh)
}

pub fn close_rect(m: &Mask, kw: usize, kh: usize) -> Mask {
    erode_rect(&dilate_rect(m, kw, kh), kw, kh)
}

/// OpenCV's ellipse element (getStructuringElement MORPH_ELLIPSE): per row its [from, to) columns
pub fn ellipse_rows(kw: usize, kh: usize) -> Vec<(usize, usize)> {
    let (r, c) = ((kh / 2) as i64, (kw / 2) as i64);
    let inv_r2 = if r > 0 { 1.0 / (r * r) as f64 } else { 0.0 };
    (0..kh as i64)
        .map(|i| {
            let dy = i - r;
            if dy.abs() <= r {
                let dx = (c as f64 * (((r * r - dy * dy) as f64) * inv_r2).sqrt()).round() as i64;
                (((c - dx).max(0)) as usize, ((c + dx + 1).min(kw as i64)) as usize)
            } else {
                (0, 0)
            }
        })
        .collect()
}

/// dilated with a kw × kh ellipse (OpenCV's element)
pub fn dilate_ellipse(m: &Mask, kw: usize, kh: usize) -> Mask {
    let (w, h) = (m.w, m.h);
    let rows = ellipse_rows(kw, kh);
    let (ay, ax) = ((kh / 2) as i64, (kw / 2) as i64);
    // per distinct element row: the image's rows dilated horizontally by that span
    let mut spans: Vec<(usize, usize)> = rows.iter().copied().filter(|s| s.1 > s.0).collect();
    spans.sort();
    spans.dedup();
    let mut out = Mask::new(w, h);
    for &(a, b) in &spans {
        // out(x) |= any m(x + j − ax) for j in a..b = a window of b − a starting at x + a − ax
        let len = b - a;
        let shift = a as i64 - ax;
        let mut hor = Mask::new(w, h);
        for y in 0..h {
            let mut pre = vec![0usize; w + 1];
            for x in 0..w {
                pre[x + 1] = pre[x] + m.px[y * w + x] as usize;
            }
            for x in 0..w as i64 {
                let s = (x + shift).clamp(0, w as i64) as usize;
                let e = (x + shift + len as i64).clamp(0, w as i64) as usize;
                hor.px[y * w + x as usize] = pre[e] > pre[s];
            }
        }
        for (i, row) in rows.iter().enumerate() {
            if *row != (a, b) {
                continue;
            }
            let dy = i as i64 - ay;
            for y in 0..h as i64 {
                let sy = y + dy;
                if sy < 0 || sy >= h as i64 {
                    continue;
                }
                for x in 0..w {
                    if hor.px[sy as usize * w + x] {
                        out.px[y as usize * w + x] = true;
                    }
                }
            }
        }
    }
    out
}

// ── components ────────────────────────────────────────────────────────────────────────────────────────────────────

/// a component's box and ink: x, y, w, h, area (OpenCV's stats)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stat {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub area: usize,
}

/// 8-connected components → (n incl. the background 0, label per pixel, stats per label); labels in raster order of
/// each component's first pixel
pub fn components(m: &Mask) -> (usize, Vec<u32>, Vec<Stat>) {
    let (w, h) = (m.w, m.h);
    let mut lab = vec![0u32; w * h];
    let mut st = vec![Stat { x: 0, y: 0, w, h, area: 0 }];
    let mut stack = Vec::new();
    for s in 0..w * h {
        if !m.px[s] || lab[s] != 0 {
            continue;
        }
        let id = st.len() as u32;
        let (mut x0, mut y0, mut x1, mut y1, mut area) = (usize::MAX, usize::MAX, 0, 0, 0);
        lab[s] = id;
        stack.push(s);
        while let Some(c) = stack.pop() {
            let (x, y) = (c % w, c / w);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
            area += 1;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                    if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                        continue;
                    }
                    let k = yy as usize * w + xx as usize;
                    if m.px[k] && lab[k] == 0 {
                        lab[k] = id;
                        stack.push(k);
                    }
                }
            }
        }
        st.push(Stat { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1, area });
    }
    st[0].area = lab.iter().filter(|&&l| l == 0).count();
    (st.len(), lab, st)
}

/// the holes of the set region filled (every unset pixel not reachable from the border through unset pixels, 4-way)
pub fn fill_holes(m: &Mask) -> Mask {
    let (w, h) = (m.w, m.h);
    let mut outside = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    for x in 0..w {
        stack.push(x);
        stack.push((h - 1) * w + x);
    }
    for y in 0..h {
        stack.push(y * w);
        stack.push(y * w + w - 1);
    }
    while let Some(c) = stack.pop() {
        if m.px[c] || outside[c] {
            continue;
        }
        outside[c] = true;
        let (x, y) = (c % w, c / w);
        if x > 0 {
            stack.push(c - 1);
        }
        if x + 1 < w {
            stack.push(c + 1);
        }
        if y > 0 {
            stack.push(c - w);
        }
        if y + 1 < h {
            stack.push(c + w);
        }
    }
    Mask { w, h, px: (0..w * h).map(|i| !outside[i]).collect() }
}

// ── distances ─────────────────────────────────────────────────────────────────────────────────────────────────────

/// every set pixel's distance to the nearest unset one (OpenCV DIST_L2 with the 3 × 3 mask: a = 0.955, b = 1.3693;
/// outside the image counts as set)
pub fn dist_l2_3(m: &Mask) -> Vec<f32> {
    let (w, h) = (m.w, m.h);
    let (a, b) = (0.955f32, 1.3693f32);
    let inf = 1e20f32;
    let mut d: Vec<f32> = m.px.iter().map(|&s| if s { inf } else { 0.0 }).collect();
    let get = |d: &Vec<f32>, x: i64, y: i64| if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 { inf } else { d[y as usize * w + x as usize] };
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let k = y as usize * w + x as usize;
            if d[k] == 0.0 {
                continue;
            }
            let v = d[k].min(get(&d, x - 1, y) + a).min(get(&d, x, y - 1) + a).min(get(&d, x - 1, y - 1) + b).min(get(&d, x + 1, y - 1) + b);
            d[k] = v;
        }
    }
    for y in (0..h as i64).rev() {
        for x in (0..w as i64).rev() {
            let k = y as usize * w + x as usize;
            if d[k] == 0.0 {
                continue;
            }
            let v = d[k].min(get(&d, x + 1, y) + a).min(get(&d, x, y + 1) + a).min(get(&d, x + 1, y + 1) + b).min(get(&d, x - 1, y + 1) + b);
            d[k] = v;
        }
    }
    d
}

/// the exact squared distance of a 1-D sampled function (Felzenszwalb–Huttenlocher), sample spacing `s`
fn edt_1d(f: &[f64], s: f64) -> Vec<f64> {
    let n = f.len();
    let mut d = vec![0f64; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0f64; n + 1];
    let mut k = 0usize;
    let first = f.iter().position(|&x| x < f64::INFINITY);
    let Some(first) = first else {
        return vec![f64::INFINITY; n];
    };
    v[0] = first;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let pos = |q: usize| q as f64 * s;
    for q in (first + 1)..n {
        if f[q] == f64::INFINITY {
            continue;
        }
        loop {
            let p = v[k];
            let sx = ((f[q] + pos(q) * pos(q)) - (f[p] + pos(p) * pos(p))) / (2.0 * (pos(q) - pos(p)));
            if sx <= z[k] {
                if k == 0 {
                    v[0] = q;
                    z[1] = f64::INFINITY;
                    break;
                }
                k -= 1;
                continue;
            }
            k += 1;
            v[k] = q;
            z[k] = sx;
            z[k + 1] = f64::INFINITY;
            break;
        }
    }
    k = 0;
    for q in 0..n {
        while z[k + 1] < pos(q) {
            k += 1;
        }
        let p = v[k];
        d[q] = (pos(q) - pos(p)).powi(2) + f[p];
    }
    d
}

/// the exact Euclidean distance of every pixel to the nearest SET pixel, rows `sy` apart and columns 1 (scipy's
/// distance_transform_edt of the complement with sampling (sy, 1)); no set pixel → infinity
pub fn edt_to_set(m: &Mask, sy: f64) -> Vec<f64> {
    let (w, h) = (m.w, m.h);
    let mut g = vec![0f64; w * h];
    for x in 0..w {
        let col: Vec<f64> = (0..h).map(|y| if m.px[y * w + x] { 0.0 } else { f64::INFINITY }).collect();
        let d = edt_1d(&col, sy);
        for y in 0..h {
            g[y * w + x] = d[y];
        }
    }
    let mut out = vec![0f64; w * h];
    for y in 0..h {
        let d = edt_1d(&g[y * w..(y + 1) * w], 1.0);
        for x in 0..w {
            out[y * w + x] = d[x].sqrt();
        }
    }
    out
}

// ── contours ──────────────────────────────────────────────────────────────────────────────────────────────────────

/// Douglas–Peucker of a CLOSED curve to `eps` (OpenCV approxPolyDP, closed): split at the point farthest from the
/// first, both halves simplified
pub fn approx_poly_closed(c: &[(i64, i64)], eps: f64) -> Vec<(i64, i64)> {
    let n = c.len();
    if n < 3 {
        return c.to_vec();
    }
    let d2 = |a: (i64, i64), b: (i64, i64)| ((a.0 - b.0).pow(2) + (a.1 - b.1).pow(2)) as f64;
    let mut far = 0;
    for i in 1..n {
        if d2(c[0], c[i]) > d2(c[0], c[far]) {
            far = i;
        }
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[far] = true;
    fn dp(c: &[(i64, i64)], i0: usize, i1: usize, eps: f64, keep: &mut [bool]) {
        let n = c.len();
        let len = (i1 + n - i0) % n;
        if len < 2 {
            return;
        }
        let (a, b) = (c[i0], c[i1 % n]);
        let (dx, dy) = ((b.0 - a.0) as f64, (b.1 - a.1) as f64);
        let l = dx.hypot(dy);
        let (mut best, mut k) = (-1.0, i0);
        for s in 1..len {
            let i = (i0 + s) % n;
            let (px, py) = ((c[i].0 - a.0) as f64, (c[i].1 - a.1) as f64);
            let d = if l > 0.0 { (px * dy - py * dx).abs() / l } else { px.hypot(py) };
            if d > best {
                best = d;
                k = i;
            }
        }
        if best > eps {
            keep[k] = true;
            dp(c, i0, k, eps, keep);
            dp(c, k, i1, eps, keep);
        }
    }
    dp(c, 0, far, eps, &mut keep);
    dp(c, far, n, eps, &mut keep);
    (0..n).filter(|&i| keep[i]).map(|i| c[i]).collect()
}

/// every 8-connected component's outer contour (Moore tracing), simplified to `eps` (OpenCV findContours EXTERNAL +
/// approxPolyDP); components of fewer than 3 contour points left out
pub fn outlines(m: &Mask, eps: f64) -> Vec<Vec<(i64, i64)>> {
    let (n, lab, st) = components(m);
    let mut out = Vec::new();
    for id in 1..n {
        let s = st[id];
        let mut piece = Mask::new(s.w, s.h);
        for y in 0..s.h {
            for x in 0..s.w {
                piece.px[y * s.w + x] = lab[(s.y + y) * m.w + s.x + x] == id as u32;
            }
        }
        let c: Vec<(i64, i64)> = piece.trace().into_iter().map(|(x, y)| (x + s.x as i64, y + s.y as i64)).collect();
        if c.len() < 3 {
            continue;
        }
        let a = approx_poly_closed(&c, eps);
        if a.len() >= 3 {
            out.push(a);
        }
    }
    out
}

/// numpy's percentile (linear interpolation) of unsorted values; None when empty
pub fn percentile(v: &[f64], q: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let pos = q / 100.0 * (s.len() - 1) as f64;
    let (i, f) = (pos.floor() as usize, pos - pos.floor());
    Some(if i + 1 < s.len() { s[i] + f * (s[i + 1] - s[i]) } else { s[i] })
}

/// numpy's median; None when empty
pub fn median(v: &[f64]) -> Option<f64> {
    percentile(v, 50.0)
}
