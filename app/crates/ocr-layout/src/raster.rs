// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! A binary raster for the geometry's raster steps (merge, hull, sides, clear, subtract): polygons filled (a pixel is
//! set when its centre is inside — non-zero winding, no anti-aliasing), outlines and lines stroked (a pixel within half
//! the width of a segment: round joins and caps), 8-connected components, a 1-px dilation, a chamfer distance and the
//! outer contour (Moore-neighbour tracing). Row y = the polygon's y (y down).

use crate::geom::Pt;

#[derive(Clone, PartialEq, Debug)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub px: Vec<bool>,
}

impl Mask {
    pub fn new(w: usize, h: usize) -> Mask {
        Mask { w, h, px: vec![false; w * h] }
    }

    pub fn at(&self, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h && self.px[y as usize * self.w + x as usize]
    }

    /// the polygon filled: every pixel whose centre lies inside (non-zero winding)
    pub fn fill(&mut self, p: &[Pt]) {
        let n = p.len();
        if n < 3 {
            return;
        }
        let (mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY);
        for q in p {
            y0 = y0.min(q.y);
            y1 = y1.max(q.y);
        }
        let ya = ((y0 - 0.5).ceil().max(0.0)) as usize;
        let yb = ((y1 - 0.5).floor().min(self.h as f64 - 1.0)) as i64;
        if yb < 0 {
            return;
        }
        let mut xs: Vec<(f64, i32)> = Vec::new();
        for y in ya..=(yb as usize) {
            let cy = y as f64 + 0.5;
            xs.clear();
            for i in 0..n {
                let a = p[i];
                let b = p[(i + 1) % n];
                if (a.y <= cy) != (b.y <= cy) {
                    let x = a.x + (cy - a.y) * (b.x - a.x) / (b.y - a.y);
                    xs.push((x, if b.y > a.y { 1 } else { -1 }));
                }
            }
            xs.sort_by(|u, v| u.0.total_cmp(&v.0));
            // inside where the winding is not 0: from the crossing that leaves 0 to the one that returns to it
            let mut wind = 0;
            let mut from = 0.0;
            for &(x, d) in &xs {
                if wind == 0 {
                    from = x;
                }
                wind += d;
                if wind == 0 {
                    self.span(y, from, x);
                }
            }
        }
    }

    /// the pixels of row y with their centre in [x0, x1)
    fn span(&mut self, y: usize, x0: f64, x1: f64) {
        let a = (x0 - 0.5).ceil().max(0.0) as i64;
        let b = ((x1 - 0.5).ceil() as i64).min(self.w as i64);
        for x in a..b {
            self.px[y * self.w + x as usize] = true;
        }
    }

    /// the segment a–b stroked with `width` (round ends)
    pub fn stroke_segment(&mut self, a: Pt, b: Pt, width: f64) {
        let r = width / 2.0;
        let xa = ((a.x.min(b.x) - r - 0.5).floor().max(0.0)) as usize;
        let xb = ((a.x.max(b.x) + r).ceil() as i64).min(self.w as i64 - 1);
        let ya = ((a.y.min(b.y) - r - 0.5).floor().max(0.0)) as usize;
        let yb = ((a.y.max(b.y) + r).ceil() as i64).min(self.h as i64 - 1);
        if xb < 0 || yb < 0 {
            return;
        }
        for y in ya..=(yb as usize) {
            for x in xa..=(xb as usize) {
                let c = Pt::new(x as f64 + 0.5, y as f64 + 0.5);
                if crate::geom::closest(a, b, c).0.dist(c) <= r {
                    self.px[y * self.w + x] = true;
                }
            }
        }
    }

    /// the closed outline of `p` stroked with `width` (round joins)
    pub fn stroke_closed(&mut self, p: &[Pt], width: f64) {
        for i in 0..p.len() {
            self.stroke_segment(p[i], p[(i + 1) % p.len()], width);
        }
    }

    /// 8-connected components → (label per pixel, −1 = unset; count)
    pub fn components(&self) -> (Vec<i64>, usize) {
        let (w, h) = (self.w, self.h);
        let mut label = vec![-1i64; w * h];
        let mut n = 0usize;
        let mut stack: Vec<usize> = Vec::new();
        for s in 0..(w * h) {
            if !self.px[s] || label[s] >= 0 {
                continue;
            }
            label[s] = n as i64;
            stack.push(s);
            while let Some(c) = stack.pop() {
                let (x, y) = ((c % w) as i64, (c / w) as i64);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                            continue;
                        }
                        let k = yy as usize * w + xx as usize;
                        if self.px[k] && label[k] < 0 {
                            label[k] = n as i64;
                            stack.push(k);
                        }
                    }
                }
            }
            n += 1;
        }
        (label, n)
    }

    /// the index of the set pixel nearest to (x, y) within ±6 px, else (x, y) clamped
    pub fn nearest_set(&self, x: i64, y: i64) -> usize {
        let (w, h) = (self.w as i64, self.h as i64);
        let mut best = (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize;
        let mut bd = i64::MAX;
        for yy in (y - 6).max(0)..=(y + 6).min(h - 1) {
            for xx in (x - 6).max(0)..=(x + 6).min(w - 1) {
                if self.px[(yy * w + xx) as usize] {
                    let d = (xx - x) * (xx - x) + (yy - y) * (yy - y);
                    if d < bd {
                        bd = d;
                        best = (yy * w + xx) as usize;
                    }
                }
            }
        }
        best
    }

    /// grown by one pixel (8 neighbours)
    pub fn dilate(&self) -> Mask {
        let mut out = self.clone();
        for y in 0..self.h as i64 {
            for x in 0..self.w as i64 {
                if self.at(x, y) {
                    continue;
                }
                let hit = (-1..=1).any(|dy| (-1..=1).any(|dx| self.at(x + dx, y + dy)));
                out.px[y as usize * self.w + x as usize] = hit;
            }
        }
        out
    }

    /// chamfer (3-4) distance of every pixel to the nearest set pixel, in pixels
    pub fn distance(&self) -> Vec<f32> {
        let (w, h) = (self.w, self.h);
        let inf = 1e9f32;
        let mut d: Vec<f32> = self.px.iter().map(|&b| if b { 0.0 } else { inf }).collect();
        for y in 0..h {
            for x in 0..w {
                let k = y * w + x;
                let mut v = d[k];
                if x > 0 {
                    v = v.min(d[k - 1] + 3.0);
                }
                if y > 0 {
                    v = v.min(d[k - w] + 3.0);
                    if x > 0 {
                        v = v.min(d[k - w - 1] + 4.0);
                    }
                    if x < w - 1 {
                        v = v.min(d[k - w + 1] + 4.0);
                    }
                }
                d[k] = v;
            }
        }
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let k = y * w + x;
                let mut v = d[k];
                if x < w - 1 {
                    v = v.min(d[k + 1] + 3.0);
                }
                if y < h - 1 {
                    v = v.min(d[k + w] + 3.0);
                    if x < w - 1 {
                        v = v.min(d[k + w + 1] + 4.0);
                    }
                    if x > 0 {
                        v = v.min(d[k + w - 1] + 4.0);
                    }
                }
                d[k] = v;
            }
        }
        d.into_iter().map(|v| v / 3.0).collect()
    }

    /// the outer boundary pixels of the (first) set region, clockwise on screen (Moore-neighbour tracing, Jacob's stop)
    pub fn trace(&self) -> Vec<(i64, i64)> {
        let Some(start) = self.px.iter().position(|&b| b) else {
            return Vec::new();
        };
        const DIRS: [(i64, i64); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];
        let s = ((start % self.w) as i64, (start / self.w) as i64);
        let mut out = vec![s];
        let mut cur = s;
        let mut back = 4usize; // entered the start from the west (the row above is empty)
        let mut steps = 0usize;
        while steps < 4 * self.w * self.h {
            steps += 1;
            let mut found = false;
            for k in 1..=8 {
                let d = (back + k) % 8;
                let (nx, ny) = (cur.0 + DIRS[d].0, cur.1 + DIRS[d].1);
                if self.at(nx, ny) {
                    back = (d + 4) % 8;
                    cur = (nx, ny);
                    found = true;
                    break;
                }
            }
            if !found {
                return out; // a single pixel
            }
            if cur == s && out.len() > 1 {
                break;
            }
            out.push(cur);
        }
        out
    }
}
