// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! The raster steps of the region geometry (ported 1:1 from the desktop GT tools): merge (union outline, enclosing
//! hull), snap to the page border, clear of the other regions, straight sides, subtract (give way). Each rasterises the
//! polygons into a box around them (≤ 900 / 1200 px), works on the pixels and traces the outline back, simplified
//! outward.

use crate::geom::{Failed, Pt, Rect, bounds, centroid, collinear_free, contains, convex_hull, covers, fail, is_simple, simplify_outward};
use crate::raster::Mask;

/// a raster box: polygon px → raster px (scale `s`, `pad` px around)
struct Frame {
    bb: Rect,
    s: f64,
    pad: f64,
    w: usize,
    h: usize,
}

impl Frame {
    fn new(bb: Rect, longest: f64, pad: usize) -> Frame {
        let s = (longest / bb.w.max(bb.h)).min(1.0);
        Frame { bb, s, pad: pad as f64, w: (bb.w * s).ceil() as usize + 2 * pad, h: (bb.h * s).ceil() as usize + 2 * pad }
    }

    fn to_r(&self, q: Pt) -> Pt {
        Pt::new((q.x - self.bb.x) * self.s + self.pad, (q.y - self.bb.y) * self.s + self.pad)
    }

    fn all_to_r(&self, p: &[Pt]) -> Vec<Pt> {
        p.iter().map(|&q| self.to_r(q)).collect()
    }

    /// a traced contour back to polygon px (pixel centres)
    fn back(&self, contour: &[(i64, i64)]) -> Vec<Pt> {
        contour
            .iter()
            .map(|&(x, y)| Pt::new((x as f64 + 0.5 - self.pad) / self.s + self.bb.x, (y as f64 + 0.5 - self.pad) / self.s + self.bb.y))
            .collect()
    }

    fn mask(&self) -> Mask {
        Mask::new(self.w, self.h)
    }

    fn filled(&self, polys: &[Vec<Pt>]) -> Mask {
        let mut m = self.mask();
        for p in polys {
            m.fill(&self.all_to_r(p));
        }
        m
    }

    /// the other regions' convex hulls, filled and stroked with 2 × `gap`
    fn blocked(&self, others: &[&Vec<Pt>], gap: f64) -> Mask {
        let mut m = self.mask();
        for o in others {
            let h = self.all_to_r(&convex_hull(o));
            m.fill(&h);
            m.stroke_closed(&h, (2.0 * gap * self.s).max(1.0));
        }
        m
    }

    /// the label of the component nearest to q (polygon px)
    fn label_at(&self, m: &Mask, label: &[i64], q: Pt) -> i64 {
        let c = self.to_r(q);
        label[m.nearest_set(c.x as i64, c.y as i64)]
    }
}

/// ONE polygon for several: the polygons rasterised (≤ 1200 px), parts that do not touch joined by a bridge (a thick
/// line between their closest corners), grown 1 raster px, the outer contour traced and simplified outward. Holes are
/// dropped (a region is its outline).
pub fn union(polys: &[Vec<Pt>], tol: f64) -> Result<Vec<Pt>, Failed> {
    if polys.len() < 2 || polys.iter().any(|p| p.len() < 3) {
        return fail("merge needs two or more polygons");
    }
    let all: Vec<Pt> = polys.iter().flatten().copied().collect();
    let f = Frame::new(bounds(&all), 1200.0, 4);
    let mut px = f.filled(polys);
    // join the parts: per round, the part of polygon 0 and the nearest other part, bridged corner to corner
    let low = polys.iter().map(|p| bounds(p).h).fold(40.0, f64::min);
    let bridge = (0.25 * low * f.s).max(3.0);
    for _ in 0..polys.len() {
        let (label, _) = px.components();
        let ids: Vec<i64> = polys.iter().map(|p| f.label_at(&px, &label, p[0])).collect();
        if ids.iter().all(|&i| i == ids[0]) {
            break;
        }
        let mut best: Option<(Pt, Pt)> = None;
        let mut bd = f64::INFINITY;
        for (i, p) in polys.iter().enumerate() {
            if ids[i] != ids[0] {
                continue;
            }
            for (j, q) in polys.iter().enumerate() {
                if ids[j] == ids[0] {
                    continue;
                }
                for &a in p {
                    for &b in q {
                        if a.dist(b) < bd {
                            bd = a.dist(b);
                            best = Some((a, b));
                        }
                    }
                }
            }
        }
        let Some((a, b)) = best else {
            break;
        };
        px.stroke_segment(f.to_r(a), f.to_r(b), bridge);
    }
    let (_, count) = px.components();
    if count != 1 {
        return fail(&format!("merge: the parts stay apart after bridging ({count} parts)"));
    }
    let contour = px.dilate().trace();
    if contour.len() < 3 {
        return fail("merge: no outline");
    }
    let simple = simplify_outward(&collinear_free(&f.back(&contour)), tol.max(1.0 / f.s));
    if !is_simple(&simple) {
        return fail("merge: the outline is not a simple polygon");
    }
    if !all.iter().all(|&q| covers(&simple, q, 1.5 / f.s)) {
        return fail("merge: the outline does not cover every input corner");
    }
    Ok(simple)
}

/// another region's test points: its centre and 8 points along each edge of its convex hull (a hull edge may cut through
/// a neighbour without taking in a corner)
fn samples(o: &[Pt]) -> Vec<Pt> {
    let h = convex_hull(o);
    let mut out = vec![centroid(o)];
    for i in 0..h.len() {
        let (a, b) = (h[i], h[(i + 1) % h.len()]);
        for k in 0..8 {
            let t = k as f64 / 8.0;
            out.push(Pt::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)));
        }
    }
    out
}

/// MERGE as an enclosing hull: the parts CLOSED with a radius R (grown by R, shrunk by R: the gaps and channels between
/// them filled, no convex bulge) minus every OTHER region (its convex hull, grown by `gap`), the piece holding the
/// parts, its outer edge simplified as coarsely as it stays clear. R steps down from a quarter of the parts' size until
/// no other region is newly enclosed. Fails when none works (the caller takes `union` then). One polygon: its own
/// enclosing hull (Simplify).
pub fn merged_hull(polys: &[Vec<Pt>], others: &[Vec<Pt>], gap: f64, tol: f64) -> Result<Vec<Pt>, Failed> {
    if polys.is_empty() || polys.iter().any(|p| p.len() < 3) {
        return fail("hull: no polygon");
    }
    let all: Vec<Pt> = polys.iter().flatten().copied().collect();
    let pb = bounds(&all);
    let r_max = 0.25 * pb.w.max(pb.h);
    let bb = pb.outset(r_max, r_max);
    let f = Frame::new(bb, 900.0, 4);
    let (w, h) = (f.w, f.h);
    let near: Vec<&Vec<Pt>> = others.iter().filter(|o| bounds(o).intersects(&bb)).collect();
    let parts = f.filled(polys);
    let blocked = f.blocked(&near, gap);
    let d_parts = parts.distance();
    let base = tol.max(1.0 / f.s);
    let mut why = "no hull clear of the other regions";
    let mut r = r_max;
    while r * f.s >= 2.0 {
        let rr = (r * f.s) as f32;
        let grown = Mask { w, h, px: d_parts.iter().map(|&d| d <= rr).collect() };
        let outside = Mask { w, h, px: grown.px.iter().map(|&b| !b).collect() };
        let d_out = outside.distance();
        let m = Mask { w, h, px: (0..w * h).map(|k| parts.px[k] || (d_out[k] > rr && !blocked.px[k])).collect() };
        // the piece holding the parts
        let (label, _) = m.components();
        let mut ids: Vec<i64> = polys.iter().map(|p| f.label_at(&m, &label, centroid(p))).collect();
        ids.sort();
        ids.dedup();
        r *= 0.6;
        if ids.len() != 1 || ids[0] < 0 {
            why = "another region separates the parts";
            continue;
        }
        let keep = ids[0];
        let piece = Mask { w, h, px: label.iter().map(|&l| l == keep).collect() }.dilate();
        let contour = piece.trace();
        if contour.len() < 3 {
            continue;
        }
        let back = f.back(&contour);
        // no point of another region taken in that lies farther than the margin + base tolerance from the parts
        // (touching and overlapping neighbours stay as they were)
        let lim = ((gap + base) * f.s) as f32;
        let clear = |cand: &[Pt]| {
            !near.iter().any(|o| {
                samples(o).iter().any(|&q| {
                    if !contains(cand, q) {
                        return false;
                    }
                    let c = f.to_r(q);
                    let (x, y) = (c.x as i64, c.y as i64);
                    if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                        return true;
                    }
                    d_parts[y as usize * w + x as usize] > lim
                })
            })
        };
        // simplified as coarsely as it stays simple, covers the parts and clear of the others
        let mut best: Option<Vec<Pt>> = None;
        let mut t = base;
        for _ in 0..3 {
            let cand = simplify_outward(&collinear_free(&back), t);
            if !is_simple(&cand) || !all.iter().all(|&q| covers(&cand, q, (1.5 / f.s).max(0.5 * gap))) || !clear(&cand) {
                break;
            }
            best = Some(cand);
            t *= 2.0;
        }
        if let Some(out) = best {
            return Ok(out);
        }
        why = "another region would lie inside";
    }
    fail(&format!("merge hull: {why}"))
}

/// SNAP TO THE BORDER: each side of `poly` whose strip to the page edge — across the region's whole width / height —
/// holds no other region (its convex hull + `gap`) is run out to the edge: every column / row of the region filled from
/// the edge to the region. A side with anything in between stays. → the outline simplified (`tol`); `poly` unchanged
/// when no side is clear.
pub fn snap_to_border(poly: &[Pt], others: &[Vec<Pt>], border: Rect, gap: f64, tol: f64) -> Vec<Pt> {
    if poly.len() < 3 || border.w <= 0.0 || border.h <= 0.0 {
        return poly.to_vec();
    }
    let f = Frame::new(border, 900.0, 0);
    let (w, h) = (f.w, f.h);
    let mut m = f.filled(&[poly.to_vec()]);
    let all: Vec<&Vec<Pt>> = others.iter().filter(|o| o.len() >= 3).collect();
    let blocked = f.blocked(&all, gap);
    let (mut x0, mut y0, mut x1, mut y1) = (w as i64, h as i64, -1i64, -1i64);
    for y in 0..h {
        for x in 0..w {
            if m.px[y * w + x] {
                x0 = x0.min(x as i64);
                x1 = x1.max(x as i64);
                y0 = y0.min(y as i64);
                y1 = y1.max(y as i64);
            }
        }
    }
    if x1 < x0 || y1 < y0 {
        return poly.to_vec();
    }
    let (x0, y0, x1, y1) = (x0 as usize, y0 as usize, x1 as usize, y1 as usize);
    let free = |m: &Mask, xa: usize, xb: usize, ya: usize, yb: usize| {
        (ya..=yb).all(|y| (xa..=xb).all(|x| !blocked.px[y * w + x] || m.px[y * w + x]))
    };
    let mut snapped = false;
    if y0 > 0 && free(&m, x0, x1, 0, y0) {
        // top
        snapped = true;
        for x in x0..=x1 {
            if let Some(first) = (0..h).find(|&y| m.px[y * w + x]) {
                for y in 0..first {
                    m.px[y * w + x] = true;
                }
            }
        }
    }
    if y1 < h - 1 && free(&m, x0, x1, y1, h - 1) {
        // bottom
        snapped = true;
        for x in x0..=x1 {
            if let Some(last) = (0..h).rev().find(|&y| m.px[y * w + x]) {
                for y in (last + 1)..h {
                    m.px[y * w + x] = true;
                }
            }
        }
    }
    if x0 > 0 && free(&m, 0, x0, y0, y1) {
        // left
        snapped = true;
        for y in 0..h {
            if let Some(first) = (0..w).find(|&x| m.px[y * w + x]) {
                for x in 0..first {
                    m.px[y * w + x] = true;
                }
            }
        }
    }
    if x1 < w - 1 && free(&m, x1, w - 1, y0, y1) {
        // right
        snapped = true;
        for y in 0..h {
            if let Some(last) = (0..w).rev().find(|&x| m.px[y * w + x]) {
                for x in (last + 1)..w {
                    m.px[y * w + x] = true;
                }
            }
        }
    }
    if !snapped {
        return poly.to_vec();
    }
    let contour = m.trace();
    if contour.len() < 3 {
        return poly.to_vec();
    }
    let clamp = |q: Pt| Pt::new(q.x.clamp(border.x, border.max_x()), q.y.clamp(border.y, border.max_y()));
    let back: Vec<Pt> = f.back(&contour).into_iter().map(clamp).collect();
    let out: Vec<Pt> = simplify_outward(&collinear_free(&back), tol.max(1.0 / f.s)).into_iter().map(clamp).collect();
    if out.len() < 3 || !is_simple(&out) || !poly.iter().all(|&q| covers(&out, q, 2.0 / f.s)) {
        return poly.to_vec();
    }
    out
}

/// `poly` with every OTHER region cut out exactly — the part `keep` (the region before) already held stays; the piece
/// holding `keep`, traced, simplified with `tol`. `poly` when nothing is cut.
pub fn clear_of_others(poly: &[Pt], keep: &[Vec<Pt>], others: &[Vec<Pt>], tol: f64) -> Vec<Pt> {
    let mut pts = poly.to_vec();
    pts.extend(keep.iter().flatten());
    let bb = bounds(&pts).outset(4.0, 4.0);
    let near: Vec<Vec<Pt>> = others.iter().filter(|o| o.len() >= 3 && bounds(o).intersects(&bb)).cloned().collect();
    if near.is_empty() || poly.len() < 3 || keep.is_empty() || keep.iter().any(|k| k.len() < 3) {
        return poly.to_vec();
    }
    let f = Frame::new(bb, 900.0, 1);
    let (w, h) = (f.w, f.h);
    let m0 = f.filled(&[poly.to_vec()]);
    let k = f.filled(keep);
    let o = f.filled(&near);
    let m = Mask { w, h, px: (0..w * h).map(|i| m0.px[i] && (!o.px[i] || k.px[i])).collect() };
    if m == m0 {
        return poly.to_vec();
    }
    // the piece holding `keep`
    let (label, _) = m.components();
    let keep_label = f.label_at(&m, &label, centroid(&keep[0]));
    if keep_label < 0 {
        return poly.to_vec();
    }
    let contour = Mask { w, h, px: label.iter().map(|&l| l == keep_label).collect() }.trace();
    if contour.len() < 3 {
        return poly.to_vec();
    }
    let out = simplify_outward(&collinear_free(&f.back(&contour)), tol.max(1.0 / f.s));
    if out.len() < 3 || !is_simple(&out) {
        return poly.to_vec();
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Top,
    Right,
    Bottom,
}

/// STRAIGHT SIDES: per side, every row (left / right) or column (top / bottom) of the region is filled out to the
/// region's outermost point on that side — the side becomes one straight line at the widest extent, the region only
/// grows. Sides applied in left, top, right, bottom order; the outline traced again, simplified with `tol`. None when no
/// simple outline comes out.
pub fn straighten_sides(poly: &[Pt], sides: &[Side], tol: f64) -> Option<Vec<Pt>> {
    if poly.len() < 3 || sides.is_empty() {
        return None;
    }
    let f = Frame::new(bounds(poly), 900.0, 3);
    let (w, h) = (f.w, f.h);
    let mut m = f.filled(&[poly.to_vec()]);
    for side in [Side::Left, Side::Top, Side::Right, Side::Bottom] {
        if !sides.contains(&side) {
            continue;
        }
        // per row (left / right) or column (top / bottom) with ink: its index, its first and its last set pixel
        let lines: Vec<(usize, usize, usize)> = match side {
            Side::Left | Side::Right => (0..h)
                .filter_map(|y| Some((y, (0..w).find(|&x| m.px[y * w + x])?, (0..w).rev().find(|&x| m.px[y * w + x])?)))
                .collect(),
            Side::Top | Side::Bottom => (0..w)
                .filter_map(|x| Some((x, (0..h).find(|&y| m.px[y * w + x])?, (0..h).rev().find(|&y| m.px[y * w + x])?)))
                .collect(),
        };
        let at = |line: usize, k: usize| if matches!(side, Side::Left | Side::Right) { line * w + k } else { k * w + line };
        match side {
            Side::Left | Side::Top => {
                let Some(edge) = lines.iter().map(|l| l.1).min() else {
                    continue;
                };
                for &(line, first, _) in &lines {
                    for k in edge..first {
                        m.px[at(line, k)] = true;
                    }
                }
            }
            Side::Right | Side::Bottom => {
                let Some(edge) = lines.iter().map(|l| l.2).max() else {
                    continue;
                };
                for &(line, _, last) in &lines {
                    for k in (last + 1)..=edge {
                        m.px[at(line, k)] = true;
                    }
                }
            }
        }
    }
    let contour = m.trace();
    if contour.len() < 3 {
        return None;
    }
    let out = simplify_outward(&collinear_free(&f.back(&contour)), tol.max(1.0 / f.s));
    if out.len() < 3 || !is_simple(&out) || !poly.iter().all(|&q| covers(&out, q, 2.0 / f.s)) {
        return None;
    }
    Some(out)
}

/// `poly` minus `cutter` grown by `gap`: the region gives way along the cutter's edge. Its largest remaining piece,
/// traced and simplified; `poly` when nothing overlaps; None when nothing is left.
pub fn subtract(poly: &[Pt], cutter: &[Pt], gap: f64, tol: f64) -> Option<Vec<Pt>> {
    if poly.len() < 3 || cutter.len() < 3 {
        return Some(poly.to_vec());
    }
    let (pb, cb) = (bounds(poly), bounds(cutter).outset(gap, gap));
    if !pb.intersects(&cb) {
        return Some(poly.to_vec());
    }
    let f = Frame::new(pb.outset(2.0, 2.0), 900.0, 2);
    let (w, h) = (f.w, f.h);
    let a = f.filled(&[poly.to_vec()]);
    let mut c = f.filled(&[cutter.to_vec()]);
    c.stroke_closed(&f.all_to_r(cutter), (2.0 * gap * f.s).max(1.0));
    let m = Mask { w, h, px: (0..w * h).map(|k| a.px[k] && !c.px[k]).collect() };
    if m == a {
        return Some(poly.to_vec());
    }
    let (label, count) = m.components();
    if count == 0 {
        return None;
    }
    let mut size = vec![0usize; count];
    for &l in &label {
        if l >= 0 {
            size[l as usize] += 1;
        }
    }
    let keep = (0..count).max_by_key(|&i| size[i])? as i64;
    let contour = Mask { w, h, px: label.iter().map(|&l| l == keep).collect() }.trace();
    if contour.len() < 3 {
        return None;
    }
    // simplified OUTWARD by at most half the gap — the growth stays inside the gap, never back over the cutter
    let out = simplify_outward(&collinear_free(&f.back(&contour)), (1.0 / f.s).max(tol.min(0.5 * gap)));
    if out.len() >= 3 && is_simple(&out) { Some(out) } else { None }
}
