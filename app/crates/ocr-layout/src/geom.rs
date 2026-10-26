// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! Region polygon geometry for the layout tools (crop px, y down) — the desktop GT tools' region geometry, ported 1:1:
//! basics, simplify outward, node editing, the knife (clip cut + split), straighten. The raster steps (merge, hull,
//! sides, clear of others, subtract) are in `shape`.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pt {
    pub x: f64,
    pub y: f64,
}

impl Pt {
    pub const fn new(x: f64, y: f64) -> Pt {
        Pt { x, y }
    }

    pub fn dist(self, o: Pt) -> f64 {
        (self.x - o.x).hypot(self.y - o.y)
    }
}

/// an axis-aligned box: x, y = its top-left corner
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    pub fn max_x(&self) -> f64 {
        self.x + self.w
    }

    pub fn max_y(&self) -> f64 {
        self.y + self.h
    }

    /// grown by dx, dy on every side (negative: shrunk)
    pub fn outset(&self, dx: f64, dy: f64) -> Rect {
        Rect { x: self.x - dx, y: self.y - dy, w: self.w + 2.0 * dx, h: self.h + 2.0 * dy }
    }

    /// the two boxes overlap (an area in common)
    pub fn intersects(&self, o: &Rect) -> bool {
        self.x < o.max_x() && o.x < self.max_x() && self.y < o.max_y() && o.y < self.max_y()
    }
}

/// a failed tool step: why, for the status line
#[derive(Debug, Clone, PartialEq)]
pub struct Failed(pub String);

impl std::fmt::Display for Failed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Failed {}

pub(crate) fn fail<T>(why: &str) -> Result<T, Failed> {
    Err(Failed(why.to_owned()))
}

/// the simplify tolerance in BAKE px (the scaled crop: ≈ ⅕ line pitch on humboldt — a bigger one fills deeper
/// concavities and may reach a neighbour's ink); in crop px it is this / the bake scale
pub const SIMPLIFY_BAKE_PX: f64 = 6.0;

// ── basics ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/// shoelace area (sign = orientation)
pub fn signed_area(p: &[Pt]) -> f64 {
    if p.len() < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for i in 0..p.len() {
        let (q, r) = (p[i], p[(i + 1) % p.len()]);
        a += q.x * r.y - r.x * q.y;
    }
    a / 2.0
}

/// the mean of the corners
pub fn centroid(p: &[Pt]) -> Pt {
    let n = p.len().max(1) as f64;
    Pt::new(p.iter().map(|q| q.x).sum::<f64>() / n, p.iter().map(|q| q.y).sum::<f64>() / n)
}

/// the bounding box (at least 1 × 1); an empty list → zero
pub fn bounds(p: &[Pt]) -> Rect {
    if p.is_empty() {
        return Rect { x: 0.0, y: 0.0, w: 0.0, h: 0.0 };
    }
    let x0 = p.iter().map(|q| q.x).fold(f64::INFINITY, f64::min);
    let x1 = p.iter().map(|q| q.x).fold(f64::NEG_INFINITY, f64::max);
    let y0 = p.iter().map(|q| q.y).fold(f64::INFINITY, f64::min);
    let y1 = p.iter().map(|q| q.y).fold(f64::NEG_INFINITY, f64::max);
    Rect { x: x0, y: y0, w: (x1 - x0).max(1.0), h: (y1 - y0).max(1.0) }
}

/// even-odd point in polygon
pub fn contains(p: &[Pt], q: Pt) -> bool {
    let mut inside = false;
    let mut j = p.len().wrapping_sub(1);
    for i in 0..p.len() {
        let (a, b) = (p[i], p[j]);
        if (a.y > q.y) != (b.y > q.y) && q.x < (b.x - a.x) * (q.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// the closest point of segment a–b to q, with its parameter t ∈ [0, 1]
pub fn closest(a: Pt, b: Pt, q: Pt) -> (Pt, f64) {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let l2 = dx * dx + dy * dy;
    if l2 <= 0.0 {
        return (a, 0.0);
    }
    let t = (((q.x - a.x) * dx + (q.y - a.y) * dy) / l2).clamp(0.0, 1.0);
    (Pt::new(a.x + t * dx, a.y + t * dy), t)
}

/// q inside p or within `tol` of its outline
pub fn covers(p: &[Pt], q: Pt, tol: f64) -> bool {
    if contains(p, q) {
        return true;
    }
    (0..p.len()).any(|i| closest(p[i], p[(i + 1) % p.len()], q).0.dist(q) <= tol)
}

pub(crate) fn cross(o: Pt, a: Pt, b: Pt) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// segments a–b and c–d cross or touch (collinear overlap included)
pub fn segments_intersect(a: Pt, b: Pt, c: Pt, d: Pt) -> bool {
    let (d1, d2, d3, d4) = (cross(c, d, a), cross(c, d, b), cross(a, b, c), cross(a, b, d));
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0)) {
        return true;
    }
    let on = |p: Pt, q: Pt, r: Pt, c: f64| {
        c.abs() < 1e-9
            && p.x.min(q.x) - 1e-9 <= r.x
            && r.x <= p.x.max(q.x) + 1e-9
            && p.y.min(q.y) - 1e-9 <= r.y
            && r.y <= p.y.max(q.y) + 1e-9
    };
    on(c, d, a, d1) || on(c, d, b, d2) || on(a, b, c, d3) || on(a, b, d, d4)
}

/// a valid region polygon: ≥ 3 corners, an area, no two non-adjacent edges meeting
pub fn is_simple(p: &[Pt]) -> bool {
    let n = p.len();
    if n < 3 || signed_area(p).abs() <= 1e-6 {
        return false;
    }
    for i in 0..n {
        let (a, b) = (p[i], p[(i + 1) % n]);
        if a.dist(b) < 1e-9 {
            return false;
        }
        for j in (i + 1)..n {
            if j == i + 1 || (i == 0 && j == n - 1) {
                continue; // adjacent edges share a corner
            }
            if segments_intersect(a, b, p[j], p[(j + 1) % n]) {
                return false;
            }
        }
    }
    true
}

/// the convex hull of points (monotone chain)
pub fn convex_hull(pts: &[Pt]) -> Vec<Pt> {
    let mut p = pts.to_vec();
    p.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    if p.len() < 3 {
        return p;
    }
    let mut lower: Vec<Pt> = Vec::new();
    for &q in &p {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], q) <= 0.0 {
            lower.pop();
        }
        lower.push(q);
    }
    let mut upper: Vec<Pt> = Vec::new();
    for &q in p.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], q) <= 0.0 {
            upper.pop();
        }
        upper.push(q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

// ── simplify (outward) ─────────────────────────────────────────────────────────────────────────────────────────────

/// Douglas–Peucker keep-flags of the open chain p[i0…i1]
fn dp(p: &[Pt], i0: usize, i1: usize, tol: f64, keep: &mut [bool]) {
    if i1 <= i0 + 1 {
        return;
    }
    let (mut far, mut k) = (-1.0, i0);
    for i in (i0 + 1)..i1 {
        let d = closest(p[i0], p[i1], p[i]).0.dist(p[i]);
        if d > far {
            far = d;
            k = i;
        }
    }
    if far > tol {
        keep[k] = true;
        dp(p, i0, k, tol, keep);
        dp(p, k, i1, tol, keep);
    }
}

/// FEWER corners, never INSIDE the original: Douglas–Peucker to `tol`, then every kept edge pushed outward until the
/// original corners it replaced lie inside it (the corners = the intersections of neighbouring pushed edges; a
/// near-straight or far-out corner gets the two edge ends instead). A result that is not simple or does not cover every
/// original corner is redone at half the tolerance; below 0.5 px the original is kept.
pub fn simplify_outward(p: &[Pt], tol: f64) -> Vec<Pt> {
    if p.len() <= 4 || !is_simple(p) {
        return p.to_vec();
    }
    let mut t = tol;
    while t >= 0.5 {
        if let Some(s) = simplify_once(p, t)
            && s.len() < p.len() && is_simple(&s) && p.iter().all(|&q| covers(&s, q, 0.5)) {
                return s;
            }
        t /= 2.0;
    }
    p.to_vec()
}

fn simplify_once(p: &[Pt], tol: f64) -> Option<Vec<Pt>> {
    let n = p.len();
    let mut far = 0;
    for i in 1..n {
        if p[i].dist(p[0]) > p[far].dist(p[0]) {
            far = i;
        }
    }
    if far == 0 {
        return None;
    }
    let mut ring = p.to_vec(); // the closed chain, p[0] at both ends
    ring.push(p[0]);
    let mut keep = vec![false; n + 1];
    keep[0] = true;
    keep[far] = true;
    keep[n] = true;
    dp(&ring, 0, far, tol, &mut keep);
    dp(&ring, far, n, tol, &mut keep);
    let idx: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    if idx.len() < 3 {
        return None;
    }
    let ccw = signed_area(p) > 0.0;
    // per kept edge (idx[k] → idx[k+1]): its outward unit normal and the push that puts the replaced corners inside
    struct Line {
        a: Pt,
        d: Pt,
    }
    let mut lines: Vec<Line> = Vec::new();
    for k in 0..idx.len() {
        let (i0, i1) = (idx[k], if k + 1 < idx.len() { idx[k + 1] } else { n });
        let (a, b) = (ring[i0], ring[i1]);
        let len = a.dist(b).max(1e-9);
        let d = Pt::new((b.x - a.x) / len, (b.y - a.y) / len);
        let nrm = if ccw { Pt::new(d.y, -d.x) } else { Pt::new(-d.y, d.x) }; // outward
        let mut off: f64 = 0.0;
        for q in &ring[i0..=i1] {
            off = off.max((q.x - a.x) * nrm.x + (q.y - a.y) * nrm.y);
        }
        off += 0.05;
        lines.push(Line { a: Pt::new(a.x + nrm.x * off, a.y + nrm.y * off), d });
    }
    let mut out: Vec<Pt> = Vec::new();
    let m = lines.len();
    for k in 0..m {
        let (l0, l1) = (&lines[(k + m - 1) % m], &lines[k]);
        let v = ring[idx[k]];
        let den = l0.d.x * l1.d.y - l0.d.y * l1.d.x;
        let mut corner = None;
        if den.abs() > 0.17 {
            // > ~10°: the two pushed edges meet
            let t = ((l1.a.x - l0.a.x) * l1.d.y - (l1.a.y - l0.a.y) * l1.d.x) / den;
            let c = Pt::new(l0.a.x + t * l0.d.x, l0.a.y + t * l0.d.y);
            if c.dist(v) <= 3.0 * tol + 2.0 {
                corner = Some(c);
            }
        }
        match corner {
            Some(c) => out.push(c),
            None => {
                // the two pushed ends at this corner
                let e0 = v.x * l0.d.x + v.y * l0.d.y - (l0.a.x * l0.d.x + l0.a.y * l0.d.y);
                let e1 = v.x * l1.d.x + v.y * l1.d.y - (l1.a.x * l1.d.x + l1.a.y * l1.d.y);
                out.push(Pt::new(l0.a.x + e0 * l0.d.x, l0.a.y + e0 * l0.d.y));
                out.push(Pt::new(l1.a.x + e1 * l1.d.x, l1.a.y + e1 * l1.d.y));
            }
        }
    }
    Some(dedup(&out))
}

pub(crate) fn dedup(p: &[Pt]) -> Vec<Pt> {
    let mut out: Vec<Pt> = Vec::new();
    for &q in p {
        if let Some(&l) = out.last()
            && l.dist(q) <= 0.01 {
                continue;
            }
        out.push(q);
    }
    if out.len() > 1 && out[0].dist(out[out.len() - 1]) <= 0.01 {
        out.pop();
    }
    out
}

pub(crate) fn collinear_free(p: &[Pt]) -> Vec<Pt> {
    if p.len() <= 3 {
        return p.to_vec();
    }
    let n = p.len();
    let out: Vec<Pt> = (0..n).filter(|&i| cross(p[(i + n - 1) % n], p[i], p[(i + 1) % n]).abs() > 1e-9).map(|i| p[i]).collect();
    if out.len() >= 3 { out } else { p.to_vec() }
}

// ── straighten ─────────────────────────────────────────────────────────────────────────────────────────────────────

/// STRAIGHTEN: the corners strictly between `a` and `b` along the SHORTER way round go; a–b becomes one straight edge,
/// moved parallel OUTWARD until every removed corner lies on or inside it (the region only grows, no ink is cut). None
/// when the result is no simple polygon (or < 3 corners).
pub fn straighten(poly: &[Pt], a: usize, b: usize) -> Option<Vec<Pt>> {
    let n = poly.len();
    if n < 4 || a >= n || b >= n || a == b {
        return None;
    }
    let (fwd, bwd) = ((b + n - a) % n, (a + n - b) % n);
    let (s, e, k) = if fwd <= bwd { (a, b, fwd) } else { (b, a, bwd) }; // walk s → e forward, k steps
    if k < 2 {
        return None; // neighbours: nothing to straighten
    }
    let removed: Vec<Pt> = (1..k).map(|j| poly[(s + j) % n]).collect();
    let (ps, pe) = (poly[s], poly[e]);
    let (dx, dy) = (pe.x - ps.x, pe.y - ps.y);
    let len = dx.hypot(dy).max(1e-9);
    let (mut nx, mut ny) = (-dy / len, dx / len); // a normal of s→e …
    let c = centroid(poly);
    if (c.x - ps.x) * nx + (c.y - ps.y) * ny > 0.0 {
        // … pointing away from the region
        nx = -nx;
        ny = -ny;
    }
    let push = removed.iter().map(|q| (q.x - ps.x) * nx + (q.y - ps.y) * ny).fold(0.0, f64::max);
    let mut out: Vec<Pt> = Vec::new();
    let mut i = e;
    while i != s {
        // the kept way round: e … s
        out.push(poly[i]);
        i = (i + 1) % n;
    }
    out.push(poly[s]);
    // the new edge: s → (s moved) → (e moved) → e; a zero push keeps just s → e
    if push > 0.5 {
        out.push(Pt::new(ps.x + push * nx, ps.y + push * ny));
        out.push(Pt::new(pe.x + push * nx, pe.y + push * ny));
    }
    let res = collinear_free(&out);
    if res.len() >= 3 && is_simple(&res) { Some(res) } else { None }
}

// ── node editing (insert on an edge, delete, drag — polygons and the cut line alike) ─────────────────────────────

/// `q` inserted on the nearest edge (within `tol`) of a polygon (`closed`) or polyline → (points, the new index); None =
/// no edge that close
pub fn insert_node(p: &[Pt], q: Pt, tol: f64, closed: bool) -> Option<(Vec<Pt>, usize)> {
    let m = if closed { p.len() } else { p.len().saturating_sub(1) };
    if m < 1 {
        return None;
    }
    let (mut best, mut bd, mut bp) = (None, f64::INFINITY, Pt::new(0.0, 0.0));
    for i in 0..m {
        let (c, t) = closest(p[i], p[(i + 1) % p.len()], q);
        let d = c.dist(q);
        if d < bd && t > 0.001 && t < 0.999 {
            bd = d;
            best = Some(i);
            bp = c;
        }
    }
    let i = best?;
    if bd > tol {
        return None;
    }
    let mut out = p.to_vec();
    out.insert(i + 1, bp);
    Some((out, i + 1))
}

/// node `k` removed — a polygon keeps ≥ 3 corners and stays simple, a polyline keeps ≥ 2 points
pub fn delete_node(p: &[Pt], k: usize, closed: bool) -> Result<Vec<Pt>, Failed> {
    if k >= p.len() {
        return fail("no such corner");
    }
    if p.len() <= if closed { 3 } else { 2 } {
        return fail(if closed { "a polygon keeps at least 3 corners" } else { "a cut line keeps at least 2 points" });
    }
    let mut out = p.to_vec();
    out.remove(k);
    if closed && !is_simple(&out) {
        return fail("removing this corner would make the outline cross itself");
    }
    Ok(out)
}

/// the index of the node of `p` within `tol` of `q` (nearest), None = none
pub fn node_near(p: &[Pt], q: Pt, tol: f64) -> Option<usize> {
    let (mut best, mut bd) = (None, tol);
    for (i, v) in p.iter().enumerate() {
        if v.dist(q) <= bd {
            bd = v.dist(q);
            best = Some(i);
        }
    }
    best
}

// ── split ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/// a point on the outline: edge index + parameter (+ the point)
#[derive(Clone, Copy, Debug)]
struct EdgePoint {
    edge: usize,
    t: f64,
    p: Pt,
}

/// `q` snapped to the outline of `poly` (a corner within `tol` wins) — None when farther than `tol`
fn snap(poly: &[Pt], q: Pt, tol: f64) -> Option<EdgePoint> {
    for (i, &v) in poly.iter().enumerate() {
        if v.dist(q) <= tol {
            return Some(EdgePoint { edge: i, t: 0.0, p: v });
        }
    }
    let (mut best, mut bd) = (None, f64::INFINITY);
    for i in 0..poly.len() {
        let (c, t) = closest(poly[i], poly[(i + 1) % poly.len()], q);
        let d = c.dist(q);
        if d < bd {
            bd = d;
            best = Some(EdgePoint { edge: i, t, p: c });
        }
    }
    if bd <= tol { best } else { None }
}

/// the outline corners passed walking FORWARD from a to b
fn walk(poly: &[Pt], a: EdgePoint, b: EdgePoint) -> Vec<Pt> {
    let n = poly.len();
    if a.edge == b.edge && b.t >= a.t {
        return Vec::new();
    }
    let mut v = (a.edge + 1) % n;
    let mut out = vec![poly[v]];
    while v != b.edge {
        v = (v + 1) % n;
        out.push(poly[v]);
    }
    out
}

/// A KNIFE line clipped to `poly`: the line may start and end anywhere — an end OUTSIDE is cut back to the line's
/// crossing with the outline, an end INSIDE is extended straight on (its end segment's direction) to the next crossing;
/// the points between stay. None when it misses the region or an extended end never meets the outline.
pub fn clip_cut(poly: &[Pt], line: &[Pt]) -> Option<Vec<Pt>> {
    if poly.len() < 3 || line.len() < 2 {
        return None;
    }
    let n = poly.len();
    let unit = |a: Pt, b: Pt| {
        let l = a.dist(b).max(1e-9);
        Pt::new((b.x - a.x) / l, (b.y - a.y) / l)
    };
    // the line with both ends extended far: parameter s = segment index + t; segment 0 and the last are the
    // extensions, the drawn line runs s ∈ [1, line.len()]
    let far = 1e5;
    let last = line.len() - 1;
    let (u0, u1) = (unit(line[1], line[0]), unit(line[last - 1], line[last]));
    let mut ext = vec![Pt::new(line[0].x + far * u0.x, line[0].y + far * u0.y)];
    ext.extend_from_slice(line);
    ext.push(Pt::new(line[last].x + far * u1.x, line[last].y + far * u1.y));
    let mut hits: Vec<(f64, Pt)> = Vec::new();
    for k in 0..(ext.len() - 1) {
        let (a, b) = (ext[k], ext[k + 1]);
        let r = Pt::new(b.x - a.x, b.y - a.y);
        for e in 0..n {
            let (c, d) = (poly[e], poly[(e + 1) % n]);
            let q = Pt::new(d.x - c.x, d.y - c.y);
            let den = r.x * q.y - r.y * q.x;
            if den.abs() <= 1e-9 {
                continue;
            }
            let t = ((c.x - a.x) * q.y - (c.y - a.y) * q.x) / den;
            let u = ((c.x - a.x) * r.y - (c.y - a.y) * r.x) / den;
            if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                hits.push((k as f64 + t, Pt::new(a.x + t * r.x, a.y + t * r.y)));
            }
        }
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (s0, s1) = (1.0, line.len() as f64); // the drawn line's first and last point
    // entry: an end inside → the last crossing before it (on the extension); outside → the first crossing after it
    let eps = 1e-6;
    let entry = if contains(poly, line[0]) {
        hits.iter().rev().find(|h| h.0 < s0 + eps)
    } else {
        hits.iter().find(|h| h.0 >= s0 && h.0 <= s1)
    };
    let exit = if contains(poly, line[last]) {
        hits.iter().find(|h| h.0 > s1 - eps)
    } else {
        hits.iter().rev().find(|h| h.0 >= s0 && h.0 <= s1)
    };
    let (a, b) = (entry?, exit?);
    if b.0 - a.0 <= 1e-6 {
        return None;
    }
    let mut path = vec![a.1];
    for (k, &q) in line.iter().enumerate() {
        let s = (k + 1) as f64;
        if s > a.0 && s < b.0 {
            path.push(q);
        }
    }
    path.push(b.1);
    Some(path)
}

/// Cut `poly` along `line` (its first and last point on the outline — snapped within `tol`; the rest inside). The line
/// must enter and leave the polygon exactly once. → the two parts, the upper / left one first.
pub fn split(poly: &[Pt], line: &[Pt], tol: f64) -> Result<(Vec<Pt>, Vec<Pt>), Failed> {
    if poly.len() < 3 || !is_simple(poly) {
        return fail("split: the region is no simple polygon");
    }
    if line.len() < 2 {
        return fail("split: draw a line with at least two points");
    }
    let Some(a) = snap(poly, line[0], tol) else {
        return fail("split: the line must START on the region's outline");
    };
    let Some(b) = snap(poly, line[line.len() - 1], tol) else {
        return fail("split: the line must END on the region's outline");
    };
    if a.p.dist(b.p) <= 1.0 {
        return fail("split: start and end are the same outline point");
    }
    let inner = &line[1..line.len() - 1];
    if inner.iter().any(|&q| !contains(poly, q)) {
        return fail("split: the line leaves the region between its ends");
    }
    let mut path = vec![a.p];
    path.extend_from_slice(inner);
    path.push(b.p);
    let n = poly.len();
    for k in 0..(path.len() - 1) {
        let (p0, p1) = (path[k], path[k + 1]);
        if !contains(poly, Pt::new((p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0)) {
            return fail("split: the line runs outside the region");
        }
        for e in 0..n {
            let (c0, c1) = (poly[e], poly[(e + 1) % n]);
            if !segments_intersect(p0, p1, c0, c1) {
                continue;
            }
            // the line's own ends on the outline are allowed — anything else crosses it again
            let p0_on = closest(c0, c1, p0).0.dist(p0) < 1e-6;
            let p1_on = closest(c0, c1, p1).0.dist(p1) < 1e-6;
            if p0_on && p1_on {
                return fail("split: the line runs along the outline");
            }
            if (k == 0 && p0_on) || (k == path.len() - 2 && p1_on) {
                continue;
            }
            return fail("split: the line crosses the outline again — it must enter and leave exactly once");
        }
    }
    let mut one = vec![a.p];
    one.extend(walk(poly, a, b));
    one.push(b.p);
    one.extend(inner.iter().rev());
    let mut two = vec![b.p];
    two.extend(walk(poly, b, a));
    two.push(a.p);
    two.extend_from_slice(inner);
    let (mut p1, mut p2) = (dedup(&one), dedup(&two));
    if !is_simple(&p1) || !is_simple(&p2) {
        return fail("split: a part would not be a valid polygon");
    }
    let (c1, c2) = (centroid(&p1), centroid(&p2));
    if c2.y < c1.y - 1.0 || ((c2.y - c1.y).abs() <= 1.0 && c2.x < c1.x) {
        std::mem::swap(&mut p1, &mut p2);
    }
    Ok((p1, p2))
}
