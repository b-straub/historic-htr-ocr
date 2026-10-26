// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! The layout tools as the editor applies them (the desktop GT tools' region editing, ported): every tool takes the
//! region's polygon and the OTHER regions' polygons (crop px) and gives the new polygon(s) — never into a neighbour.
//! `tol` = the simplify tolerance in crop px (SIMPLIFY_BAKE_PX at the bake's scale).

use crate::geom::{self, Failed, Pt, Rect, bounds, is_simple, simplify_outward};
use crate::ink::{self, Page};
use crate::scale::{self, Decision};
use crate::shape::{self, Side};

/// the simplify tolerance in crop px at the bake's scale `bake_scale` (bake px per crop px)
pub fn tolerance(bake_scale: f64) -> f64 {
    geom::SIMPLIFY_BAKE_PX / bake_scale
}

/// the EDITED region gives way: minus every other region grown by 2 × tol (its largest piece); the neighbours stay as
/// they are → (the polygon, how many neighbours it gave way to)
pub fn yield_to_neighbours(poly: &[Pt], others: &[Vec<Pt>], tol: f64) -> (Vec<Pt>, usize) {
    let mut p = poly.to_vec();
    let mut n = 0;
    for o in others {
        if let Some(r) = shape::subtract(&p, o, 2.0 * tol, tol) {
            if r != p {
                p = r;
                n += 1;
            }
        }
    }
    (p, n)
}

/// STRAIGHTEN between corners a and b, then give way → (polygon, neighbours given way to)
pub fn straighten(poly: &[Pt], a: usize, b: usize, others: &[Vec<Pt>], tol: f64) -> Result<(Vec<Pt>, usize), Failed> {
    let r = geom::straighten(poly, a, b).ok_or_else(|| Failed(format!("no clean outline between corners {} and {} (neighbours, or it would cross itself)", a + 1, b + 1)))?;
    Ok(yield_to_neighbours(&r, others, tol))
}

/// STRAIGHT SIDES, then give way → (polygon, neighbours given way to)
pub fn sides(poly: &[Pt], sides: &[Side], others: &[Vec<Pt>], tol: f64) -> Result<(Vec<Pt>, usize), Failed> {
    let r = shape::straighten_sides(poly, sides, tol).ok_or_else(|| Failed("no clean outline came out".into()))?;
    Ok(yield_to_neighbours(&r, others, tol))
}

/// MERGE: an enclosing hull clear of the other regions (few corners); the traced union when no hull works → (polygon,
/// how: "hull" or "union (why the hull failed)")
pub fn merge(parts: &[Vec<Pt>], others: &[Vec<Pt>], tol: f64) -> Result<(Vec<Pt>, String), Failed> {
    if parts.len() < 2 {
        return Err(Failed("select two or more regions".into()));
    }
    match shape::merged_hull(parts, others, 2.0 * tol, tol) {
        Ok(h) => Ok((shape::clear_of_others(&h, parts, others, tol), "hull".into())),
        Err(e) => Ok((shape::union(parts, tol)?, format!("union ({e})"))),
    }
}

/// SIMPLIFY the selected region: fewer corners; its enclosing hull when that has fewer, run out to the page edge where
/// nothing lies between, never into a neighbour
pub fn simplify(poly: &[Pt], others: &[Vec<Pt>], page: Rect, tol: f64) -> Vec<Pt> {
    let mut q = simplify_outward(poly, tol);
    if let Ok(h) = shape::merged_hull(&[poly.to_vec()], others, 2.0 * tol, tol) {
        if h.len() < q.len() {
            q = h;
        }
    }
    q = shape::snap_to_border(&q, others, page, 2.0 * tol, tol);
    shape::clear_of_others(&q, &[poly.to_vec()], others, tol)
}

/// the cuts a knife line makes: per region it crosses cleanly (the selected one first when it cuts that) its index, the
/// line clipped to its outline and the two parts (upper / left first); Err = why nothing is cut
pub fn cuts(regions: &[Vec<Pt>], line: &[Pt], selected: Option<usize>, snap: f64) -> Result<Vec<(usize, Vec<Pt>, (Vec<Pt>, Vec<Pt>))>, Failed> {
    if line.len() < 2 {
        return Err(Failed("draw a line across a region".into()));
    }
    let cut = |targets: Vec<usize>| -> Vec<(usize, Vec<Pt>, (Vec<Pt>, Vec<Pt>))> {
        targets
            .into_iter()
            .filter_map(|i| {
                let l = geom::clip_cut(&regions[i], line)?;
                let parts = geom::split(&regions[i], &l, snap.max(1.0)).ok()?;
                Some((i, l, parts))
            })
            .collect()
    };
    if let Some(s) = selected {
        let sel = cut(vec![s]);
        if !sel.is_empty() {
            return Ok(sel);
        }
    }
    let all = cut((0..regions.len()).collect());
    if !all.is_empty() {
        return Ok(all);
    }
    for r in regions {
        if let Some(l) = geom::clip_cut(r, line) {
            if let Err(e) = geom::split(r, &l, snap.max(1.0)) {
                return Err(e);
            }
        }
    }
    Err(Failed("the line crosses no region — draw it across one".into()))
}

/// SNAP TO INK: the drawn region's text blobs → their hull (one piece: simplified outward; several: their enclosing hull
/// clear of the others), never into a neighbour. `drawn` / `others` in crop px; the page analysed on the scaled crop
/// (`fx`, `fy` = scaled px per crop px) → (polygon, blobs)
pub fn snap(pg: &Page, text: &[bool], drawn: &[Pt], others: &[Vec<Pt>], fx: f64, fy: f64, tol: f64) -> Result<(Vec<Pt>, usize), Failed> {
    let scaled: Vec<Pt> = drawn.iter().map(|p| Pt::new(p.x * fx, p.y * fy)).collect();
    let w = ink::wrap(pg, text, &scaled);
    let pieces: Vec<Vec<Pt>> = w.pieces.iter().map(|q| q.iter().map(|p| Pt::new(p.x / fx, p.y / fy)).collect::<Vec<Pt>>()).filter(|q| q.len() >= 3).collect();
    if pieces.is_empty() {
        return Err(Failed("no text ink in the drawn shape — left as drawn".into()));
    }
    let poly = if pieces.len() == 1 { simplify_outward(&pieces[0], tol) } else { shape::merged_hull(&pieces, others, 2.0 * tol, tol)? };
    let poly = shape::clear_of_others(&poly, &pieces, others, tol);
    if poly.len() < 3 || !is_simple(&poly) {
        return Err(Failed("no clean outline after the hull / clearing — left as drawn".into()));
    }
    Ok((poly, w.blobs))
}

/// a region's box (for the reading order)
pub fn region_box(poly: &[Pt]) -> [f64; 4] {
    let b = bounds(poly);
    [b.x, b.y, b.max_x(), b.max_y()]
}

/// a page's writing size (bake px): its glyph height (ink::Page::xh — the median height of its text blobs), its line
/// pitch, and the box [x, y, w, h] of one text blob of that height, the one nearest the page's centre (shown beside the
/// target, so the measure can be checked against the letters)
pub struct Glyphs {
    pub xh: f64,
    pub pitch: f64,
    pub sample: [f64; 4],
}

/// a plausible measure: the glyph height at most this share of the line pitch (pages: 0.38 – 0.64; a sticky note's
/// letters, far beyond the pitch search's range: 1.05) … — UNMEASURED beyond four test images
const XH_PITCH_MAX: f64 = 0.9;
/// … and at least this share of the ink in the text blobs (pages ≥ 0.94; the note: 0.44 — its letters dropped as too
/// large) — UNMEASURED likewise
const KEPT_INK_MIN: f64 = 0.8;

/// `rgb` = the page at its bake scale (w × h, unpadded) → its writing size. The ink analysis finds line pitches of
/// 12 … 90 px; much larger writing (a few lines filling a photo) is measured on the page halved, then quartered —
/// the first plausible measure counts (XH_PITCH_MAX, KEPT_INK_MIN). Err = no ink / no plausible measure
pub fn page_glyphs(rgb: &[u8], w: usize, h: usize) -> Result<Glyphs, ink::page::NoInk> {
    let full = ink::img::luma_pil_rgb(rgb, w, h);
    let mut why = Vec::new();
    for k in [1.0, 0.5, 0.25] {
        let g = if k == 1.0 { full.clone() } else { ink::img::resize_area(&full, ((w as f64) * k).round() as usize, ((h as f64) * k).round() as usize) };
        let pg = match ink::page_from_gray(g) {
            Ok(pg) => pg,
            Err(e) => {
                why.push(format!("×{k}: {e}"));
                continue;
            }
        };
        let kept = (1..pg.n).filter(|&c| pg.keep[c]).map(|c| pg.st[c].area).sum::<usize>() as f64;
        let ink = pg.ink.px.iter().filter(|&&b| b).count() as f64;
        let (ratio, share) = (pg.xh / pg.pitch, kept / ink.max(1.0));
        if ratio > XH_PITCH_MAX || share < KEPT_INK_MIN {
            why.push(format!("×{k}: glyph / pitch {ratio:.2}, text ink {share:.2}"));
            continue;
        }
        // px of this measure → bake px
        let to_bake = 1.0 / (pg.f * k);
        let (mx, my) = (pg.w() as f64 / 2.0, pg.h() as f64 / 2.0);
        let near = |c: usize| (pg.cx[c] - mx).powi(2) + (pg.cy[c] - my).powi(2);
        let sample = (1..pg.n)
            .filter(|&c| pg.keep[c] && (pg.st[c].h as f64 - pg.xh).abs() < 0.5)
            .min_by(|&a, &b| near(a).total_cmp(&near(b)))
            .map(|c| {
                let s = &pg.st[c];
                [s.x as f64 * to_bake, s.y as f64 * to_bake, s.w as f64 * to_bake, s.h as f64 * to_bake]
            })
            .ok_or_else(|| ink::page::NoInk("no text blob of the median height".into()))?;
        return Ok(Glyphs { xh: pg.xh * to_bake, pitch: pg.pitch * to_bake, sample });
    }
    Err(ink::page::NoInk(format!("no plausible writing size ({})", why.join("; "))))
}

/// NORMALIZE a whole page (OCR Lite's Text size: a photo, a close-up): `rgb` = the page at its bake scale (w × h,
/// unpadded), its GLYPH HEIGHT (page_glyphs) brought to `target_xh` (scale::XH_*), capped by the bake's edge; no ink =
/// no scale (a warning)
pub fn page_scale(rgb: &[u8], w: usize, h: usize, target_xh: f64) -> Decision {
    match page_glyphs(rgb, w, h) {
        Ok(g) => glyph_scale(&g, w, h, target_xh),
        Err(e) => Decision { factor: 1.0, reason: "normalize: no ink".into(), warning: Some(format!("{e} — no scale derived")), own_pitch: None, lines: 0 },
    }
}

/// page_scale's decision from a measure already taken (`g` of the w × h page at its bake scale)
pub fn glyph_scale(g: &Glyphs, w: usize, h: usize, target_xh: f64) -> Decision {
    let cap = scale::LONGEST / (w.max(h) as f64);
    let factor = (target_xh / g.xh).min(cap);
    let capped = if cap < target_xh / g.xh { format!(" (capped: the page reaches the {} px edge)", scale::LONGEST as i64) } else { String::new() };
    Decision {
        factor,
        reason: format!("glyph height {} → {} (target {}; line pitch {}){capped}", g.xh.round() as i64, (g.xh * factor).round() as i64, target_xh, g.pitch.round() as i64),
        warning: None,
        own_pitch: Some(g.pitch),
        lines: 0,
    }
}
