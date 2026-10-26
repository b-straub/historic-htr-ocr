// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! GT-free region TYPES + READING ORDER from geometry (the desktop tools' layout order, ported): types relative to the
//! page's MAIN column(s) (column groups — x-ranges overlapping ≥ half of the narrower — that hold a BIG region):
//! body, heading, insertion, margin-left / -right, mTop / mBottom, page-number; the ORDER by a preset (PRESETS — data;
//! the first is the default: the Escher convention, each margin note right after the paragraph it sits next to; "auto"
//! picks multi-column only when the geometry clearly shows ≥ 2 body columns).

const BIG_AREA: f64 = 0.25;
const HEAD_LINES: f64 = 2.5;
const HEAD_GAP: f64 = 4.0;
const INS_AREA: f64 = 0.15;
const PN_PITCH: f64 = 1.8;
const PN_WIDTH: f64 = 5.0;
const PN_EDGE: f64 = 0.12;
const PITCH_SHARE: f64 = 0.02;

/// where the margin notes go
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Notes {
    /// each note after the body paragraph it overlaps most (or, `same_column`, the nearest column's)
    AfterAnchor { same_column: bool },
    /// the main text, then the notes per side (left, then right)
    AfterMainBySide,
    /// the main text, then every note top → bottom
    AfterMain,
}

/// the "auto" preset's choice
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Auto {
    pub multi: &'static str,
    pub otherwise: &'static str,
    pub min_gutter_pitch: f64,
    pub min_bodies_per_column: usize,
    pub min_height_ratio: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Preset {
    pub id: &'static str,
    pub label: &'static str,
    pub notes: Option<Notes>,
    /// a note / bottom block starting in this bottom share of the page = the page foot (after everything)
    pub foot_share: Option<f64>,
    pub auto: Option<Auto>,
}

/// the presets — the first is the default
pub const PRESETS: &[Preset] = &[
    Preset {
        id: "escher.insert.side.open",
        label: "Escher: L→R columns, T→B, each note after the paragraph it sits next to (default)",
        notes: Some(Notes::AfterAnchor { same_column: false }),
        foot_share: Some(0.15),
        auto: None,
    },
    Preset {
        id: "auto",
        label: "Auto: multi-column when the geometry clearly shows ≥ 2 body columns, else the Escher default",
        notes: None,
        foot_share: None,
        auto: Some(Auto { multi: "columns.multi", otherwise: "escher.insert.side.open", min_gutter_pitch: 1.0, min_bodies_per_column: 2, min_height_ratio: 0.6 }),
    },
    Preset {
        id: "columns.multi",
        label: "Multi-column: column 1 fully, then column 2 …, each column's notes anchored inside it",
        notes: Some(Notes::AfterAnchor { same_column: true }),
        foot_share: Some(0.15),
        auto: None,
    },
    Preset {
        id: "columns.first",
        label: "Columns first: the main text fully, then the notes per side (left, then right)",
        notes: Some(Notes::AfterMainBySide),
        foot_share: Some(0.15),
        auto: None,
    },
    Preset { id: "notes.end", label: "Notes at the end: the main text, then every note top → bottom", notes: Some(Notes::AfterMain), foot_share: None, auto: None },
];

pub fn preset(id: Option<&str>) -> Result<&'static Preset, crate::Failed> {
    match id {
        None => Ok(&PRESETS[0]),
        Some(id) => PRESETS.iter().find(|p| p.id == id).ok_or_else(|| crate::Failed(format!("unknown order preset {id}"))),
    }
}

/// a region's box: x0, y0, x1, y1
pub type BoxF = [f64; 4];

fn w(b: &BoxF) -> f64 {
    b[2] - b[0]
}

fn h(b: &BoxF) -> f64 {
    b[3] - b[1]
}

fn area(b: &BoxF) -> f64 {
    w(b).max(0.0) * h(b).max(0.0)
}

/// column groups of `idx` (x-ranges overlapping ≥ half of the narrower, transitively), left → right
fn columns(boxes: &[BoxF], idx: &[usize]) -> Vec<Vec<usize>> {
    let mut parent: std::collections::BTreeMap<usize, usize> = idx.iter().map(|&i| (i, i)).collect();
    fn find(parent: &mut std::collections::BTreeMap<usize, usize>, mut i: usize) -> usize {
        while parent[&i] != i {
            let gp = parent[&parent[&i]];
            parent.insert(i, gp);
            i = gp;
        }
        i
    }
    for &a in idx {
        for &b in idx {
            if a < b {
                let (aa, bb) = (&boxes[a], &boxes[b]);
                let ov = aa[2].min(bb[2]) - aa[0].max(bb[0]);
                if ov >= 0.5 * w(aa).min(w(bb)) {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    parent.insert(ra, rb);
                }
            }
        }
    }
    // groups in the order their root first appears (as a dict keeps insertion order)
    let mut roots: Vec<usize> = Vec::new();
    let mut cols: Vec<Vec<usize>> = Vec::new();
    for &i in idx {
        let r = find(&mut parent, i);
        match roots.iter().position(|&x| x == r) {
            Some(k) => cols[k].push(i),
            None => {
                roots.push(r);
                cols.push(vec![i]);
            }
        }
    }
    let left = |c: &Vec<usize>| c.iter().map(|&i| boxes[i][0]).fold(f64::INFINITY, f64::min);
    cols.sort_by(|a, b| left(a).total_cmp(&left(b)));
    cols
}

fn is_page_number_text(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_whitespace() || c.is_ascii_digit() || ".,;:()[]/-–—°'\"*".contains(c))
}

/// the types of the region boxes (a page `width` × `height`, line `pitch` or 2 % of the height; `reads`: a page number
/// must read as digits)
pub fn classify(boxes: &[BoxF], height: f64, pitch: Option<f64>, reads: &[Option<String>]) -> Vec<&'static str> {
    let n = boxes.len();
    if n == 0 {
        return Vec::new();
    }
    let p = pitch.unwrap_or(PITCH_SHARE * height);
    let biggest = boxes.iter().map(area).fold(0.0, f64::max);
    let all: Vec<usize> = (0..n).collect();
    let cols = columns(boxes, &all);
    let main: Vec<&Vec<usize>> = cols.iter().filter(|c| c.iter().any(|&i| area(&boxes[i]) >= BIG_AREA * biggest)).collect();
    let main_idx: Vec<usize> = main.iter().flat_map(|c| c.iter().copied()).collect();
    let big: Vec<usize> = main_idx.iter().copied().filter(|&i| area(&boxes[i]) >= BIG_AREA * biggest).collect();
    let mx0 = big.iter().map(|&i| boxes[i][0]).fold(f64::INFINITY, f64::min);
    let mx1 = big.iter().map(|&i| boxes[i][2]).fold(f64::NEG_INFINITY, f64::max);
    let my0 = big.iter().map(|&i| boxes[i][1]).fold(f64::INFINITY, f64::min);
    let my1 = big.iter().map(|&i| boxes[i][3]).fold(f64::NEG_INFINITY, f64::max);
    let mut body_areas: Vec<f64> = big.iter().map(|&i| area(&boxes[i])).collect();
    body_areas.sort_by(|a, b| a.total_cmp(b));
    let med = body_areas[body_areas.len() / 2];
    let page_number = |i: usize| {
        let b = &boxes[i];
        let cy = (b[1] + b[3]) / 2.0;
        let near = cy <= PN_EDGE * height || cy >= (1.0 - PN_EDGE) * height;
        let small = h(b) <= PN_PITCH * p && w(b) <= PN_WIDTH * p;
        let text_ok = match reads.get(i) {
            Some(Some(r)) => is_page_number_text(r),
            _ => true,
        };
        near && small && text_ok
    };
    (0..n)
        .map(|i| {
            let b = &boxes[i];
            let cx = (b[0] + b[2]) / 2.0;
            let is_big = big.contains(&i);
            if !is_big && page_number(i) {
                "page-number"
            } else if cx < mx0 {
                "margin-left"
            } else if cx > mx1 {
                "margin-right"
            } else if is_big {
                "body"
            } else if b[3] <= my0 + 0.5 * p {
                let below = big.iter().any(|&j| boxes[j][1] >= b[3] - 0.5 * p && boxes[j][1] - b[3] <= HEAD_GAP * p);
                if h(b) <= HEAD_LINES * p && below { "heading" } else { "mTop" }
            } else if b[1] >= my1 - 0.5 * p {
                "mBottom"
            } else if area(b) <= INS_AREA * med {
                let below = big.iter().any(|&j| boxes[j][1] >= b[3] - 0.25 * p && boxes[j][1] - b[3] <= HEAD_GAP * p)
                    && !big.iter().any(|&k| boxes[k][1] < b[1] && b[1] < boxes[k][3]);
                if below && h(b) <= HEAD_LINES * p { "heading" } else { "insertion" }
            } else {
                "body"
            }
        })
        .collect()
}

/// the "auto" preset's choice: (preset id, why)
fn auto_choice(boxes: &[BoxF], types: &[&str], pitch: f64, a: &Auto) -> (&'static str, String) {
    let body: Vec<usize> = (0..types.len()).filter(|&i| types[i] == "body").collect();
    let cols = if body.is_empty() { Vec::new() } else { columns(boxes, &body) };
    if cols.len() < 2 {
        return (a.otherwise, "one body column".into());
    }
    if let Some(k) = cols.iter().position(|c| c.len() < a.min_bodies_per_column) {
        return (a.otherwise, format!("column {} has {} body region(s) < {}", k + 1, cols[k].len(), a.min_bodies_per_column));
    }
    let xs: Vec<(f64, f64)> = cols
        .iter()
        .map(|c| (c.iter().map(|&i| boxes[i][0]).fold(f64::INFINITY, f64::min), c.iter().map(|&i| boxes[i][2]).fold(f64::NEG_INFINITY, f64::max)))
        .collect();
    let gutter = xs.windows(2).map(|p| p[1].0 - p[0].1).fold(f64::INFINITY, f64::min);
    if gutter < a.min_gutter_pitch * pitch {
        return (a.otherwise, format!("gutter {gutter:.0} px < {:.0}", a.min_gutter_pitch * pitch));
    }
    let hs: Vec<f64> = cols
        .iter()
        .map(|c| c.iter().map(|&i| boxes[i][3]).fold(f64::NEG_INFINITY, f64::max) - c.iter().map(|&i| boxes[i][1]).fold(f64::INFINITY, f64::min))
        .collect();
    let (lo, hi) = (hs.iter().copied().fold(f64::INFINITY, f64::min), hs.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    if lo < a.min_height_ratio * hi {
        return (a.otherwise, format!("column heights {lo:.0} / {hi:.0} px differ"));
    }
    (a.multi, format!("{} columns (gutter {gutter:.0} px)", cols.len()))
}

/// the reading order + types under a preset
#[derive(Debug, Clone, PartialEq)]
pub struct Ordered {
    pub order: Vec<usize>,
    pub types: Vec<&'static str>,
    pub preset: &'static str,
    pub why: String,
}

/// the reading order of the region boxes (a page `width` × `height`) under the preset (None = the default)
pub fn order(boxes: &[BoxF], height: f64, pitch: Option<f64>, reads: &[Option<String>], preset_id: Option<&str>) -> Result<Ordered, crate::Failed> {
    let mut pr = preset(preset_id)?;
    let types = classify(boxes, height, pitch, reads);
    let n = boxes.len();
    let mut why = String::new();
    if let Some(a) = &pr.auto {
        let (chosen, reason) = auto_choice(boxes, &types, pitch.unwrap_or(PITCH_SHARE * height), a);
        pr = preset(Some(chosen))?;
        why = format!("auto → {chosen} ({reason})");
    }
    let Some(notes_rule) = pr.notes else {
        return Err(crate::Failed(format!("preset {}: no notes placement", pr.id)));
    };
    if n == 0 {
        return Ok(Ordered { order: Vec::new(), types, preset: pr.id, why });
    }
    let top_cmp = |a: &usize, b: &usize| boxes[*a][1].total_cmp(&boxes[*b][1]).then(boxes[*a][0].total_cmp(&boxes[*b][0]));
    let in_main: Vec<usize> = (0..n).filter(|&i| matches!(types[i], "body" | "heading" | "insertion")).collect();
    let cols = if in_main.is_empty() { Vec::new() } else { columns(boxes, &in_main) };
    let mut seq: Vec<usize> = Vec::new();
    for c in &cols {
        let mut c = c.clone();
        c.sort_by(top_cmp);
        seq.extend(c);
    }
    let mut notes: Vec<usize> = (0..n).filter(|&i| matches!(types[i], "margin-left" | "margin-right")).collect();
    let mut bottom: Vec<usize> = (0..n).filter(|&i| types[i] == "mBottom").collect();
    bottom.sort_by(top_cmp);
    let mut foot: Vec<usize> = Vec::new();
    if let Some(share) = pr.foot_share {
        let cut = (1.0 - share) * height;
        foot = notes.iter().chain(&bottom).copied().filter(|&i| boxes[i][1] >= cut).collect();
        notes.retain(|i| !foot.contains(i));
        bottom.retain(|i| !foot.contains(i));
    }
    let body_bottom = seq.iter().filter(|&&j| types[j] == "body").map(|&j| boxes[j][3]).fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v)))).unwrap_or(height);
    let below: Vec<usize> = notes.iter().copied().filter(|&i| boxes[i][1] >= body_bottom).collect();
    notes.retain(|i| !below.contains(i));
    bottom.extend(below);
    bottom.sort_by(top_cmp);
    let side_cmp = |a: &usize, b: &usize| {
        boxes[*a][1].total_cmp(&boxes[*b][1]).then((types[*a] != "margin-left").cmp(&(types[*b] != "margin-left"))).then(boxes[*a][0].total_cmp(&boxes[*b][0]))
    };
    let mut out: Vec<usize> = (0..n).filter(|&i| types[i] == "mTop").collect();
    out.sort_by(top_cmp);
    match notes_rule {
        Notes::AfterAnchor { same_column } => {
            let mut after: Vec<(Option<usize>, usize)> = Vec::new();
            for &i in &notes {
                let b = &boxes[i];
                let mut cand: Vec<usize> = seq.clone();
                if same_column {
                    let cx = (b[0] + b[2]) / 2.0;
                    let dist = |c: &Vec<usize>| {
                        let l = c.iter().map(|&k| boxes[k][0]).fold(f64::INFINITY, f64::min);
                        let r = c.iter().map(|&k| boxes[k][2]).fold(f64::NEG_INFINITY, f64::max);
                        0f64.max(l - cx).max(cx - r)
                    };
                    let mut near = &cols[0];
                    for c in &cols {
                        if dist(c) < dist(near) {
                            near = c;
                        }
                    }
                    cand = seq.iter().copied().filter(|j| near.contains(j)).collect();
                }
                let (mut best, mut bv) = (None, 0.0);
                for &j in &cand {
                    if types[j] != "body" {
                        continue;
                    }
                    let ov = b[3].min(boxes[j][3]) - b[1].max(boxes[j][1]);
                    if ov > bv {
                        best = Some(j);
                        bv = ov;
                    }
                }
                if best.is_none() && !seq.is_empty() {
                    let cy = (b[1] + b[3]) / 2.0;
                    let pool = if cand.is_empty() { &seq } else { &cand };
                    let d = |j: usize| ((boxes[j][1] + boxes[j][3]) / 2.0 - cy).abs();
                    let mut k = pool[0];
                    for &j in pool {
                        if d(j) < d(k) {
                            k = j;
                        }
                    }
                    best = Some(k);
                }
                after.push((best, i));
            }
            let of = |a: Option<usize>| {
                let mut v: Vec<usize> = after.iter().filter(|x| x.0 == a).map(|x| x.1).collect();
                v.sort_by(side_cmp);
                v
            };
            for &j in &seq {
                out.push(j);
                out.extend(of(Some(j)));
            }
            out.extend(of(None));
        }
        Notes::AfterMainBySide => {
            out.extend(&seq);
            for side in ["margin-left", "margin-right"] {
                let mut v: Vec<usize> = notes.iter().copied().filter(|&i| types[i] == side).collect();
                v.sort_by(top_cmp);
                out.extend(v);
            }
        }
        Notes::AfterMain => {
            out.extend(&seq);
            let mut v = notes.clone();
            v.sort_by(side_cmp);
            out.extend(v);
        }
    }
    out.extend(bottom);
    foot.sort_by(top_cmp);
    out.extend(foot);
    let mut pn: Vec<usize> = (0..n).filter(|&i| types[i] == "page-number").collect();
    pn.sort_by(top_cmp);
    out.extend(pn);
    let why = if why.is_empty() { pr.id.to_owned() } else { why };
    Ok(Ordered { order: out, types, preset: pr.id, why })
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: f64 = 1400.0;

    fn o(boxes: &[BoxF], preset: Option<&str>) -> Ordered {
        order(boxes, H, Some(28.0), &vec![None; boxes.len()], preset).unwrap()
    }

    /// the desktop tools' layout-order self-test, case for case
    #[test]
    fn two_columns_left_then_right() {
        let r = o(&[[520., 100., 950., 700.], [50., 720., 480., 1300.], [50., 100., 480., 700.], [520., 720., 950., 1300.]], None);
        assert_eq!(r.order, vec![2, 1, 0, 3]);
        assert_eq!(r.types, vec!["body"; 4]);
    }

    #[test]
    fn notes_after_their_body() {
        let r = o(&[[300., 100., 950., 650.], [300., 680., 950., 1300.], [30., 900., 260., 1000.], [30., 150., 260., 260.]], None);
        assert_eq!(r.order, vec![0, 3, 1, 2]);
        assert_eq!(r.types, vec!["body", "body", "margin-left", "margin-left"]);
    }

    #[test]
    fn heading_page_number_bottom_note() {
        let boxes = [[200., 160., 800., 210.], [100., 240., 900., 1200.], [900., 30., 960., 70.], [150., 1250., 700., 1330.]];
        let reads = [None, None, Some("47".to_owned()), None];
        let r = order(&boxes, H, Some(28.0), &reads, None).unwrap();
        assert_eq!(r.types, vec!["heading", "body", "page-number", "mBottom"]);
        assert_eq!(r.order, vec![0, 1, 3, 2]);
        let letters = [None, None, Some("Cumaná".to_owned()), None];
        assert_ne!(order(&boxes, H, Some(28.0), &letters, None).unwrap().types[2], "page-number");
    }

    #[test]
    fn presets_on_notes_left_and_right() {
        let boxes = [[300., 100., 750., 650.], [300., 680., 750., 1100.], [30., 700., 260., 800.], [800., 150., 980., 260.], [30., 200., 260., 300.]];
        assert_eq!(o(&boxes, Some("escher.insert.side.open")).order, vec![0, 3, 4, 1, 2]);
        assert_eq!(o(&boxes, Some("columns.first")).order, vec![0, 1, 4, 2, 3]);
        assert_eq!(o(&boxes, Some("notes.end")).order, vec![0, 1, 3, 4, 2]);
    }

    #[test]
    fn spanning_note_and_foot() {
        let r = o(&[[300., 100., 950., 600.], [300., 620., 950., 1100.], [30., 540., 260., 700.], [30., 1250., 260., 1320.]], None);
        assert_eq!(r.order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn auto_picks_columns_only_when_clear() {
        let two = [[50., 100., 450., 600.], [50., 620., 450., 1250.], [550., 100., 950., 650.], [550., 670., 950., 1250.], [460., 300., 540., 380.]];
        assert_eq!(o(&two, Some("auto")).preset, "columns.multi");
        let one = [[50., 100., 450., 600.], [50., 620., 450., 1250.], [460., 100., 950., 1250.]];
        assert_eq!(o(&one, Some("auto")).preset, "escher.insert.side.open");
    }
}
