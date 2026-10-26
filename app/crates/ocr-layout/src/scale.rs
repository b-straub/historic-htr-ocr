// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! A region's SCALE before the read (the desktop tools' Normalize / Enlarge small text, ported; work in progress there
//! too): its line pitch (bake px, `ink::measure`) brought to a target pitch — the trained bakes' quartiles p25 43 ·
//! median 50.5 · p75 59.5 — up or down, capped only by the bake's 1540 px edge. Two measurements, the page-range pitch
//! and the WIDE one (a large hand); when they differ, the one whose lines FILL the ink's height best; then its lines'
//! core heights must be one hand size, else no scale (a warning). A region with no usable pitch of its own borrows the
//! page's (the other note-sized regions' median). Enlarge small text (auto) only raises a small pitch.

/// the bake's longest edge
pub const LONGEST: f64 = 1540.0;
/// auto: a cut whose line pitch (bake px) is below this is enlarged …
pub const SMALL_PITCH: f64 = 20.0;
/// … until its pitch is about this
pub const TARGET_PITCH: f64 = 50.0;
/// no measurable pitch (one line) on a SMALL page (crop < 1540 before the bake)
pub const NO_PITCH_SMALL_PAGE: f64 = 2.0;
/// a line's core height beyond this ratio of the median is off-size …
const CORE_RATIO: f64 = 2.0;
/// … and this share of the lines off-size = inconsistent (≤ 3 lines: max / min)
const CORE_OFF_SHARE: f64 = 0.25;
/// … or two size groups whose medians lie this far apart
const CORE_SPREAD: f64 = 2.5;
/// a region of at most this many lines is a note (the page's pitch borrowed from the notes first) — UNMEASURED
pub const NOTE_LINES: usize = 5;

/// the target presets (bake px): the trained bakes' line-pitch quartiles; the slider runs 35…70 in steps of 5
pub const TARGET_LOW: f64 = 43.0;
pub const TARGET_MID: f64 = 50.5;
pub const TARGET_HIGH: f64 = 59.5;

/// the WHOLE-PAGE targets (OCR Lite's Text size): the glyph height (ink::Page::xh — the median height of the page's
/// text blobs, bake px) of the trained pages, weighted by each corpus' training pages: p25 18 · median 23 · p75 29
/// (p5 11 · p95 43; the same pages and measure as the pitch quartiles). A page is scaled by its glyph height, not its
/// line pitch: a dense hand (textura) has large letters at an ordinary pitch
pub const XH_SMALLER: f64 = 18.0;
pub const XH_AVERAGE: f64 = 23.0;
pub const XH_LARGER: f64 = 29.0;

/// Text size by name ("smaller" / "average" / "larger") → its glyph height target
pub fn xh_target(name: &str) -> Option<f64> {
    match name {
        "smaller" => Some(XH_SMALLER),
        "average" => Some(XH_AVERAGE),
        "larger" => Some(XH_LARGER),
        _ => None,
    }
}

/// a scale decision
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub factor: f64,
    pub reason: String,
    /// no scale could be derived (no pitch / inconsistent lines) — shown at the region
    pub warning: Option<String>,
    /// the region's OWN usable pitch and its line count — what other regions borrow
    pub own_pitch: Option<f64>,
    pub lines: usize,
}

impl Decision {
    fn plain(factor: f64, reason: String) -> Decision {
        Decision { factor, reason, warning: None, own_pitch: None, lines: 0 }
    }

    /// "enlarged ×2.0 (…)" / "reduced ×0.50 (…)" / "native (…)"
    pub fn summary(&self) -> String {
        if self.factor > 1.0001 {
            format!("enlarged ×{:.1} ({})", self.factor, self.reason)
        } else if self.factor < 0.9999 {
            format!("reduced ×{:.2} ({})", self.factor, self.reason)
        } else {
            format!("native ({})", self.reason)
        }
    }
}

/// a pitch borrowed from the page's other regions: the pitch and where it came from
#[derive(Debug, Clone, PartialEq)]
pub struct Borrowed {
    pub pitch: f64,
    pub from: String,
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// `s` (sorted cores, ≥ 4) splits at its largest jump into two groups of ≥ CORE_OFF_SHARE each whose medians differ by
/// more than CORE_SPREAD
fn two_sizes(s: &[f64]) -> bool {
    let mut k = 1;
    for i in 2..s.len() {
        if s[i] / s[i - 1].max(0.5) > s[k] / s[k - 1].max(0.5) {
            k = i;
        }
    }
    let min_group = CORE_OFF_SHARE * s.len() as f64;
    if (k as f64) < min_group || ((s.len() - k) as f64) < min_group {
        return false;
    }
    median(&s[k..]) > CORE_SPREAD * median(&s[..k]).max(0.5)
}

/// the lines' core heights one hand size: ≤ 3 lines max / min ≤ CORE_RATIO; more: < CORE_OFF_SHARE of them beyond
/// CORE_RATIO of the median and no two size groups
pub fn consistent(cores: &[f64]) -> bool {
    if cores.len() < 2 {
        return true;
    }
    let lo = cores.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = cores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if cores.len() <= 3 {
        return hi <= CORE_RATIO * lo.max(0.5);
    }
    let mut s = cores.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let med = median(&s);
    let off = cores.iter().filter(|&&c| c > CORE_RATIO * med || c * CORE_RATIO < med).count();
    (off as f64) < CORE_OFF_SHARE * cores.len() as f64 && !two_sizes(&s)
}

/// a region's line measures (bake px) — `ink::Measures` in the image's px
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Facts {
    pub pitch: Option<f64>,
    pub core: Vec<f64>,
    pub wide_pitch: Option<f64>,
    pub wide_core: Vec<f64>,
    pub extent: f64,
    pub band_pitch: Option<f64>,
    pub band_core: Vec<f64>,
}

impl From<&crate::ink::Measures> for Facts {
    fn from(m: &crate::ink::Measures) -> Facts {
        Facts {
            pitch: m.pitch,
            core: m.core.clone(),
            wide_pitch: m.wide_pitch,
            wide_core: m.wide_core.clone(),
            extent: m.extent,
            band_pitch: m.band_pitch,
            band_core: m.band_core.clone(),
        }
    }
}

fn g(v: f64) -> String {
    // Swift's %g for the targets: 43, 50.5
    if v.fract() == 0.0 { format!("{}", v as i64) } else { format!("{v}") }
}

/// NORMALIZE a region of longest edge `cut_longest` (bake px) to `target`
pub fn normalize(target: f64, f: &Facts, cut_longest: f64, borrowed: Option<&Borrowed>) -> Decision {
    let cap = LONGEST / cut_longest.max(1.0);
    let borrow = |why: &str| {
        borrowed.map(|b| {
            let f = (target / b.pitch).min(cap);
            let capped = if cap < target / b.pitch { format!(" (capped: the cut reaches the {} px edge)", LONGEST as i64) } else { String::new() };
            Decision::plain(f, format!("target {}: {why} — pitch {} of {} → {}{capped}", g(target), b.pitch.round() as i64, b.from, (b.pitch * f).round() as i64))
        })
    };
    let fill = |p: f64, c: &[f64]| c.len() as f64 / (f.extent / p + 1.0);
    let mut pick: (Option<f64>, &[f64], &str) = (f.pitch, &f.core, "");
    if let Some(w) = f.wide_pitch {
        if (w - f.pitch.unwrap_or(-1.0)).abs() > 0.5 {
            pick = match f.pitch {
                Some(p) if fill(p, &f.core) >= fill(w, &f.wide_core) => (Some(p), &f.core, ""),
                _ => (Some(w), &f.wide_core, ", wide"),
            };
        }
    }
    if pick.0.is_none() {
        if let Some(b) = f.band_pitch {
            pick = (Some(b), &f.band_core, ", lines");
        }
    }
    let Some(p) = pick.0 else {
        if let Some(d) = borrow("no line pitch of its own") {
            return d;
        }
        return Decision {
            warning: Some("no line pitch measurable (one line / too little ink) — no scale derived".into()),
            ..Decision::plain(1.0, "normalize: no pitch".into())
        };
    };
    if !consistent(pick.1) {
        let lo = pick.1.iter().copied().fold(f64::INFINITY, f64::min).round() as i64;
        let hi = pick.1.iter().copied().fold(f64::NEG_INFINITY, f64::max).round() as i64;
        if let Some(d) = borrow(&format!("own lines inconsistent (core {lo}–{hi} px)")) {
            return d;
        }
        return Decision {
            warning: Some(format!("inconsistent line heights (core {lo}–{hi} px at pitch {}{}) — no scale derived from the ink; split the region", p.round() as i64, pick.2)),
            ..Decision::plain(1.0, "normalize: inconsistent lines".into())
        };
    }
    let factor = (target / p).min(cap);
    let capped = if cap < target / p { format!(" (capped: the cut reaches the {} px edge)", LONGEST as i64) } else { String::new() };
    Decision {
        own_pitch: Some(p),
        lines: pick.1.len(),
        ..Decision::plain(factor, format!("target {}: pitch {}{} → {}{capped}", g(target), p.round() as i64, pick.2, (p * factor).round() as i64))
    }
}

/// a region with no usable pitch of its own borrows the page's: the median own pitch of the OTHER note-sized regions
/// (≤ NOTE_LINES lines) when there are any, else of all the others; None = none measured
pub fn page_borrow(others: &[(f64, usize)]) -> Option<Borrowed> {
    let notes: Vec<f64> = others.iter().filter(|o| o.1 <= NOTE_LINES).map(|o| o.0).collect();
    let pool: Vec<f64> = if notes.is_empty() { others.iter().map(|o| o.0).collect() } else { notes.clone() };
    if pool.is_empty() {
        return None;
    }
    let what = if notes.is_empty() { "the page's regions" } else { "the page's notes" };
    Some(Borrowed { pitch: median(&pool), from: format!("{what} (median of {})", pool.len()) })
}

/// Enlarge small text (auto): a cut with a small pitch enlarged toward TARGET_PITCH, never below 1, never past 1540
pub fn enlarge_small(pitch: Option<f64>, cut_longest: f64, page_small: bool) -> Decision {
    let cap = LONGEST / cut_longest.max(1.0);
    let Some(p) = pitch else {
        if page_small {
            return Decision::plain(NO_PITCH_SMALL_PAGE.min(cap).max(1.0), "no pitch, small page".into());
        }
        return Decision::plain(1.0, "no pitch".into());
    };
    if p >= SMALL_PITCH {
        return Decision::plain(1.0, format!("pitch {} ≥ {}", p.round() as i64, SMALL_PITCH as i64));
    }
    let f = (TARGET_PITCH / p).min(cap).max(1.0);
    Decision::plain(f, format!("pitch {} → {}", p.round() as i64, (p * f).round() as i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(target: f64, pitch: Option<f64>, cores: &[f64], longest: f64) -> f64 {
        normalize(target, &Facts { pitch, core: cores.to_vec(), ..Default::default() }, longest, None).factor
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    /// the desktop tools' decision table, case for case
    #[test]
    fn decision_table() {
        close(enlarge_small(Some(14.0), 300.0, false).factor, 50.0 / 14.0);
        close(enlarge_small(Some(19.9), 300.0, false).factor, 50.0 / 19.9);
        close(enlarge_small(Some(20.0), 300.0, false).factor, 1.0);
        close(enlarge_small(Some(10.0), 1000.0, false).factor, 1.54);
        close(enlarge_small(None, 300.0, true).factor, 2.0);
        close(enlarge_small(None, 300.0, false).factor, 1.0);
        close(enlarge_small(None, 1000.0, true).factor, 1.54);
        close(n(TARGET_MID, Some(25.25), &[10., 11., 12.], 300.0), 2.0);
        close(n(TARGET_LOW, Some(86.0), &[40., 42.], 900.0), 0.5);
        close(n(TARGET_HIGH, Some(59.5), &[20., 22., 21.], 900.0), 1.0);
        close(n(TARGET_MID, Some(25.0), &[10., 25.], 300.0), 1.0);
        close(n(TARGET_MID, Some(25.25), &[8., 6., 7., 6., 6., 6., 6., 7., 5., 6.], 300.0), 2.0);
        close(n(42.0, Some(21.0), &[18., 14., 9., 6., 11., 18., 7.], 300.0), 2.0);
        let b = page_borrow(&[(33., 36), (28., 9), (23., 4), (29.5, 3), (20., 5), (27.5, 4), (23., 4)]);
        let f = Facts { pitch: Some(21.0), core: vec![18., 7.], ..Default::default() };
        close(normalize(43.0, &f, 300.0, b.as_ref()).factor, 43.0 / 23.0);
        close(normalize(43.0, &Facts::default(), 300.0, page_borrow(&[]).as_ref()).factor, 1.0);
        close(n(TARGET_MID, Some(25.0), &[6., 6., 6., 16., 16., 17.], 300.0), 1.0);
        close(n(TARGET_MID, Some(31.0), &[15., 8., 9., 26., 18., 9., 11., 9., 16.], 600.0), 50.5 / 31.0);
        close(n(TARGET_MID, None, &[], 300.0), 1.0);
        let small = Facts { extent: 43.0, band_pitch: Some(27.0), band_core: vec![13., 8.], ..Default::default() };
        close(normalize(TARGET_HIGH, &small, 72.0, None).factor, 59.5 / 27.0);
        close(n(TARGET_MID, Some(10.0), &[4., 4.], 1000.0), 1.54);
        let large = Facts {
            pitch: Some(60.0),
            core: vec![18., 13., 36., 19., 14., 22., 30., 23., 29., 26., 10.],
            wide_pitch: Some(350.0),
            wide_core: vec![146., 202., 233.],
            extent: 1014.0,
            ..Default::default()
        };
        close(normalize(TARGET_MID, &large, 1540.0, None).factor, 50.5 / 350.0);
    }

    #[test]
    fn warnings_and_summary() {
        let d = normalize(TARGET_MID, &Facts::default(), 300.0, None);
        assert!(d.warning.is_some() && d.summary().starts_with("native"));
        let d = normalize(TARGET_MID, &Facts { pitch: Some(25.25), core: vec![10., 11.], ..Default::default() }, 300.0, None);
        assert_eq!(d.summary(), "enlarged ×2.0 (target 50.5: pitch 25 → 51)");
        assert_eq!((d.own_pitch, d.lines), (Some(25.25), 2));
    }
}
