// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The tagged markup → HTML. The marks come from app/markup.json (compiled in — the same marks the model was trained
// to write). FORGIVING: a close without its open, a close
// across another open mark, an open of a mark already open, or an open left unclosed at the line end shows as a plain
// character; the rest of the line renders normally.
//   ⟦del⟧ ⟨ins⟩ 〔u〕 ⌈sup⌉ ⌊sub⌋ → the HTML element of the profile tag's name · ▨ (gap) kept, marked
//   consecutive lines holding a TAB = one table (TAB = cell, ↵ = line break inside a cell, ↵↵ = block break)

use std::collections::{BTreeSet, HashMap};

use anyhow::{Context, Result, bail};
use serde::Serialize;

const PROFILE: &str = include_str!("../../../markup.json");
/// the rendered page's style — ONE file for the app and the saved .html
pub const RENDER_CSS: &str = include_str!("../../../ui/render.css");

/// ▨ — an illegible passage: kept in the text, marked in the rendering
pub const GAP: char = '\u{25A8}';

/// the tags this renderer can draw: each one IS its HTML element
const HTML_TAGS: [&str; 5] = ["del", "ins", "u", "sup", "sub"];

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct Pair {
    pub tag: String,
    pub open: char,
    pub close: char,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Marks {
    /// sorted by tag
    pub pairs: Vec<Pair>,
    pub cell_sep: char,
    pub cell_break: char,
    pub gap: char,
}

fn one_char(s: &str, what: &str) -> Result<char> {
    let mut it = s.chars();
    match (it.next(), it.next()) {
        (Some(c), None) => Ok(c),
        _ => bail!("markup.json {what} '{s}' is not one character"),
    }
}

impl Marks {
    /// the compiled-in profile; no fallback: a malformed profile is an error
    pub fn profile() -> Result<Self> {
        let j: serde_json::Value = serde_json::from_str(PROFILE)?;
        let m = j.get("markup").context("markup.json: no markup")?;
        let inline = m.get("inline").and_then(|v| v.as_object()).context("markup.json: no markup.inline")?;
        let mut pairs = Vec::new();
        for (tag, v) in inline {
            let s = v.as_str().with_context(|| format!("markup.inline.{tag} is no string"))?;
            let c: Vec<char> = s.chars().collect();
            if c.len() != 2 {
                bail!("markup.inline.{tag} '{s}' is not an open + close pair");
            }
            if !HTML_TAGS.contains(&tag.as_str()) {
                bail!("markup.inline.{tag}: the renderer draws only {HTML_TAGS:?}");
            }
            pairs.push(Pair { tag: tag.clone(), open: c[0], close: c[1] });
        }
        pairs.sort_by(|a, b| a.tag.cmp(&b.tag));
        let str_of = |k: &str| m.get(k).and_then(|v| v.as_str()).with_context(|| format!("markup.json: no markup.{k}"));
        let cell_sep = one_char(str_of("cellSep")?, "cellSep")?;
        let cell_break = one_char(str_of("cellBreak")?, "cellBreak")?;
        Ok(Self { pairs, cell_sep, cell_break, gap: GAP })
    }

    /// one line → runs: each paired mark hidden, its tag on the text inside
    pub fn runs(&self, line: &str) -> Vec<(String, BTreeSet<String>)> {
        let c: Vec<char> = line.chars().collect();
        let opens: HashMap<char, &str> = self.pairs.iter().map(|p| (p.open, p.tag.as_str())).collect();
        let closes: HashMap<char, &str> = self.pairs.iter().map(|p| (p.close, p.tag.as_str())).collect();
        let mut role: HashMap<usize, (&str, bool)> = HashMap::new();
        let mut stack: Vec<(&str, usize)> = Vec::new();
        for (k, ch) in c.iter().enumerate() {
            if let Some(&t) = opens.get(ch) {
                if !stack.iter().any(|(s, _)| *s == t) {
                    stack.push((t, k));
                    continue;
                }
            }
            if let Some(&t) = closes.get(ch) {
                if let Some(&(top, at)) = stack.last() {
                    if top == t {
                        role.insert(at, (t, true));
                        role.insert(k, (t, false));
                        stack.pop();
                    }
                }
            }
        }
        let mut out = Vec::new();
        let mut active: BTreeSet<String> = BTreeSet::new();
        let mut cur = String::new();
        for (k, ch) in c.iter().enumerate() {
            if let Some(&(t, opens)) = role.get(&k) {
                if !cur.is_empty() {
                    out.push((std::mem::take(&mut cur), active.clone()));
                }
                if opens {
                    active.insert(t.to_string());
                } else {
                    active.remove(t);
                }
                continue;
            }
            cur.push(*ch);
        }
        if !cur.is_empty() {
            out.push((cur, active));
        }
        out
    }

    fn line_html(&self, line: &str, out: &mut String) {
        for (text, tags) in self.runs(line) {
            for t in &tags {
                out.push_str(&format!("<{t}>"));
            }
            for ch in text.chars() {
                match ch {
                    '&' => out.push_str("&amp;"),
                    '<' => out.push_str("&lt;"),
                    '>' => out.push_str("&gt;"),
                    '"' => out.push_str("&quot;"),
                    c if c == self.gap => out.push_str("<span class=\"gap\">▨</span>"),
                    c => out.push(c),
                }
            }
            for t in tags.iter().rev() {
                out.push_str(&format!("</{t}>"));
            }
        }
    }

    /// the saved page: a standalone .html with the render style inlined
    pub fn document(&self, title: &str, text: &str) -> String {
        format!(
            "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>{}</title><style>\n{RENDER_CSS}</style></head>\n<body><div class=\"page\">\n{}</div></body></html>\n",
            title.replace('<', "&lt;"),
            self.html(text)
        )
    }

    /// the read as an HTML fragment (`<div class="l">` per line, `<table class="t">` per table block)
    pub fn html(&self, text: &str) -> String {
        let lines: Vec<&str> = text.split('\n').collect();
        let mut out = String::new();
        let mut i = 0;
        while i < lines.len() {
            if !lines[i].contains(self.cell_sep) {
                if lines[i].trim().is_empty() {
                    // a <br> holds the empty line open (an editable caret position; one line high in a saved page)
                    out.push_str("<div class=\"l empty\"><br></div>\n");
                } else {
                    out.push_str("<div class=\"l\">");
                    self.line_html(lines[i], &mut out);
                    out.push_str("</div>\n");
                }
                i += 1;
                continue;
            }
            let mut rows: Vec<Vec<&str>> = Vec::new();
            while i < lines.len() && lines[i].contains(self.cell_sep) {
                rows.push(lines[i].split(self.cell_sep).collect());
                i += 1;
            }
            let ncol = rows.iter().map(|r| r.len()).max().unwrap_or(0);
            out.push_str("<table class=\"t\">\n");
            for row in &rows {
                out.push_str("<tr>");
                for col in 0..ncol {
                    out.push_str("<td>");
                    if let Some(cell) = row.get(col) {
                        for (n, piece) in cell.split(self.cell_break).enumerate() {
                            if n > 0 {
                                out.push_str("<br>");
                            }
                            self.line_html(piece.trim(), &mut out);
                        }
                    }
                    out.push_str("</td>");
                }
                out.push_str("</tr>\n");
            }
            out.push_str("</table>\n");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_loads() {
        let m = Marks::profile().unwrap();
        assert_eq!(m.pairs.len(), 5);
        assert_eq!(m.cell_sep, '\t');
        assert_eq!(m.cell_break, '↵');
    }

    #[test]
    fn pairs_nest_and_unpaired_stay_text() {
        let m = Marks::profile().unwrap();
        assert_eq!(m.html("a⟦b⌈c⌉⟧d"), "<div class=\"l\">a<del>b</del><del><sup>c</sup></del>d</div>\n");
        // a close without its open, an open left unclosed: plain characters
        assert_eq!(m.html("x⟧ ⟨y"), "<div class=\"l\">x⟧ ⟨y</div>\n");
    }

    #[test]
    fn table_rows_cells_breaks() {
        let m = Marks::profile().unwrap();
        let h = m.html("A\tB↵C\n\tD");
        assert_eq!(h, "<table class=\"t\">\n<tr><td>A</td><td>B<br>C</td></tr>\n<tr><td></td><td>D</td></tr>\n</table>\n");
    }

    #[test]
    fn empty_line_holds_a_br() {
        let m = Marks::profile().unwrap();
        assert_eq!(m.html("a\n\nb"), "<div class=\"l\">a</div>\n<div class=\"l empty\"><br></div>\n<div class=\"l\">b</div>\n");
    }

    #[test]
    fn escapes_and_gap() {
        let m = Marks::profile().unwrap();
        assert_eq!(m.html("<&▨"), "<div class=\"l\">&lt;&amp;<span class=\"gap\">▨</span></div>\n");
    }
}
