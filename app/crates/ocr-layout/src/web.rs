// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
//! The layout tools in the browser (the editor's layout worker): JSON in, JSON out. `ink_page` analyses the page once
//! (the scaled crop the model reads) and keeps it; `tool(name, args)` runs one tool. Polygons are [[x, y], …].

use std::cell::RefCell;

use serde_json::{Value, json};
use wasm_bindgen::prelude::*;

use crate::geom::{self, Pt, Rect};
use crate::ink::{self, Page};
use crate::scale::{self, Borrowed, Facts};
use crate::shape::Side;
use crate::{order, tools};

thread_local! {
    /// the analysed page and its text blobs
    static PAGE: RefCell<Option<(Page, Vec<bool>)>> = const { RefCell::new(None) };
}

fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

fn poly(v: &Value) -> Result<Vec<Pt>, JsError> {
    let a = v.as_array().ok_or_else(|| err("a polygon must be [[x, y], …]"))?;
    a.iter()
        .map(|p| match (p.get(0).and_then(Value::as_f64), p.get(1).and_then(Value::as_f64)) {
            (Some(x), Some(y)) => Ok(Pt::new(x, y)),
            _ => Err(err("a point must be [x, y]")),
        })
        .collect()
}

fn polys(v: &Value) -> Result<Vec<Vec<Pt>>, JsError> {
    v.as_array().ok_or_else(|| err("expected a list of polygons"))?.iter().map(poly).collect()
}

fn out(p: &[Pt]) -> Value {
    Value::Array(p.iter().map(|q| json!([q.x, q.y])).collect())
}

fn num(a: &Value, k: &str) -> Result<f64, JsError> {
    a.get(k).and_then(Value::as_f64).ok_or_else(|| err(format!("{k}: a number")))
}

fn idx(a: &Value, k: &str) -> Result<usize, JsError> {
    a.get(k).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| err(format!("{k}: an index")))
}

fn facts(v: &Value) -> Facts {
    let list = |k: &str| v.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
    Facts {
        pitch: v.get("pitch").and_then(Value::as_f64),
        core: list("core"),
        wide_pitch: v.get("widePitch").and_then(Value::as_f64),
        wide_core: list("wideCore"),
        extent: v.get("extent").and_then(Value::as_f64).unwrap_or(0.0),
        band_pitch: v.get("bandPitch").and_then(Value::as_f64),
        band_core: list("bandCore"),
    }
}

/// the page analysed (RGB of the scaled crop) and kept → { pitch } (bake px)
#[wasm_bindgen]
pub fn ink_page(rgb: &[u8], w: usize, h: usize) -> Result<String, JsError> {
    let pg = ink::page_from_gray(ink::img::luma_pil_rgb(rgb, w, h)).map_err(err)?;
    let text = ink::page::text_blobs(&pg);
    let pitch = pg.pitch / pg.f;
    PAGE.with(|p| *p.borrow_mut() = Some((pg, text)));
    Ok(json!({ "pitch": pitch }).to_string())
}

/// one tool: `name` with its JSON `args` → its JSON result
#[wasm_bindgen]
pub fn tool(name: &str, args: &str) -> Result<String, JsError> {
    let a: Value = serde_json::from_str(args).map_err(err)?;
    let r = match name {
        "snap" => PAGE.with(|p| -> Result<Value, JsError> {
            let b = p.borrow();
            let (pg, text) = b.as_ref().ok_or_else(|| err("no page analysed (ink_page first)"))?;
            let (polygon, blobs) = tools::snap(pg, text, &poly(&a["drawn"])?, &polys(&a["others"])?, num(&a, "fx")?, num(&a, "fy")?, num(&a, "tol")?).map_err(err)?;
            Ok(json!({ "polygon": out(&polygon), "blobs": blobs }))
        })?,
        "measure" => PAGE.with(|p| -> Result<Value, JsError> {
            let b = p.borrow();
            let (pg, _) = b.as_ref().ok_or_else(|| err("no page analysed (ink_page first)"))?;
            let (fx, fy) = (num(&a, "fx")?, num(&a, "fy")?);
            let list: Vec<Value> = polys(&a["polygons"])?
                .iter()
                .map(|q| {
                    let scaled: Vec<Pt> = q.iter().map(|p| Pt::new(p.x * fx, p.y * fy)).collect();
                    let m = ink::measure(pg, &[scaled]);
                    json!({ "pitch": m.pitch, "core": m.core, "widePitch": m.wide_pitch, "wideCore": m.wide_core, "extent": m.extent, "bandPitch": m.band_pitch, "bandCore": m.band_core })
                })
                .collect();
            Ok(Value::Array(list))
        })?,
        "normalize" => {
            let borrowed = match a.get("borrowed").filter(|b| !b.is_null()) {
                Some(b) => Some(Borrowed { pitch: num(b, "pitch")?, from: b["from"].as_str().ok_or_else(|| err("borrowed.from: a string"))?.to_owned() }),
                None => None,
            };
            let d = scale::normalize(num(&a, "target")?, &facts(&a["facts"]), num(&a, "cutLongest")?, borrowed.as_ref());
            json!({ "factor": d.factor, "reason": d.reason, "summary": d.summary(), "warning": d.warning, "ownPitch": d.own_pitch, "lines": d.lines })
        }
        "borrow" => {
            let others: Vec<(f64, usize)> = a["others"].as_array().ok_or_else(|| err("others: [[pitch, lines], …]"))?.iter().filter_map(|o| Some((o.get(0)?.as_f64()?, o.get(1)?.as_u64()? as usize))).collect();
            match scale::page_borrow(&others) {
                Some(b) => json!({ "pitch": b.pitch, "from": b.from }),
                None => Value::Null,
            }
        }
        "straighten" => {
            let (p, n) = tools::straighten(&poly(&a["polygon"])?, idx(&a, "a")?, idx(&a, "b")?, &polys(&a["others"])?, num(&a, "tol")?).map_err(err)?;
            json!({ "polygon": out(&p), "gaveWay": n })
        }
        "sides" => {
            let sides: Vec<Side> = a["sides"]
                .as_array()
                .ok_or_else(|| err("sides: [\"left\", …]"))?
                .iter()
                .map(|s| match s.as_str() {
                    Some("left") => Ok(Side::Left),
                    Some("top") => Ok(Side::Top),
                    Some("right") => Ok(Side::Right),
                    Some("bottom") => Ok(Side::Bottom),
                    _ => Err(err("a side is left, top, right or bottom")),
                })
                .collect::<Result<_, _>>()?;
            let (p, n) = tools::sides(&poly(&a["polygon"])?, &sides, &polys(&a["others"])?, num(&a, "tol")?).map_err(err)?;
            json!({ "polygon": out(&p), "gaveWay": n })
        }
        "merge" => {
            let (p, how) = tools::merge(&polys(&a["parts"])?, &polys(&a["others"])?, num(&a, "tol")?).map_err(err)?;
            json!({ "polygon": out(&p), "how": how })
        }
        "simplify" => {
            let page = Rect { x: 0.0, y: 0.0, w: num(&a, "width")?, h: num(&a, "height")? };
            json!({ "polygon": out(&tools::simplify(&poly(&a["polygon"])?, &polys(&a["others"])?, page, num(&a, "tol")?)) })
        }
        "cuts" => {
            let selected = a.get("selected").and_then(Value::as_u64).map(|v| v as usize);
            match tools::cuts(&polys(&a["regions"])?, &poly(&a["line"])?, selected, num(&a, "snap")?) {
                Ok(c) => json!({ "cuts": c.iter().map(|(i, l, (p, q))| json!({ "index": i, "line": out(l), "parts": [out(p), out(q)] })).collect::<Vec<_>>() }),
                Err(e) => json!({ "cuts": [], "why": e.0 }),
            }
        }
        "insertNode" => match geom::insert_node(&poly(&a["polygon"])?, Pt::new(num(&a, "x")?, num(&a, "y")?), num(&a, "tol")?, a["closed"].as_bool().unwrap_or(true)) {
            Some((p, k)) => json!({ "polygon": out(&p), "index": k }),
            None => Value::Null,
        },
        "deleteNode" => json!({ "polygon": out(&geom::delete_node(&poly(&a["polygon"])?, idx(&a, "index")?, a["closed"].as_bool().unwrap_or(true)).map_err(err)?) }),
        "order" => {
            let boxes: Vec<[f64; 4]> = polys(&a["regions"])?.iter().map(|p| tools::region_box(p)).collect();
            let reads: Vec<Option<String>> = a["reads"].as_array().map(|r| r.iter().map(|v| v.as_str().map(str::to_owned)).collect()).unwrap_or_else(|| vec![None; boxes.len()]);
            let o = order::order(&boxes, num(&a, "height")?, a.get("pitch").and_then(Value::as_f64), &reads, a["preset"].as_str()).map_err(err)?;
            json!({ "order": o.order, "types": o.types, "preset": o.preset, "why": o.why })
        }
        "presets" => Value::Array(order::PRESETS.iter().map(|p| json!({ "id": p.id, "label": p.label })).collect()),
        "tolerance" => json!(tools::tolerance(num(&a, "bakeScale")?)),
        _ => return Err(err(format!("no layout tool {name}"))),
    };
    Ok(r.to_string())
}
