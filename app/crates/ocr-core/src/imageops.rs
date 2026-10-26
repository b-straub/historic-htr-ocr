// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Cheap image operations before a read (a hand-drawn crop, rotate, contrast, pencil) + THE bake — the model input as
// the model was trained on it: the longest edge scaled DOWN to 1540 (never up, Lanczos like CILanczosScaleTransform), the short edge padded to 56,
// then white padded to the 28 px grid — each pad split half left / top (the smaller half), half right / bottom.

use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use image::{DynamicImage, ImageDecoder, ImageReader, Rgb, RgbImage, imageops};
use jxl_oxide::integration::JxlDecoder;
use moxcms::{ColorProfile, DataColorSpace, Layout, TransformOptions};
use serde::{Deserialize, Serialize};

/// Load an image the way Apple's image loader hands it over: EXIF orientation applied, an embedded RGB ICC profile
/// converted to sRGB (in a parity run against Core Graphics, 9 of 154 scans moved 5–22 px only because Core Graphics
/// converts scanner / Adobe RGB profiles and the image crate ignores them). A grey profile (Gray Gamma 2.2) is left as
/// is: those scans already crop identically. A profile that cannot be read is an error.
pub fn load(path: &Path) -> Result<DynamicImage> {
    let mut head = [0u8; 12];
    let n = std::io::Read::read(&mut std::fs::File::open(path)?, &mut head)?;
    if is_jxl(&head[..n]) {
        return decode(JxlDecoder::new(std::io::BufReader::new(std::fs::File::open(path)?))?);
    }
    decode(ImageReader::open(path)?.with_guessed_format()?.into_decoder()?)
}

/// `load` for an image already in memory (the browser build: a dropped / picked file's bytes)
pub fn load_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    if is_jxl(bytes) {
        return decode(JxlDecoder::new(Cursor::new(bytes))?);
    }
    decode(ImageReader::new(Cursor::new(bytes)).with_guessed_format()?.into_decoder()?)
}

/// JPEG XL: a bare codestream (FF 0A) or the ISO BMFF container ("JXL " box) — the image crate does not know the format
fn is_jxl(head: &[u8]) -> bool {
    head.starts_with(&[0xFF, 0x0A]) || head.starts_with(&[0, 0, 0, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A])
}

fn decode(mut dec: impl ImageDecoder) -> Result<DynamicImage> {
    let orientation = dec.orientation()?;
    let icc = dec.icc_profile()?;
    let mut img = DynamicImage::from_decoder(dec)?;
    if let Some(icc) = icc {
        img = to_srgb(img, &icc)?;
    }
    img.apply_orientation(orientation);
    Ok(img)
}

fn to_srgb(img: DynamicImage, icc: &[u8]) -> Result<DynamicImage> {
    let src = ColorProfile::new_from_slice(icc).map_err(|e| anyhow!("embedded ICC profile unreadable: {e:?}"))?;
    if src.color_space != DataColorSpace::Rgb {
        return Ok(img);
    }
    let opts = TransformOptions { prefer_fixed_point: false, ..TransformOptions::default() };
    let t = src
        .create_transform_8bit(Layout::Rgb, &ColorProfile::new_srgb(), Layout::Rgb, opts)
        .map_err(|e| anyhow!("ICC → sRGB transform: {e:?}"))?;
    // an sRGB variant (a camera's / scanner's "sRGB") maps every colour to itself within 1 level: leave the pixels alone
    // (ColorSync does — converting them moved crops of matching scans in the parity run)
    let grid: Vec<u8> = (0..17u32)
        .flat_map(|r| (0..17u32).flat_map(move |g| (0..17u32).flat_map(move |b| [r * 255 / 16, g * 255 / 16, b * 255 / 16])))
        .map(|v| v as u8)
        .collect();
    let mut probe = vec![0u8; grid.len()];
    t.transform(&grid, &mut probe).map_err(|e| anyhow!("ICC → sRGB: {e:?}"))?;
    if grid.iter().zip(&probe).all(|(a, b)| a.abs_diff(*b) <= 1) {
        return Ok(img);
    }
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut out = vec![0u8; rgb.as_raw().len()];
    t.transform(rgb.as_raw(), &mut out).map_err(|e| anyhow!("ICC → sRGB: {e:?}"))?;
    Ok(DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, out).context("ICC → sRGB: buffer size")?))
}

/// One step of the prep chain; a crop rect is in the pixels of the image the steps BEFORE it produced.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "op", rename_all = "camelCase")]
pub enum Op {
    RotateLeft,
    RotateRight,
    Rotate180,
    /// levels stretch: luminance 0.5 % / 99.5 % percentiles → 0 / 255, every channel
    AutoContrast,
    /// the "pencil" preset (flatten 1, contrast 1, channel G): the green channel divided
    /// by its paper estimate (a Gaussian of sigma 25 px, edges clamped) → paper 255, ink kept; written as grey
    Pencil,
    Crop { x: u32, y: u32, w: u32, h: u32 },
}

pub fn apply(src: &DynamicImage, ops: &[Op]) -> Result<RgbImage> {
    let mut img = src.to_rgb8();
    for op in ops {
        img = match op {
            Op::RotateLeft => imageops::rotate270(&img),
            Op::RotateRight => imageops::rotate90(&img),
            Op::Rotate180 => imageops::rotate180(&img),
            Op::AutoContrast => auto_contrast(img),
            Op::Pencil => pencil(&img),
            Op::Crop { x, y, w, h } => {
                if *w == 0 || *h == 0 || x + w > img.width() || y + h > img.height() {
                    bail!("crop {w}×{h} at ({x},{y}) is outside the {}×{} image", img.width(), img.height());
                }
                imageops::crop_imm(&img, *x, *y, *w, *h).to_image()
            }
        };
    }
    Ok(img)
}

fn auto_contrast(mut img: RgbImage) -> RgbImage {
    let mut hist = [0u64; 256];
    for p in img.pixels() {
        hist[luma(p) as usize] += 1;
    }
    let n = img.width() as u64 * img.height() as u64;
    let pct = |q: f64| -> f64 {
        let target = (q * n as f64) as u64;
        let mut acc = 0u64;
        for (v, c) in hist.iter().enumerate() {
            acc += c;
            if acc > target {
                return v as f64;
            }
        }
        255.0
    };
    let (lo, hi) = (pct(0.005), pct(0.995));
    if hi <= lo {
        return img;
    }
    let lut: Vec<u8> = (0..256).map(|v| (((v as f64 - lo) * 255.0 / (hi - lo)).clamp(0.0, 255.0)).round() as u8).collect();
    for p in img.pixels_mut() {
        for c in p.0.iter_mut() {
            *c = lut[*c as usize];
        }
    }
    img
}

/// RegionAdjust.pencil's sigma: 25 px at the ORIGINAL resolution — OCR Lite's ops run on the full-resolution prepared
/// image, so it applies unscaled
const PENCIL_SIGMA: f64 = 25.0;

/// RegionAdjust.apply with the pencil preset: per pixel out = min(1, G / max(blur(G), 0.004)), R = G = B. Only G is
/// computed (flatten is per channel, then channel G is taken). The blur approximates CIGaussianBlur on clampedToExtent
/// by three box passes per axis (edges clamped) — O(pixels), independent of sigma.
fn pencil(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let g: Vec<f32> = img.pixels().map(|p| p.0[1] as f32).collect();
    let mut blur = g.clone();
    let mut tmp = vec![0f32; w * h];
    for size in boxes_for_gauss(PENCIL_SIGMA, 3) {
        let r = size / 2;
        box_rows(&blur, &mut tmp, w, h, r);
        box_cols(&tmp, &mut blur, w, h, r);
    }
    let mut out = RgbImage::new(img.width(), img.height());
    for ((p, v), b) in out.pixels_mut().zip(&g).zip(&blur) {
        let o = (v * 255.0 / b.max(255.0 * 0.004)).min(255.0).round() as u8;
        p.0 = [o, o, o];
    }
    out
}

/// the odd box widths whose n successive passes approximate a Gaussian of `sigma` (Kuckir / Wells)
fn boxes_for_gauss(sigma: f64, n: usize) -> Vec<usize> {
    let nf = n as f64;
    let ideal = (12.0 * sigma * sigma / nf + 1.0).sqrt();
    let mut wl = ideal.floor() as i64;
    if wl % 2 == 0 {
        wl -= 1;
    }
    let wu = wl + 2;
    let wlf = wl as f64;
    let m = ((12.0 * sigma * sigma - nf * wlf * wlf - 4.0 * nf * wlf - 3.0 * nf) / (-4.0 * wlf - 4.0)).round() as usize;
    (0..n).map(|i| if i < m { wl as usize } else { wu as usize }).collect()
}

/// a horizontal box mean of radius r per row, edge pixels repeated beyond the border
fn box_rows(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    let norm = 1.0 / (2 * r + 1) as f64;
    let last = w - 1;
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out = &mut dst[y * w..(y + 1) * w];
        let mut acc: f64 = (0..=2 * r).map(|k| row[k.saturating_sub(r).min(last)] as f64).sum();
        for x in 0..w {
            out[x] = (acc * norm) as f32;
            acc += row[(x + r + 1).min(last)] as f64 - row[x.saturating_sub(r)] as f64;
        }
    }
}

/// a vertical box mean of radius r per column (row-major sweep with one running sum per column), edges repeated
fn box_cols(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    let norm = 1.0 / (2 * r + 1) as f64;
    let last = h - 1;
    let mut acc = vec![0f64; w];
    for k in 0..=2 * r {
        let row = &src[k.saturating_sub(r).min(last) * w..][..w];
        for (a, v) in acc.iter_mut().zip(row) {
            *a += *v as f64;
        }
    }
    for y in 0..h {
        let add = &src[(y + r + 1).min(last) * w..][..w];
        let sub = &src[y.saturating_sub(r) * w..][..w];
        let out = &mut dst[y * w..(y + 1) * w];
        for x in 0..w {
            out[x] = (acc[x] * norm) as f32;
            acc[x] += add[x] as f64 - sub[x] as f64;
        }
    }
}

fn luma(p: &Rgb<u8>) -> u8 {
    ((p.0[0] as u32 * 299 + p.0[1] as u32 * 587 + p.0[2] as u32 * 114) / 1000) as u8
}

/// the on-screen preview's longest edge: `sharp` = the zoomed-in view (UI zoom beyond the normal preview's own pixels —
/// one larger preview, still bounded for the browser's memory), else the normal one
pub fn preview_longest(sharp: bool) -> u32 {
    if sharp { 4096 } else { 1800 }
}

/// the on-screen preview: scaled down to `longest` px (Triangle — fast, display only), JPEG q85
pub fn preview_jpeg(img: &RgbImage, longest: u32) -> Result<Vec<u8>> {
    let (w, h) = img.dimensions();
    let s = (longest as f64 / w.max(h) as f64).min(1.0);
    let mut buf = Cursor::new(Vec::new());
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 85);
    if s < 1.0 {
        let (sw, sh) = (((w as f64 * s).round() as u32).max(1), ((h as f64 * s).round() as u32).max(1));
        enc.encode_image(&imageops::resize(img, sw, sh, imageops::FilterType::Triangle))?;
    } else {
        enc.encode_image(img)?;
    }
    Ok(buf.into_inner())
}

/// the model input saved as PNG
pub fn png(img: &RgbImage) -> Result<Vec<u8>> {
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)?;
    Ok(buf.into_inner())
}

/// the model family's input: longest edge + patch grid (Pixtral, LightOnOCR-2 / -3-1B: 1540 / 28)
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Bake {
    pub longest: u32,
    pub grid: u32,
}

impl Bake {
    pub const PIXTRAL: Bake = Bake { longest: 1540, grid: 28 };
}

#[derive(Serialize, Clone, Copy, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct BakePlan {
    pub scaled_w: u32,
    pub scaled_h: u32,
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
    /// the model's patch grid in final-image pixels (Bake.grid) — the UI's scan overlay draws it
    pub grid: u32,
}

impl BakePlan {
    pub fn final_w(&self) -> u32 {
        self.scaled_w + self.left + self.right
    }
    pub fn final_h(&self) -> u32 {
        self.scaled_h + self.top + self.bottom
    }
}

/// the bake's plan for a w0 × h0 image: the scaled size and the white pad on each side
pub fn plan(b: Bake, w0: u32, h0: u32) -> BakePlan {
    let (mut w, mut h) = (w0 as f64, h0 as f64);
    let l = w.max(h);
    if l > b.longest as f64 {
        let scale = b.longest as f64 / l;
        w = (w * scale).round().max(1.0);
        h = (h * scale).round().max(1.0);
    }
    let (scaled_w, scaled_h) = (w as u32, h as u32);
    let (mut left, mut top, mut right, mut bottom) = (0u32, 0u32, 0u32, 0u32);
    let mut pad = |w: &mut f64, h: &mut f64, nw: f64, nh: f64| {
        let pw = (nw - *w) as u32;
        let ph = (nh - *h) as u32;
        left += pw / 2;
        right += pw - pw / 2;
        top += ph / 2;
        bottom += ph - ph / 2;
        *w = nw;
        *h = nh;
    };
    let min_short = 56.0;
    if w.min(h) < min_short {
        let (nw, nh) = if w <= h { (min_short, h) } else { (w, min_short) };
        pad(&mut w, &mut h, nw, nh);
    }
    let g = b.grid as f64;
    let (sw, sh) = ((w / g).ceil() * g, (h / g).ceil() * g);
    if sw != w || sh != h {
        pad(&mut w, &mut h, sw, sh);
    }
    BakePlan { scaled_w, scaled_h, left, top, right, bottom, grid: b.grid }
}

/// the baked model input: one resize + one white pad
pub fn bake(img: &RgbImage, b: Bake) -> (RgbImage, BakePlan) {
    let p = plan(b, img.width(), img.height());
    let scaled = if (p.scaled_w, p.scaled_h) != img.dimensions() {
        imageops::resize(img, p.scaled_w, p.scaled_h, imageops::FilterType::Lanczos3)
    } else {
        img.clone()
    };
    let mut out = RgbImage::from_pixel(p.final_w(), p.final_h(), Rgb([255, 255, 255]));
    imageops::replace(&mut out, &scaled, p.left as i64, p.top as i64);
    (out, p)
}

/// how a region's model input is made beyond its cut (the desktop GT tools' region read): AutoContrast with the levels
/// from INSIDE the polygon, the outside in the paper tone instead of white, the cut scaled (the region's line-pitch
/// decision), quarter turns clockwise
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RegionInput {
    pub contrast: bool,
    pub paper_fill: bool,
    pub scale: f64,
    pub quarters: u8,
}

impl RegionInput {
    /// the cut as it is: white outside, no contrast, no scale, upright
    pub const PLAIN: RegionInput = RegionInput { contrast: false, paper_fill: false, scale: 1.0, quarters: 0 };
}

/// a region's luminance levels: below this share of the paper tone = scanner background, not ink …
const BORDER_SHARE: f64 = 0.3;
/// … above this share of it = a white pad / patch, not paper
const PAD_SHARE: f64 = 1.25;

/// AutoContrast with the levels from the pixels `inside` only (the LUT maps every pixel): the paper = the inside's 75th
/// luminance percentile, pixels darker than BORDER_SHARE × it or brighter than PAD_SHARE × it left out, then the 0.5 /
/// 99.5 % levels; unchanged when hi ≤ lo or nothing is left
fn auto_contrast_inside(img: &mut RgbImage, inside: &[bool]) {
    let mut hist = [0u64; 256];
    let mut n = 0u64;
    for (p, &i) in img.pixels().zip(inside) {
        if i {
            hist[luma(p) as usize] += 1;
            n += 1;
        }
    }
    let pct = |hist: &[u64; 256], n: u64, q: f64| -> f64 {
        let target = (q * n as f64) as u64;
        let mut acc = 0u64;
        for (v, c) in hist.iter().enumerate() {
            acc += c;
            if acc > target {
                return v as f64;
            }
        }
        255.0
    };
    if n == 0 {
        return;
    }
    // the brighter half's median (the 75th percentile, n / 2 + n / 4 as the desktop counts it)
    let target = n / 2 + n / 4;
    let mut acc = 0u64;
    let mut paper = 255usize;
    for (v, c) in hist.iter().enumerate() {
        acc += c;
        if acc > target {
            paper = v;
            break;
        }
    }
    let (floor, ceil) = ((BORDER_SHARE * paper as f64) as usize, (PAD_SHARE * paper as f64) as usize);
    for (v, c) in hist.iter_mut().enumerate() {
        if v < floor || v > ceil {
            n -= *c;
            *c = 0;
        }
    }
    if n == 0 {
        return;
    }
    let (lo, hi) = (pct(&hist, n, 0.005), pct(&hist, n, 0.995));
    if hi <= lo {
        return;
    }
    let lut: Vec<u8> = (0..256).map(|v| (((v as f64 - lo) * 255.0 / (hi - lo)).clamp(0.0, 255.0)).round() as u8).collect();
    for p in img.pixels_mut() {
        for c in p.0.iter_mut() {
            *c = lut[*c as usize];
        }
    }
}

/// the paper tone of the pixels `inside`: per channel the median of those in the inside's brighter half by luminance
/// (sampled to ≤ ~200 k pixels); None = no pixel inside
fn paper_tone(img: &RgbImage, inside: &[bool]) -> Option<Rgb<u8>> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let step = ((w * h) as f64 / 200_000.0).sqrt().max(1.0) as usize;
    let mut lum: Vec<(u32, [u8; 3])> = Vec::new();
    for y in (0..h).step_by(step) {
        for x in (0..w).step_by(step) {
            if inside[y * w + x] {
                let p = img.get_pixel(x as u32, y as u32).0;
                lum.push((299 * p[0] as u32 + 587 * p[1] as u32 + 114 * p[2] as u32, p));
            }
        }
    }
    if lum.is_empty() {
        return None;
    }
    lum.sort_by_key(|l| l.0);
    let bright = &lum[lum.len() / 2..];
    let med = |c: usize| {
        let mut v: Vec<u8> = bright.iter().map(|l| l.1[c]).collect();
        v.sort();
        v[v.len() / 2]
    };
    Some(Rgb([med(0), med(1), med(2)]))
}

/// a region's model input: the page (after its steps) at the PAGE's bake scale — the writing keeps the size it has in a
/// whole-page read — the region's bounding box cut; then as the desktop GT tools read a region: AutoContrast with the
/// levels from inside the polygon (`input.contrast`), the outside painted white or in the inside's paper tone (even-odd,
/// row by row), the cut scaled by `input.scale` (the region's line-pitch decision; Lanczos), turned, baked (pad only
/// unless the scale passed the bake's edge). `polygon` in the page's pixels.
pub fn bake_region(page: &RgbImage, polygon: &[(f64, f64)], b: Bake, input: &RegionInput) -> Result<(RgbImage, BakePlan)> {
    if polygon.len() < 3 {
        bail!("a region needs at least 3 corners, got {}", polygon.len());
    }
    if !(input.scale > 0.0) {
        bail!("a region's scale must be positive, got {}", input.scale);
    }
    let (pw, ph) = page.dimensions();
    let page_plan = plan(b, pw, ph);
    // the page's scale per axis exactly as its bake rounds it: a whole-page region is pixel for pixel the page's bake
    let (fx, fy) = (page_plan.scaled_w as f64 / pw as f64, page_plan.scaled_h as f64 / ph as f64);
    let xs = polygon.iter().map(|p| p.0);
    let ys = polygon.iter().map(|p| p.1);
    let x0 = xs.clone().fold(f64::INFINITY, f64::min).floor().max(0.0) as u32;
    let y0 = ys.clone().fold(f64::INFINITY, f64::min).floor().max(0.0) as u32;
    let x1 = (xs.fold(f64::NEG_INFINITY, f64::max).ceil() as u32).min(pw);
    let y1 = (ys.fold(f64::NEG_INFINITY, f64::max).ceil() as u32).min(ph);
    if x1 <= x0 || y1 <= y0 {
        bail!("the region lies outside the page ({pw} × {ph})");
    }
    let cut = imageops::crop_imm(page, x0, y0, x1 - x0, y1 - y0).to_image();
    let (cw, ch) = (((x1 - x0) as f64 * fx).round().max(1.0) as u32, ((y1 - y0) as f64 * fy).round().max(1.0) as u32);
    let mut region = if (cw, ch) != cut.dimensions() { imageops::resize(&cut, cw, ch, imageops::FilterType::Lanczos3) } else { cut };
    // inside the polygon: per row of the scaled cut, the polygon's crossings at the row's centre (in page pixels)
    let (sx, sy) = ((x1 - x0) as f64 / cw as f64, (y1 - y0) as f64 / ch as f64);
    let mut inside = vec![false; (cw * ch) as usize];
    for row in 0..ch {
        let y = y0 as f64 + (row as f64 + 0.5) * sy;
        let mut xs: Vec<f64> = Vec::new();
        for i in 0..polygon.len() {
            let (a, c) = (polygon[i], polygon[(i + 1) % polygon.len()]);
            if (a.1 <= y) != (c.1 <= y) {
                xs.push(a.0 + (y - a.1) / (c.1 - a.1) * (c.0 - a.0));
            }
        }
        for col in 0..cw {
            let x = x0 as f64 + (col as f64 + 0.5) * sx;
            inside[(row * cw + col) as usize] = xs.iter().filter(|&&c| c < x).count() % 2 == 1;
        }
    }
    if input.contrast {
        auto_contrast_inside(&mut region, &inside);
    }
    let fill = if input.paper_fill { paper_tone(&region, &inside).unwrap_or(Rgb([255, 255, 255])) } else { Rgb([255, 255, 255]) };
    for (p, &i) in region.pixels_mut().zip(&inside) {
        if !i {
            *p = fill;
        }
    }
    if (input.scale - 1.0).abs() > 1e-4 {
        let (w, h) = region.dimensions();
        let (nw, nh) = (((w as f64 * input.scale).round() as u32).max(1), ((h as f64 * input.scale).round() as u32).max(1));
        region = imageops::resize(&region, nw, nh, imageops::FilterType::Lanczos3);
    }
    region = match input.quarters % 4 {
        1 => imageops::rotate90(&region),
        2 => imageops::rotate180(&region),
        3 => imageops::rotate270(&region),
        _ => region,
    };
    Ok(bake(&region, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_scales_down_and_pads_to_grid() {
        let p = plan(Bake::PIXTRAL, 3000, 2000);
        assert_eq!((p.scaled_w, p.scaled_h), (1540, 1027));
        assert_eq!((p.final_w(), p.final_h()), (1540, 1036));
        assert_eq!((p.top, p.bottom), (4, 5));
        assert_eq!(p.grid, 28);
    }

    #[test]
    fn plan_never_upscales() {
        let p = plan(Bake::PIXTRAL, 500, 30);
        assert_eq!((p.scaled_w, p.scaled_h), (500, 30));
        assert_eq!((p.final_w(), p.final_h()), (504, 56));
    }

    #[test]
    fn boxes_match_the_gaussian_variance() {
        // n boxes of width b add variance (b² − 1) / 12 each: the sum must be ≈ sigma²
        let b = boxes_for_gauss(25.0, 3);
        let var: f64 = b.iter().map(|&x| ((x * x) as f64 - 1.0) / 12.0).sum();
        assert!(b.iter().all(|x| x % 2 == 1));
        assert!((var.sqrt() - 25.0).abs() < 1.0, "{b:?} → sigma {}", var.sqrt());
    }

    #[test]
    fn pencil_flattens_paper_and_keeps_ink() {
        // a paper gradient (G 140 → 230 left → right, R / B off-tone) with a dark horizontal stroke across it
        let (w, h) = (400u32, 300u32);
        let img = RgbImage::from_fn(w, h, |x, y| {
            let paper = 140 + (90 * x / (w - 1)) as u8;
            if (148..154).contains(&y) { Rgb([60, 40, 30]) } else { Rgb([paper.saturating_add(20), paper, paper - 30]) }
        });
        let out = apply(&DynamicImage::ImageRgb8(img), &[Op::Pencil]).unwrap();
        assert_eq!(out.dimensions(), (w, h));
        for (x, y, p) in out.enumerate_pixels() {
            assert!(p.0[0] == p.0[1] && p.0[1] == p.0[2], "grey output at ({x},{y})");
            if (148..154).contains(&y) {
                assert!(p.0[1] < 120, "stroke stays dark at ({x},{y}): {}", p.0[1]);
            } else {
                assert!(p.0[1] >= 250, "paper ≈ 255 at ({x},{y}): {}", p.0[1]);
            }
        }
    }

    #[test]
    fn ops_serde_names() {
        let ops: Vec<Op> = serde_json::from_str(r#"[{"op":"autoContrast"},{"op":"pencil"},{"op":"rotate180"}]"#).unwrap();
        assert_eq!(ops, vec![Op::AutoContrast, Op::Pencil, Op::Rotate180]);
        assert!(serde_json::from_str::<Vec<Op>>(r#"[{"op":"grey"}]"#).is_err());
        assert!(serde_json::from_str::<Vec<Op>>(r#"[{"op":"invert"}]"#).is_err());
    }

    #[test]
    #[ignore = "timing on a 6000 × 4000 scan: cargo test --release -p ocr-core pencil_timing -- --ignored --nocapture"]
    fn pencil_timing() {
        let img = RgbImage::from_fn(6000, 4000, |x, y| Rgb([((x ^ y) & 255) as u8, ((x * 7 + y) & 255) as u8, 200]));
        let t = std::time::Instant::now();
        let out = pencil(&img);
        println!("pencil 6000×4000: {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(out.dimensions(), (6000, 4000));
    }

    #[test]
    fn crop_outside_is_an_error() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(10, 10));
        assert!(apply(&img, &[Op::Crop { x: 5, y: 5, w: 6, h: 2 }]).is_err());
        let r = apply(&img, &[Op::RotateRight, Op::Crop { x: 1, y: 1, w: 4, h: 3 }]).unwrap();
        assert_eq!(r.dimensions(), (4, 3));
    }

    /// a rectangle region = its box, at the page's scale; a triangle whitens the corners outside it
    #[test]
    fn bake_region_cuts_at_the_page_scale_and_whitens_outside() {
        // a 3080 × 1540 grey page: the page bakes at scale 0.5
        let page = RgbImage::from_pixel(3080, 1540, Rgb([100, 100, 100]));
        let rect = [(400.0, 200.0), (1000.0, 200.0), (1000.0, 600.0), (400.0, 600.0)];
        let (img, p) = bake_region(&page, &rect, Bake::PIXTRAL, &RegionInput::PLAIN).unwrap();
        assert_eq!((p.scaled_w, p.scaled_h), (300, 200));
        assert_eq!(img.width() % 28, 0);
        assert_eq!(*img.get_pixel(p.left + 150, p.top + 100), Rgb([100, 100, 100]));
        let tri = [(400.0, 200.0), (1000.0, 200.0), (400.0, 600.0)];
        let (img, p) = bake_region(&page, &tri, Bake::PIXTRAL, &RegionInput::PLAIN).unwrap();
        assert_eq!(*img.get_pixel(p.left + 5, p.top + 5), Rgb([100, 100, 100]), "inside, near the right angle");
        assert_eq!(*img.get_pixel(p.left + 295, p.top + 195), Rgb([255, 255, 255]), "outside, the far corner");
    }

    /// a region over the whole page is the page's bake, pixel for pixel (the per-axis rounding of a large page too)
    #[test]
    fn bake_region_of_the_whole_page_is_the_page_bake() {
        let page = RgbImage::from_fn(2001, 3003, |x, y| Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8]));
        let whole = [(0.0, 0.0), (2001.0, 0.0), (2001.0, 3003.0), (0.0, 3003.0)];
        let (region, rp) = bake_region(&page, &whole, Bake::PIXTRAL, &RegionInput::PLAIN).unwrap();
        let (baked, bp) = bake(&page, Bake::PIXTRAL);
        assert_eq!(rp, bp);
        assert!(region == baked, "the whole-page region differs from the page bake");
    }

    #[test]
    fn bake_region_refuses_too_few_corners_and_an_outside_region() {
        let page = RgbImage::from_pixel(200, 100, Rgb([0, 0, 0]));
        assert!(bake_region(&page, &[(0.0, 0.0), (10.0, 10.0)], Bake::PIXTRAL, &RegionInput::PLAIN).is_err());
        assert!(bake_region(&page, &[(300.0, 10.0), (400.0, 10.0), (400.0, 50.0)], Bake::PIXTRAL, &RegionInput::PLAIN).is_err());
    }

    /// contrast from inside the polygon (a dark border outside does not set the levels), the paper tone outside, the
    /// scale and a quarter turn
    #[test]
    fn bake_region_contrast_paper_scale_turn() {
        // paper 180 with ink strokes 90 inside x < 600; a black scanner border right of it (outside the polygon)
        let page = RgbImage::from_fn(800, 400, |x, y| {
            if x >= 600 {
                Rgb([0, 0, 0])
            } else if y % 20 < 3 {
                Rgb([90, 90, 90])
            } else {
                Rgb([180, 180, 180])
            }
        });
        let poly = [(0.0, 0.0), (590.0, 0.0), (590.0, 400.0), (0.0, 400.0), (0.0, 0.0)];
        let wide = [(0.0, 0.0), (800.0, 0.0), (800.0, 400.0), (0.0, 400.0)];
        let input = RegionInput { contrast: true, paper_fill: false, scale: 1.0, quarters: 0 };
        let (img, p) = bake_region(&page, &poly, Bake::PIXTRAL, &input).unwrap();
        assert_eq!(img.get_pixel(p.left + 10, p.top + 10).0, [255, 255, 255], "the paper stretched to white");
        assert_eq!(img.get_pixel(p.left + 10, p.top + 1).0, [0, 0, 0], "the ink to black");
        // the paper fill: the outside in the paper tone (here no contrast: 180)
        let paper = RegionInput { contrast: false, paper_fill: true, scale: 1.0, quarters: 0 };
        let tri = [(0.0, 0.0), (590.0, 0.0), (0.0, 390.0)];
        let (img, p) = bake_region(&page, &tri, Bake::PIXTRAL, &paper).unwrap();
        assert_eq!(img.get_pixel(p.left + 580, p.top + 380).0, [180, 180, 180], "outside in the paper tone");
        // scale ×0.5 and a quarter turn: 800 × 400 → 200 × 400
        let turned = RegionInput { contrast: false, paper_fill: false, scale: 0.5, quarters: 1 };
        let (_, p) = bake_region(&page, &wide, Bake::PIXTRAL, &turned).unwrap();
        assert_eq!((p.scaled_w, p.scaled_h), (200, 400));
    }
}
