// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The text crop's sides on a synthetic spread: two written pages, a dark spine between them — Left / Right keep one
// page each, Double both, all three find the spine.
use ocr_layout::crop::{Side, text_crop_side};

const W: usize = 1400;
const H: usize = 1000;
const SPINE: (usize, usize) = (690, 710);

/// two pages of "words" (letter-sized dark blocks in lines) beside a spine → RGB
fn spread() -> Vec<u8> {
    let mut g = vec![225u8; W * H];
    for y in 0..H {
        for x in SPINE.0..SPINE.1 {
            g[y * W + x] = 120;
        }
    }
    for page_x in [80, 770] {
        for line in 0..20 {
            let y0 = 100 + line * 40;
            let mut x = page_x;
            let mut k = 0;
            while x + 10 < page_x + 560 {
                // a word of 4 letters, then a space
                if k % 5 != 4 {
                    for y in y0..y0 + 14 {
                        for xx in x..x + 8 {
                            g[y * W + xx] = 30;
                        }
                    }
                }
                x += 12;
                k += 1;
            }
        }
    }
    g.iter().flat_map(|&v| [v, v, v]).collect()
}

#[test]
fn left_right_double_split_at_the_spine() {
    let rgb = spread();
    let near = |g: Option<f64>| g.is_some_and(|g| (g - 700.0).abs() < 15.0);
    let l = text_crop_side(&rgb, W, H, Side::Left).expect("left");
    assert!(near(l.gutter), "left gutter {:?}", l.gutter);
    assert!(l.b[0] < 100.0 && l.b[2] <= 700.0, "left {:?}", l.b);
    let r = text_crop_side(&rgb, W, H, Side::Right).expect("right");
    assert!(near(r.gutter), "right gutter {:?}", r.gutter);
    assert!(r.b[0] >= 690.0 && r.b[2] > 1300.0, "right {:?}", r.b);
    let d = text_crop_side(&rgb, W, H, Side::Double).expect("double");
    assert!(near(d.gutter), "double gutter {:?}", d.gutter);
    assert!(d.b[0] < 100.0 && d.b[2] > 1300.0, "double {:?}", d.b);
}
