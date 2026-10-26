// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The desktop GT tools' region geometry self-test, case for case: simplify keeps the ink, merge = one valid polygon,
// split of a convex and a concave polygon, rejects, node editing, the knife, the hull, the border, straighten, sides,
// subtract.
use ocr_layout::geom::*;
use ocr_layout::shape::*;

fn p(x: f64, y: f64) -> Pt {
    Pt::new(x, y)
}

fn sq(x: f64, y: f64, w: f64) -> Vec<Pt> {
    vec![p(x, y), p(x + w, y), p(x + w, y + w), p(x, y + w)]
}

fn u_shape() -> Vec<Pt> {
    vec![p(0., 0.), p(30., 0.), p(30., 70.), p(70., 70.), p(70., 0.), p(100., 0.), p(100., 100.), p(0., 100.)]
}

#[test]
fn simplify_a_wobbly_ring_keeps_the_ink() {
    let ring: Vec<Pt> = (0..240)
        .map(|k| {
            let a = k as f64 / 240.0 * 2.0 * std::f64::consts::PI;
            let r = 100.0 + 1.5 * (k as f64 * 1.7).sin() + if k % 7 == 0 { 1.2 } else { 0.0 };
            p(200.0 + r * a.cos(), 200.0 + r * a.sin())
        })
        .collect();
    let ink: Vec<Pt> = (0..120)
        .map(|k| {
            let a = k as f64 / 120.0 * 2.0 * std::f64::consts::PI;
            p(200.0 + 95.0 * a.cos(), 200.0 + 95.0 * a.sin())
        })
        .collect();
    let s = simplify_outward(&ring, 4.0);
    assert!(s.len() < 60, "{} corners", s.len());
    assert!(is_simple(&s));
    assert!(ink.iter().all(|&q| contains(&s, q)));
    assert!(ring.iter().all(|&q| covers(&s, q, 0.5)));
}

#[test]
fn simplify_a_concave_comb_does_not_cut_the_teeth() {
    let mut comb = vec![p(0., 0.)];
    for k in 0..10 {
        let x = k as f64 * 20.0;
        comb.extend([p(x + 2., 0.), p(x + 2., -1.), p(x + 10., -1.), p(x + 10., 0.)]);
    }
    comb.extend([p(200., 0.), p(200., 60.), p(0., 60.)]);
    let s = simplify_outward(&comb, 4.0);
    assert!(is_simple(&s));
    assert!(comb.iter().all(|&q| covers(&s, q, 0.5)));
}

#[test]
fn merge_overlapping_and_apart() {
    let (a, b) = (sq(0., 0., 100.), sq(60., 60., 100.));
    let u = union(&[a.clone(), b.clone()], 2.0).unwrap();
    assert!(is_simple(&u) && a.iter().chain(&b).all(|&q| covers(&u, q, 1.5)), "{u:?}");
    let (c, d) = (sq(0., 0., 50.), sq(120., 10., 50.));
    let v = union(&[c.clone(), d.clone()], 2.0).unwrap();
    assert!(is_simple(&v) && c.iter().chain(&d).all(|&q| covers(&v, q, 1.5)), "{v:?}");
}

#[test]
fn split_convex_upper_first() {
    let (a, b) = split(&sq(0., 0., 100.), &[p(-1., 50.), p(50., 52.), p(101., 50.)], 3.0).unwrap();
    assert!(is_simple(&a) && is_simple(&b));
    assert!((signed_area(&a).abs() + signed_area(&b).abs() - 10000.0).abs() < 1.0);
    assert!(centroid(&a).y < centroid(&b).y);
}

#[test]
fn split_concave_u_across_its_right_arm() {
    let u = u_shape();
    let (a, b) = split(&u, &[p(70., 30.), p(100., 30.)], 2.0).unwrap();
    let (ua, aa, ab) = (signed_area(&u).abs(), signed_area(&a).abs(), signed_area(&b).abs());
    assert!(is_simple(&a) && is_simple(&b) && (aa + ab - ua).abs() < 1.0 && aa.min(ab) == 900.0, "{aa} + {ab} = {ua}");
}

#[test]
fn split_along_a_five_point_polyline() {
    let zig = [p(0., 40.), p(25., 60.), p(50., 30.), p(75., 70.), p(100., 45.)];
    let (a, b) = split(&sq(0., 0., 100.), &zig, 2.0).unwrap();
    assert!(is_simple(&a) && is_simple(&b));
    assert!((signed_area(&a).abs() + signed_area(&b).abs() - 10000.0).abs() < 1.0);
    assert_eq!((a.len(), b.len()), (7, 7));
}

#[test]
fn split_rejects() {
    assert!(split(&sq(0., 0., 100.), &[p(50., 50.), p(100., 50.)], 2.0).is_err(), "a start inside");
    assert!(split(&u_shape(), &[p(0., 30.), p(100., 30.)], 2.0).is_err(), "across the U's gap");
    assert!(split(&sq(0., 0., 100.), &[p(0., 0.), p(100., 0.)], 2.0).is_err(), "along the outline");
}

#[test]
fn node_editing() {
    let (ins, k) = insert_node(&sq(0., 0., 100.), p(50., 1.), 3.0, true).unwrap();
    assert_eq!((ins.len(), k, ins[1]), (5, 1, p(50., 0.)));
    assert_eq!(delete_node(&ins, k, true).unwrap(), sq(0., 0., 100.));
    assert!(insert_node(&sq(0., 0., 100.), p(50., 50.), 3.0, true).is_none(), "no insert far from every edge");
    assert!(delete_node(&[p(0., 0.), p(10., 0.), p(0., 10.)], 0, true).is_err(), "a polygon keeps 3 corners");
    let notch = [p(0., 0.), p(100., 0.), p(100., 100.), p(60., 100.), p(60., 20.), p(40., 20.), p(40., 100.), p(0., 100.)];
    assert!(delete_node(&notch, 1, true).is_err(), "a deletion that makes the outline cross");
    let (cl, ck) = insert_node(&[p(0., 0.), p(100., 0.)], p(30., 2.), 3.0, false).unwrap();
    assert_eq!((cl.len(), ck), (3, 1));
}

#[test]
fn knife() {
    let cut = clip_cut(&sq(0., 0., 100.), &[p(-40., 50.), p(160., 60.)]).unwrap();
    assert!(cut.len() == 2 && cut[0].x.abs() < 1e-6 && (cut[1].x - 100.0).abs() < 1e-6, "{cut:?}");
    let (a, b) = split(&sq(0., 0., 100.), &cut, 1.0).unwrap();
    assert!((signed_area(&a).abs() + signed_area(&b).abs() - 10000.0).abs() < 1.0);
    let bent = clip_cut(&sq(0., 0., 100.), &[p(-20., 20.), p(50., 50.), p(120., 30.)]).unwrap();
    assert!(bent.len() == 3 && bent[1] == p(50., 50.), "{bent:?}");
    assert!(clip_cut(&sq(0., 0., 100.), &[p(-20., 150.), p(120., 160.)]).is_none(), "a line that misses");
    let inside = clip_cut(&sq(0., 0., 100.), &[p(-20., 50.), p(50., 50.)]).unwrap();
    assert!(inside[0].x.abs() < 1e-6 && (inside[inside.len() - 1].x - 100.0).abs() < 1e-6, "an end inside is extended: {inside:?}");
    let down = clip_cut(&sq(0., 0., 100.), &[p(50., 5.), p(50., 130.)]).unwrap();
    assert!(down[0].y.abs() < 1e-6 && (down[down.len() - 1].y - 100.0).abs() < 1e-6, "starting just inside: {down:?}");
}

#[test]
fn merged_hull_leaves_the_other_region_out() {
    let cells = [sq(0., 0., 40.), sq(0., 60., 40.), sq(60., 60., 40.)];
    let other = sq(60., 0., 40.);
    let h = merged_hull(&cells, std::slice::from_ref(&other), 2.0, 2.0).unwrap();
    assert!(h.len() <= 12, "{} corners", h.len());
    assert!(!contains(&h, p(80., 20.)) && !other.iter().any(|&q| contains(&h, q)));
    assert!(cells.iter().flatten().all(|&q| covers(&h, q, 2.0)));
    assert!(merged_hull(&[sq(0., 0., 40.), sq(120., 0., 40.)], &[sq(50., -20., 60.)], 2.0, 2.0).is_err(), "a region between the parts");
}

#[test]
fn snap_to_border_runs_free_sides_out() {
    let page = Rect { x: 0., y: 0., w: 200., h: 200. };
    let s = snap_to_border(&sq(40., 20., 60.), &[sq(40., 120., 60.)], page, 2.0, 2.0);
    let b = bounds(&s);
    assert!(b.y < 1.0 && b.x < 1.0 && b.max_x() > 199.0 && b.max_y() < 110.0, "{b:?}");
}

#[test]
fn straighten_moves_the_edge_out_to_the_widest_corner() {
    let bump = [p(0., 0.), p(100., 0.), p(100., 100.), p(0., 100.), p(-10., 60.), p(5., 40.)];
    let r = straighten(&bump, 3, 0).unwrap();
    assert!(r.iter().all(|q| q.x >= -10.01) && r.iter().any(|q| (q.x + 10.0).abs() < 0.01), "{r:?}");
    assert!(!r.contains(&p(5., 40.)) && is_simple(&r));
}

#[test]
fn straight_sides_left() {
    let ragged = [p(20., 0.), p(100., 0.), p(100., 100.), p(10., 100.), p(30., 70.), p(0., 40.)];
    let r = straighten_sides(&ragged, &[Side::Left], 1.0).unwrap();
    let lefts: Vec<f64> = r.iter().filter(|q| q.x < 50.0).map(|q| q.x).collect();
    assert!(lefts.iter().all(|x| x.abs() < 1.5), "left x {lefts:?}");
    let b = bounds(&r);
    assert!(b.max_x() > 99.0 && b.max_x() < 101.5, "{b:?}");
}

#[test]
fn subtract_cuts_the_spike_back_with_the_gap() {
    let b = [p(0., 0.), p(50., 0.), p(50., 40.), p(70., 50.), p(50., 60.), p(50., 100.), p(0., 100.)];
    let r = subtract(&b, &sq(60., 0., 100.), 3.0, 1.0).unwrap();
    let bb = bounds(&r);
    assert!(bb.max_x() <= 57.5 && bb.max_x() >= 49.0 && bb.h > 99.0, "{bb:?}");
}

#[test]
fn clear_of_others_cuts_a_neighbour_out() {
    let grown = sq(0., 0., 100.);
    let r = clear_of_others(&grown, &[sq(0., 0., 50.)], &[sq(70., 0., 50.)], 1.0);
    // the neighbour (x 70…120, y 0…50) is cut out: an L — the region below it stays
    assert!(!contains(&r, p(85., 25.)) && contains(&r, p(85., 75.)) && contains(&r, p(25., 25.)), "{r:?}");
}
