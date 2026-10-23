// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// UI check (no GPU, no model): zoom (buttons, ctrl + wheel, pinch, double tap), the crop mapping at any zoom (mouse and
// touch), the pencil filter, the sharp preview, About and Settings, the phone layout — on the built site.
//   node tests/browser/ui.cjs
const { launch, serve, outDir, results, testImages } = require("./lib.cjs");
const R = results("ui");
const ok = R.ok;
(async () => {
  const OUT = outDir("ui");
  const srv = await serve(8801);
  const URL0 = `${srv.base}?gpuCheck=skip`;
  const browser = await launch();
  const { big: IMG } = await testImages(browser);
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 920, deviceScaleFactor: 1 });
  page.on("pageerror", (e) => console.log("pageerror", e.message));
  page.on("console", (m) => { if (m.type() === "error") console.log("console.error", m.text()); });
  // capture every prep chain the bridge hands to the image worker (the crop op): the ops of each worker message
  await page.evaluateOnNewDocument(() => {
    const post = Worker.prototype.postMessage;
    window.__ops = [];
    Worker.prototype.postMessage = function (m, ...r) {
      if (m && Array.isArray(m.ops)) window.__ops.push(JSON.stringify(m.ops));
      return post.call(this, m, ...r);
    };
  });
  await page.goto(URL0, { waitUntil: "networkidle0" });
  await page.waitForFunction(() => document.getElementById("status").textContent !== "Ready" || true);
  // the first-run setup opens by itself while the model is missing: closed here (this test needs no model)
  await new Promise((r) => setTimeout(r, 800));
  if (await page.evaluate(() => document.getElementById("setupDlg").open)) await page.click("#suClose");
  const [chooser] = await Promise.all([page.waitForFileChooser(), page.click("#openBtn")]);
  await chooser.accept([IMG]);
  await page.waitForFunction(() => { const i = document.getElementById("img"); return i.complete && i.naturalWidth > 0 && !document.getElementById("stage").hidden; });
  await new Promise((r) => setTimeout(r, 300));
  // the opened image lives in the image worker: the page's own wasm memory stays small (it held ~100 MiB for this
  // 3000 × 2000 image, ~400 MiB for a 24 MP photo, when the page decoded it itself)
  const pageWasm = await page.evaluate(async () => {
    const app = await (await fetch([...document.scripts].map((s) => s.src).find((s) => /app\./.test(s)))).text();
    const host = await import(new URL(app.match(/bridge-web\.[0-9a-f]+\.js/)[0], location.href).href);
    return host.pageWasmBytes();
  });
  ok(pageWasm < 16 * 2 ** 20, `the page's wasm memory after opening a 3000 × 2000 image: ${(pageWasm / 2 ** 20).toFixed(1)} MiB (the image is in the image worker)`);
  const st = () => page.evaluate(() => {
    const i = document.getElementById("img"), r = i.getBoundingClientRect(), p = document.getElementById("imgPane");
    return { w: r.width, h: r.height, left: r.left, top: r.top, nat: i.naturalWidth, zoom: document.getElementById("zoomVal").textContent,
      sl: p.scrollLeft, st: p.scrollTop, pane: p.getBoundingClientRect().toJSON(), info: document.getElementById("bakeInfo").textContent };
  });
  const fit = await st();
  console.log("fit", JSON.stringify(fit));
  await page.screenshot({ path: `${OUT}/01_fit.png` });
  for (let k = 0; k < 3; k++) await page.click("#zoomInBtn");
  let s = await st();
  ok(Math.abs(s.w / fit.w - 1.953125) < 0.002 && s.zoom === "195 %", `zoom in ×3 → ${s.zoom}, width ratio ${(s.w / fit.w).toFixed(4)}`);
  // the pane centre keeps its image point: centre fraction before = 0.5 (fit centred)
  const cx = s.pane.x + s.pane.width / 2, cy = s.pane.y + s.pane.height / 2;
  ok(Math.abs((cx - s.left) / s.w - 0.5) < 0.01 && Math.abs((cy - s.top) / s.h - 0.5) < 0.01, `zoom about the pane centre: centre at ${((cx - s.left) / s.w).toFixed(3)}, ${((cy - s.top) / s.h).toFixed(3)}`);
  await page.click("#zoomOutBtn");
  s = await st();
  ok(Math.abs(s.w / fit.w - 1.5625) < 0.002, `zoom out → ${s.zoom}`);
  await page.click("#zoomFitBtn");
  s = await st();
  ok(Math.abs(s.w - fit.w) < 0.5 && s.zoom === "100 %", `fit → ${s.zoom} width ${s.w}`);
  // ctrl + wheel about a pointer → exactly 200 %, the point under the pointer stays
  const px = fit.left + fit.w * 0.3, py = fit.top + fit.h * 0.4;
  await page.evaluate((x, y) => {
    const dy = -100 * Math.log(2) / Math.log(1.25);
    document.getElementById("imgPane").dispatchEvent(new WheelEvent("wheel", { ctrlKey: true, deltaY: dy, clientX: x, clientY: y, bubbles: true, cancelable: true }));
  }, px, py);
  s = await st();
  ok(s.zoom === "200 %" && Math.abs((px - s.left) / s.w - 0.3) < 0.003 && Math.abs((py - s.top) / s.h - 0.4) < 0.003,
    `ctrl+wheel → ${s.zoom}, pointer point kept (${((px - s.left) / s.w).toFixed(4)}, ${((py - s.top) / s.h).toFixed(4)})`);
  await page.screenshot({ path: `${OUT}/02_zoom200.png` });
  // outside the crop mode a mouse drag pans and draws nothing
  {
    const s0 = await st(), q = { x: s0.pane.x + s0.pane.width * 0.5, y: s0.pane.y + s0.pane.height * 0.5 };
    await page.mouse.move(q.x, q.y); await page.mouse.down(); await page.mouse.move(q.x - 60, q.y - 40, { steps: 5 }); await page.mouse.up();
    const s1 = await st(), rectVis = await page.evaluate(() => !document.getElementById("rect").hidden);
    ok(!rectVis && Math.abs(s1.sl - s0.sl - 60) < 2 && Math.abs(s1.st - s0.st - 40) < 2, `no crop mode: drag pans +${(s1.sl - s0.sl).toFixed(0)}, +${(s1.st - s0.st).toFixed(0)}, no rect`);
    await page.evaluate(() => { const p = document.getElementById("imgPane"); p.scrollLeft -= 60; p.scrollTop -= 40; });
  }
  await page.click("#autoCropBtn"); await page.click("#drawCropBtn");
  ok(await page.evaluate(() => document.getElementById("stage").classList.contains("cropping") && document.getElementById("autoCropMenu").hidden), "Draw crop → crop mode, menu closed");
  s = await st();
  // draw a crop at 200 %: from 25 % / 30 % of the pane to 70 % / 75 %
  const W = 3000, H = 2000;
  const a = { x: s.pane.x + s.pane.width * 0.25, y: s.pane.y + s.pane.height * 0.3 }, b = { x: s.pane.x + s.pane.width * 0.7, y: s.pane.y + s.pane.height * 0.75 };
  await page.mouse.move(a.x, a.y); await page.mouse.down(); await page.mouse.move((a.x + b.x) / 2, (a.y + b.y) / 2, { steps: 4 }); await page.mouse.move(b.x, b.y, { steps: 4 }); await page.mouse.up();
  s = await st();
  const D = s.w / W;
  const clampx = (v) => Math.min(W, Math.max(0, v)), clampy = (v) => Math.min(H, Math.max(0, v));
  const ex0 = Math.round(clampx((a.x - s.left) / D)), ey0 = Math.round(clampy((a.y - s.top) / D));
  const ex1 = Math.round(clampx((b.x - s.left) / D)), ey1 = Math.round(clampy((b.y - s.top) / D));
  const expect = { x: ex0, y: ey0, w: ex1 - ex0, h: ey1 - ey0 };
  const rectCss = await page.evaluate(() => document.getElementById("rect").getBoundingClientRect().toJSON());
  ok(Math.abs(rectCss.left - (s.left + ex0 * D)) < 1.5 && Math.abs(rectCss.width - expect.w * D) < 1.5, `pending rect drawn at the zoomed position (${rectCss.left.toFixed(1)} vs ${(s.left + ex0 * D).toFixed(1)})`);
  const pill = await page.evaluate(() => { const p = document.getElementById("cropPill").getBoundingClientRect(), q = document.getElementById("imgPane").getBoundingClientRect(); return { vis: !document.getElementById("cropPill").hidden, inside: p.left >= q.left && p.right <= q.right && p.top >= q.top && p.bottom <= q.bottom }; });
  ok(pill.vis && pill.inside, `crop pill visible inside the pane`);
  await page.screenshot({ path: `${OUT}/03_crop_pending_200.png` });
  // scroll the pane: the pill stays inside the visible pane
  await page.evaluate(() => { document.getElementById("imgPane").scrollTop += 120; document.getElementById("imgPane").dispatchEvent(new Event("scroll")); });
  const pill2 = await page.evaluate(() => { const p = document.getElementById("cropPill").getBoundingClientRect(), q = document.getElementById("imgPane").getBoundingClientRect(); return p.top >= q.top && p.bottom <= q.bottom; });
  ok(pill2, "pill follows a scroll");
  await page.evaluate(() => { window.__ops = []; });
  await page.click("#cropApply");
  await page.waitForFunction(() => window.__ops.some((o) => o.includes("crop")));
  const got = JSON.parse((await page.evaluate(() => window.__ops.find((o) => o.includes("crop")))))[0];
  ok(got.x === expect.x && got.y === expect.y && got.w === expect.w && got.h === expect.h, `crop at 200 %: got ${JSON.stringify(got)} expected ${JSON.stringify(expect)}`);
  await page.waitForFunction((w) => document.getElementById("bakeInfo").textContent.startsWith(`${w} ×`), {}, expect.w);
  s = await st();
  ok(!(await page.evaluate(() => document.getElementById("stage").classList.contains("cropping"))), "after Crop the mode is left");
  ok(s.zoom === "100 %", `after the crop: back to fit (${s.zoom}), ${s.info}`);
  await page.screenshot({ path: `${OUT}/04_cropped.png` });
  // reset, pencil
  await page.click("#resetBtn");
  await page.waitForFunction(() => document.getElementById("bakeInfo").textContent.startsWith("3000 ×"));
  await new Promise((r) => setTimeout(r, 300));
  const sample = () => page.evaluate(async () => {
    const i = document.getElementById("img");
    await i.decode();
    const c = document.createElement("canvas"); c.width = i.naturalWidth; c.height = i.naturalHeight;
    const x = c.getContext("2d"); x.drawImage(i, 0, 0);
    const pts = [[0.5, 0.02], [0.05, 0.5], [0.95, 0.5]].map(([fx, fy]) => [...x.getImageData(Math.round(fx * (c.width - 1)), Math.round(fy * (c.height - 1)), 1, 1).data].slice(0, 3));
    return { src: i.src.slice(0, 40), pts };
  });
  const before = await sample();
  await page.click('[data-op="pencil"]');
  await page.waitForFunction(() => document.getElementById("opsChips").textContent.includes("pencil"));
  await new Promise((r) => setTimeout(r, 400));
  const after = await sample();
  console.log("before", JSON.stringify(before.pts), "after", JSON.stringify(after.pts));
  ok(after.pts.every((p) => Math.abs(p[0] - p[1]) <= 3 && Math.abs(p[1] - p[2]) <= 3 && p[1] >= 245), "pencil: paper → grey ≈ 255 everywhere (left and right of the gradient)");
  ok(before.pts[1][1] < 200, "before pencil the left paper was darker");
  await page.screenshot({ path: `${OUT}/05_pencil.png` });
  // sharp preview at 8× (keeps the pencil chain)
  for (let k = 0; k < 10; k++) await page.click("#zoomInBtn");
  await page.waitForFunction(() => document.getElementById("img").naturalWidth === 3000, { timeout: 15000 }).catch(() => {});
  s = await st();
  ok(s.zoom === "800 %" && s.nat === 3000, `8× → sharp preview natural width ${s.nat} (normal 1800)`);
  await page.screenshot({ path: `${OUT}/06_zoom800_sharp.png` });
  // touch: two-finger pinch about a midpoint, then a two-finger pan
  await page.click("#zoomFitBtn");
  const cdp = await page.createCDPSession();
  s = await st();
  const m = { x: s.pane.x + s.pane.width * 0.5, y: s.pane.y + s.pane.height * 0.5 };
  const tp = (d, dx = 0, dy = 0) => [{ x: m.x - d + dx, y: m.y + dy, id: 1 }, { x: m.x + d + dx, y: m.y + dy, id: 2 }];
  await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: tp(50) });
  for (let d = 55; d <= 150; d += 5) await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: tp(d) });
  await new Promise((r) => setTimeout(r, 200));
  s = await st();
  const pinchZoom = s.zoom;
  ok(s.zoom === "300 %", `pinch 100 → 300 px finger spread → ${s.zoom}`);
  ok(Math.abs((m.x - s.left) / s.w - 0.5) < 0.01, `pinch about the fingers' midpoint (${((m.x - s.left) / s.w).toFixed(3)})`);
  const sl0 = s.sl, st0 = s.st;
  for (let k = 1; k <= 10; k++) await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: tp(150, -8 * k, -6 * k) });
  await new Promise((r) => setTimeout(r, 200));
  await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  s = await st();
  ok(s.zoom === pinchZoom && Math.abs(s.sl - sl0 - 80) < 2 && Math.abs(s.st - st0 - 60) < 2, `two-finger pan → scroll +${(s.sl - sl0).toFixed(0)}, +${(s.st - st0).toFixed(0)} (expected 80, 60)`);
  await page.screenshot({ path: `${OUT}/07_pinch.png` });
  // one finger outside the crop mode pans, draws nothing
  {
    const s0 = await st(), g = { x: s0.pane.x + 300, y: s0.pane.y + 250 };
    await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ ...g, id: 5 }] });
    for (let k = 1; k <= 5; k++) await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [{ x: g.x + 10 * k, y: g.y + 8 * k, id: 5 }] });
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    await new Promise((r) => setTimeout(r, 200));
    const s1 = await st(), rectVis = await page.evaluate(() => !document.getElementById("rect").hidden);
    ok(!rectVis && Math.abs(s1.sl - s0.sl + 50) < 2 && Math.abs(s1.st - s0.st + 40) < 2, `no crop mode: one finger pans ${(s1.sl - s0.sl).toFixed(0)}, ${(s1.st - s0.st).toFixed(0)}, no rect`);
    s = s1;
  }
  // Esc leaves the crop mode
  await page.click("#autoCropBtn"); await page.click("#drawCropBtn");
  await page.keyboard.press("Escape");
  ok(!(await page.evaluate(() => document.getElementById("stage").classList.contains("cropping"))), "Esc leaves the crop mode");
  await page.click("#autoCropBtn"); await page.click("#drawCropBtn");
  // one finger draws a crop
  await page.evaluate(() => { window.__ops = []; });
  const f0 = { x: s.pane.x + 200, y: s.pane.y + 150 }, f1 = { x: s.pane.x + 420, y: s.pane.y + 330 };
  await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ ...f0, id: 3 }] });
  for (let k = 1; k <= 6; k++) await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [{ x: f0.x + (f1.x - f0.x) * k / 6, y: f0.y + (f1.y - f0.y) * k / 6, id: 3 }] });
  await new Promise((r) => setTimeout(r, 200));
  await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  await new Promise((r) => setTimeout(r, 200));
  const s2 = await st();
  const D2 = s2.w / W;
  const e2 = { x: Math.round((f0.x - s2.left) / D2), y: Math.round((f0.y - s2.top) / D2) };
  e2.w = Math.round((f1.x - s2.left) / D2) - e2.x; e2.h = Math.round((f1.y - s2.top) / D2) - e2.y;
  await page.click("#cropApply");
  await page.waitForFunction(() => window.__ops.some((o) => o.includes("crop")));
  const got2 = JSON.parse((await page.evaluate(() => window.__ops.find((o) => o.includes("crop"))))).find((o) => o.op === "crop");
  ok(got2.x === e2.x && got2.y === e2.y && got2.w === e2.w && got2.h === e2.h, `one-finger crop at ${pinchZoom}: got ${JSON.stringify(got2)} expected ${JSON.stringify(e2)}`);
  // double tap = fit
  await page.waitForFunction(() => document.getElementById("zoomVal").textContent === "100 %");
  for (let k = 0; k < 2; k++) await page.click("#zoomInBtn");
  for (let k = 0; k < 2; k++) {
    await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ ...m, id: 5 + k }] });
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    await new Promise((r) => setTimeout(r, 80));
  }
  s = await st();
  ok(s.zoom === "100 %", `double tap → ${s.zoom}`);
  // about
  await page.click("#aboutBtn");
  await new Promise((r) => setTimeout(r, 300));
  const about = await page.evaluate(() => {
    const d = document.getElementById("aboutDlg");
    const vis = [...d.querySelectorAll(".lic li")].filter((li) => !li.hidden).map((li) => li.textContent);
    return { open: d.open, text: d.querySelector(".lic").innerText, vis, links: [...d.querySelectorAll(".lic a")].map((a) => `${a.target} ${a.href}`) };
  });
  ok(about.open && about.text.includes("MIT") && about.text.includes("CC BY-SA 4.0") && about.text.includes("Staatsbibliothek zu Berlin – Preußischer Kulturbesitz"), "About shows the licences");
  ok(about.vis.some((t) => t.startsWith("wasm-bindgen")) && !about.vis.some((t) => t.startsWith("llama.cpp")) && !about.vis.some((t) => t.startsWith("Tauri")), "components: wasm-bindgen listed, no llama.cpp / Tauri");
  ok(about.links.every((l) => l.startsWith("_blank ")), `links open a new tab: ${about.links.join(" | ")}`);
  await page.screenshot({ path: `${OUT}/08_about.png` });
  await page.keyboard.press("Escape");
  // settings presets
  await page.click("#settingsBtn");
  await new Promise((r) => setTimeout(r, 400));
  const cut = await page.evaluate(() => [...document.querySelectorAll("#presetTable tr td:first-child input")].map((i) => ({ v: i.value, cut: i.scrollWidth > i.clientWidth })));
  ok(cut.length > 0 && cut.every((c) => !c.cut), `preset names not cut: ${cut.map((c) => c.v + (c.cut ? "(CUT)" : "")).join(", ")}`);
  await page.screenshot({ path: `${OUT}/09_settings.png` });
  await page.keyboard.press("Escape");
  const sel = await page.evaluate(() => { const s = document.getElementById("presetSel"); return { sw: s.scrollWidth, cw: s.clientWidth, v: s.value, w: s.getBoundingClientRect().width }; });
  console.log("preset select", JSON.stringify(sel));
  for (const w of [1366, 1180, 1032, 1024, 834, 820]) {
    await page.setViewport({ width: w, height: 900, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    await new Promise((r) => setTimeout(r, 200));
    const o = await page.evaluate(() => ({ page: document.documentElement.scrollWidth, bar: document.querySelector(".titlebar").scrollWidth, vw: innerWidth }));
    console.log(`width ${w}: page / titlebar scrollWidth ${o.page} / ${o.bar} (viewport ${o.vw})`);
  }
  // iPad portrait-ish + phone width
  for (const [n, w, h] of [["10_ipad_portrait", 820, 1180], ["11_phone", 390, 844]]) {
    await page.setViewport({ width: w, height: h, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    await new Promise((r) => setTimeout(r, 400));
    await page.screenshot({ path: `${OUT}/${n}.png` });
  }
  await page.click("#aboutBtn");
  await new Promise((r) => setTimeout(r, 300));
  const scroll = await page.evaluate(() => { const f = document.querySelector("#aboutDlg form"); return { sh: f.scrollHeight, ch: f.clientHeight, vh: innerHeight }; });
  console.log("about on the phone", JSON.stringify(scroll));
  ok(scroll.ch <= scroll.vh * 0.9 + 1, "about dialog fits the phone screen (body scrolls)");
  await page.screenshot({ path: `${OUT}/12_phone_about.png` });
  await browser.close();
  srv.kill();
  R.done();
})().catch((e) => { console.log("ERROR", e); process.exit(2); });
