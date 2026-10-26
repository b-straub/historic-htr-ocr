// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// UI check (no GPU, no model): zoom (buttons, ctrl + wheel, pinch, double tap), the crop mapping at any zoom (mouse and
// touch), the pencil filter, the sharp preview, About and Settings, the phone layout — on the built site.
//   node tests/browser/ui.cjs
const path = require("path");
const { launch, serve, outDir, results, testImages, REPO } = require("./lib.cjs");
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
  // Draw in the crop bar (open after a load, closed by a Crop — the crop button opens it again)
  const drawMode = async () => {
    if (await page.evaluate(() => document.getElementById("cropBar").hidden)) {
      await page.click("#autoCropBtn");
    }
    await page.click("#drawCropBtn");
  };
  const bar0 = await page.evaluate(() => !document.getElementById("cropBar").hidden); // open after the load
  // the zoom buttons lie under the open crop bar (it covers the image head): closed first, as by hand
  const zoomBtn = async (id) => {
    if (!(await page.evaluate(() => document.getElementById("cropBar").hidden))) {
      await page.click("#autoCropBtn");
    }
    await page.click(id);
  };
  const fit = await st();
  console.log("fit", JSON.stringify(fit));
  await page.screenshot({ path: `${OUT}/01_fit.png` });
  for (let k = 0; k < 3; k++) await zoomBtn("#zoomInBtn");
  let s = await st();
  ok(Math.abs(s.w / fit.w - 1.953125) < 0.002 && s.zoom === "195 %", `zoom in ×3 → ${s.zoom}, width ratio ${(s.w / fit.w).toFixed(4)}`);
  // the pane centre keeps its image point: centre fraction before = 0.5 (fit centred)
  const cx = s.pane.x + s.pane.width / 2, cy = s.pane.y + s.pane.height / 2;
  ok(Math.abs((cx - s.left) / s.w - 0.5) < 0.01 && Math.abs((cy - s.top) / s.h - 0.5) < 0.01, `zoom about the pane centre: centre at ${((cx - s.left) / s.w).toFixed(3)}, ${((cy - s.top) / s.h).toFixed(3)}`);
  await zoomBtn("#zoomOutBtn");
  s = await st();
  ok(Math.abs(s.w / fit.w - 1.5625) < 0.002, `zoom out → ${s.zoom}`);
  await zoomBtn("#zoomFitBtn");
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
  // the crop bar is open after the load; a page mode finds the paper at once (a dashed, adjustable rectangle) or says
  // there is nothing to trim; the crop button closes the bar and drops the rectangle
  await page.click("#autoCropBtn"); // the crop button opens the bar again
  const marked0 = await page.evaluate(() => document.querySelectorAll("#pageSideSeg .on").length);
  await page.click('#pageSideSeg [data-side=""]');
  await page.waitForFunction(() => /^Auto crop/.test(document.getElementById("status").textContent), { timeout: 30000 });
  const auto = await page.evaluate(() => ({ st: document.getElementById("status").textContent, rect: !document.getElementById("rect").hidden, act: !document.getElementById("cropAct").hidden,
    crop: !document.getElementById("cropApplyBtn").hidden, text: document.getElementById("cropText").textContent, mode: document.getElementById("stage").classList.contains("cropping"),
    marked: [...document.querySelectorAll("#pageSideSeg .on")].map((b) => b.textContent).join() }));
  await page.click("#drawCropBtn");
  const swap = await page.evaluate(() => ({ marked: [...document.querySelectorAll("#pageSideSeg .on")].length, draw: document.getElementById("drawCropBtn").classList.contains("on"), rect: !document.getElementById("rect").hidden }));
  ok(swap.marked === 0 && swap.draw && !swap.rect, `Draw after a found rectangle: page mode unmarked, Draw marked, the found rectangle gone ${JSON.stringify(swap)}`);
  await page.click("#autoCropBtn");
  const closed = await page.evaluate(() => document.getElementById("cropBar").hidden && document.getElementById("rect").hidden);
  ok(bar0 && closed && marked0 === 0 && (/nothing to trim/.test(auto.st) || (auto.rect && auto.act && auto.crop && auto.mode && /^Found /.test(auto.text) && auto.marked === "Auto")),
    `crop bar open after the load, no page mode marked before a find (${marked0}); Auto → ${auto.st} (${auto.text}, marked ${auto.marked}); the crop button closes the bar`);
  await drawMode();
  const ctl0 = await page.evaluate(() => ({ mode: document.getElementById("stage").classList.contains("cropping"), draw: document.getElementById("drawCropBtn").classList.contains("on"),
    act: !document.getElementById("cropAct").hidden, crop: !document.getElementById("cropApplyBtn").hidden, text: document.getElementById("cropText").textContent }));
  ok(ctl0.mode && ctl0.draw && ctl0.act && !ctl0.crop, `Draw → crop mode, Draw on, "${ctl0.text}" + Cancel, no Crop before a rectangle`);
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
  const ctl1 = await page.evaluate(() => { const top = (id) => document.getElementById(id).getBoundingClientRect().top;
    return { act: !document.getElementById("cropAct").hidden, crop: !document.getElementById("cropApplyBtn").hidden, text: document.getElementById("cropText").textContent,
      oneLine: Math.abs(top("resetBtn") - top("cropApplyBtn")) < 2 && Math.abs(top("drawCropBtn") - top("cropApplyBtn")) < 4 }; });
  ok(ctl1.act && ctl1.crop && ctl1.oneLine, `a drawn rectangle: "${ctl1.text}", Cancel + Crop, all on one line with Reset ${ctl1.oneLine}`);
  await page.screenshot({ path: `${OUT}/03_crop_pending_200.png` });
  // adjust it: its right edge dragged 40 px left
  {
    const r = await page.evaluate(() => document.getElementById("rect").getBoundingClientRect().toJSON());
    const y = r.top + r.height / 3;
    // the mouse's cursor says what a press does: resize at the edge, move inside, draw anew outside
    const cur = async (x, yy) => { await page.mouse.move(x, yy); return page.evaluate(() => document.getElementById("stage").style.cursor); };
    const curs = [await cur(r.right, y), await cur(r.right, r.bottom), await cur(r.left + r.width / 2, r.top + r.height / 2), await cur(r.right + 60, y)];
    ok(curs[0] === "ew-resize" && curs[1] === "nwse-resize" && curs[2] === "move" && curs[3] === "", `cursors: edge ${curs[0]}, corner ${curs[1]}, inside ${curs[2]}, outside "${curs[3]}" (crosshair)`);
    await page.mouse.move(r.right, y); await page.mouse.down(); await page.mouse.move(r.right - 40, y, { steps: 4 }); await page.mouse.up();
    const r2 = await page.evaluate(() => document.getElementById("rect").getBoundingClientRect().toJSON());
    ok(Math.abs(r2.left - r.left) < 1 && Math.abs(r.width - r2.width - 40) < 1.5, `crop adjusted by its right edge: width ${r.width.toFixed(0)} → ${r2.width.toFixed(0)}, left kept`);
    expect.w = Math.round(ex1 - 40 / D) - ex0;
  }
  await page.evaluate(() => { window.__ops = []; });
  await page.click("#cropApplyBtn");
  await page.waitForFunction(() => window.__ops.some((o) => o.includes("crop")));
  const got = JSON.parse((await page.evaluate(() => window.__ops.find((o) => o.includes("crop")))))[0];
  ok(got.x === expect.x && got.y === expect.y && got.w === expect.w && got.h === expect.h, `crop at 200 %: got ${JSON.stringify(got)} expected ${JSON.stringify(expect)}`);
  await page.waitForFunction((w) => document.getElementById("bakeInfo").textContent.startsWith(`${w} ×`), {}, expect.w);
  s = await st();
  ok(await page.evaluate(() => !document.getElementById("stage").classList.contains("cropping") && document.getElementById("cropBar").hidden), "after Crop the mode is left, the crop bar closed");
  ok(s.zoom === "100 %", `after the crop: back to fit (${s.zoom}), ${s.info}`);
  await page.screenshot({ path: `${OUT}/04_cropped.png` });
  // a cropped image: the crop bar's tools off, its Reset back to the original
  await page.click("#autoCropBtn");
  const done = await page.evaluate(() => ({ draw: document.getElementById("drawCropBtn").disabled, modes: [...document.querySelectorAll("#pageSideSeg button")].every((b) => b.disabled),
    reset: !document.getElementById("resetBtn").disabled, said: !document.getElementById("cropDone").hidden }));
  ok(done.draw && done.modes && done.reset && done.said, `cropped: Draw + page modes off, "Cropped" + Reset in the crop bar ${JSON.stringify(done)}`);
  const resetAt = await page.evaluate(() => { const r = document.getElementById("resetBtn").getBoundingClientRect(), b = document.getElementById("cropBar").getBoundingClientRect(); return Math.round(b.right - r.right); });
  const doneText = await page.evaluate(() => document.getElementById("cropDone").textContent);
  ok(resetAt <= 12 && /^Cropped to \d+ × \d+$/.test(doneText), `desktop: Reset at the crop bar's right end (${resetAt} px in), "${doneText}"`);
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
  for (let k = 0; k < 10; k++) await zoomBtn("#zoomInBtn");
  await page.waitForFunction(() => document.getElementById("img").naturalWidth === 3000, { timeout: 15000 }).catch(() => {});
  s = await st();
  ok(s.zoom === "800 %" && s.nat === 3000, `8× → sharp preview natural width ${s.nat} (normal 1800)`);
  await page.screenshot({ path: `${OUT}/06_zoom800_sharp.png` });
  // touch: two-finger pinch about a midpoint, then a two-finger pan
  await zoomBtn("#zoomFitBtn");
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
  // zoomed in, Draw crop leaves the image where it is
  await zoomBtn("#zoomInBtn");
  const at0 = await page.evaluate(() => document.getElementById("img").getBoundingClientRect().toJSON());
  await drawMode();
  await new Promise((r) => setTimeout(r, 300));
  const at1 = await page.evaluate(() => document.getElementById("img").getBoundingClientRect().toJSON());
  await page.click("#cropCancelBtn");
  await zoomBtn("#zoomFitBtn");
  ok(Math.abs(at1.left - at0.left) < 0.5 && Math.abs(at1.top - at0.top) < 0.5 && Math.abs(at1.width - at0.width) < 0.5, `Draw crop keeps the zoomed image in place: ${at0.left.toFixed(1)},${at0.top.toFixed(1)} → ${at1.left.toFixed(1)},${at1.top.toFixed(1)}`);
  await drawMode();
  await page.keyboard.press("Escape");
  ok(!(await page.evaluate(() => document.getElementById("stage").classList.contains("cropping"))), "Esc leaves the crop mode");
  // without a keyboard: Cancel is the way out; a tap on the waiting Read says why
  await drawMode();
  const bar = await page.evaluate(() => !document.getElementById("cropAct").hidden);
  const rb = await page.evaluate(() => { const r = document.getElementById("readBtn").getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; });
  await page.touchscreen.tap(rb.x, rb.y);
  const why = await page.evaluate(() => document.getElementById("status").textContent);
  await page.click("#cropCancelBtn");
  const left = await page.evaluate(() => !document.getElementById("stage").classList.contains("cropping") && document.getElementById("cropAct").hidden && !document.getElementById("cropBar").hidden);
  ok(bar && /^Read waits: /.test(why) && left, `Cancel shown, a tap on Read: "${why}", Cancel leaves the mode (the bar stays)`);
  await drawMode();
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
  // a finger beside the image: the rectangle kept on it
  const e2 = { x: Math.max(0, Math.round((f0.x - s2.left) / D2)), y: Math.max(0, Math.round((f0.y - s2.top) / D2)) };
  e2.w = Math.round((f1.x - s2.left) / D2) - e2.x; e2.h = Math.round((f1.y - s2.top) / D2) - e2.y;
  await page.click("#cropApplyBtn");
  await page.waitForFunction(() => window.__ops.some((o) => o.includes("crop")));
  const got2 = JSON.parse((await page.evaluate(() => window.__ops.find((o) => o.includes("crop"))))).find((o) => o.op === "crop");
  ok(got2.x === e2.x && got2.y === e2.y && got2.w === e2.w && got2.h === e2.h, `one-finger crop at ${pinchZoom}: got ${JSON.stringify(got2)} expected ${JSON.stringify(e2)}`);
  // while a crop is drawn a second finger neither zooms nor pans: the rectangle goes on (the original back first: a
  // cropped image is not cropped again)
  await page.waitForFunction(() => document.getElementById("zoomVal").textContent === "100 %");
  await page.click("#autoCropBtn"); await page.click("#resetBtn");
  await page.waitForFunction(() => document.getElementById("bakeInfo").textContent.startsWith("3000 ×"));
  await drawMode();
  const g0 = await page.evaluate(() => { const r = document.getElementById("img").getBoundingClientRect(); return { x: r.x + 10, y: r.y + 10 }; });
  const v0 = await page.evaluate(() => { const p = document.getElementById("imgPane"); return { zoom: document.getElementById("zoomVal").textContent, sl: p.scrollLeft, st: p.scrollTop }; });
  await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ ...g0, id: 7 }] });
  await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [{ x: g0.x + 60, y: g0.y + 50, id: 7 }] });
  await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ x: g0.x + 60, y: g0.y + 50, id: 7 }, { x: g0.x + 100, y: g0.y + 80, id: 8 }] });
  for (let k = 1; k <= 6; k++) await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [{ x: g0.x + 60 + 10 * k, y: g0.y + 50 + 8 * k, id: 7 }, { x: g0.x + 100 + 25 * k, y: g0.y + 80 + 20 * k, id: 8 }] });
  await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  await new Promise((r) => setTimeout(r, 200));
  const v1 = await page.evaluate(() => { const p = document.getElementById("imgPane"); return { zoom: document.getElementById("zoomVal").textContent, sl: p.scrollLeft, st: p.scrollTop, rect: !document.getElementById("rect").hidden }; });
  await page.click("#cropCancelBtn");
  ok(v1.zoom === v0.zoom && v1.sl === v0.sl && v1.st === v0.st && v1.rect, `crop + a second finger: zoom ${v0.zoom} → ${v1.zoom}, scroll ${v0.sl},${v0.st} → ${v1.sl},${v1.st}, rect kept ${v1.rect}`);
  await new Promise((r) => setTimeout(r, 600)); // the quiet time after the crop
  // after Draw crop → Cancel two fingers zoom again; in the crop mode two fingers landing together zoom too
  {
    const p = await page.evaluate(() => { const r = document.getElementById("img").getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; });
    const two = (d) => [{ x: p.x - d, y: p.y, id: 31 }, { x: p.x + d, y: p.y, id: 32 }];
    const pinchOut = async () => {
      await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: two(30) });
      for (let d = 35; d <= 90; d += 5) await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: two(d) });
      await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
      await new Promise((r) => setTimeout(r, 200));
      return page.evaluate(() => document.getElementById("zoomVal").textContent);
    };
    const afterCancel = await pinchOut();
    await zoomBtn("#zoomFitBtn");
    await drawMode();
    const inCrop = await pinchOut();
    const rectAfter = await page.evaluate(() => !document.getElementById("rect").hidden);
    await page.click("#cropCancelBtn");
    await zoomBtn("#zoomFitBtn");
    await new Promise((r) => setTimeout(r, 600));
    ok(afterCancel !== "100 %" && inCrop !== "100 %" && !rectAfter, `pinch after Draw crop → Cancel: ${afterCancel}; a pinch in the crop mode: ${inCrop}, no rectangle ${!rectAfter}`);
  }
  // double tap = fit
  for (let k = 0; k < 2; k++) await zoomBtn("#zoomInBtn");
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
  // the phone upright: Image / Text tabs — one card at a time, the image steps only with the image, zoom + scroll kept
  {
    // the viewport change reloaded the page: the setup closed, the image opened again
    await page.waitForFunction(() => document.getElementById("setupDlg").open, { timeout: 10000 });
    await page.click("#suClose");
    const [ch] = await Promise.all([page.waitForFileChooser(), page.click("#openBtn")]);
    await ch.accept([IMG]);
    await page.waitForFunction(() => !document.getElementById("stage").hidden);
    await new Promise((r) => setTimeout(r, 500));
    const st = () => page.evaluate(() => ({
      img: getComputedStyle(document.getElementById("imgCard")).display !== "none",
      text: getComputedStyle(document.getElementById("textCard")).display !== "none",
      steps: getComputedStyle(document.getElementById("imgSteps")).display !== "none",
      zoom: document.getElementById("zoomVal").textContent,
      left: document.getElementById("imgPane").scrollLeft,
    }));
    await zoomBtn("#zoomInBtn"); await zoomBtn("#zoomInBtn");
    await new Promise((r) => setTimeout(r, 200));
    await page.evaluate(() => { document.getElementById("imgPane").scrollLeft = 150; });
    const a = await st();
    const tabAt = () => page.evaluate(() => { const r = document.getElementById("paneTabs").getBoundingClientRect(); return Math.round(r.y); });
    const y0 = await tabAt();
    await page.click("#paneTabs [data-tab=text]");
    await new Promise((r) => setTimeout(r, 300));
    const b = await st();
    await page.screenshot({ path: `${OUT}/11b_phone_text_tab.png` });
    const y1 = await tabAt();
    await page.click("#paneTabs [data-tab=image]");
    await new Promise((r) => setTimeout(r, 300));
    const c = await st();
    ok(a.img && !a.text && a.steps && !b.img && b.text && !b.steps && c.img && c.zoom === a.zoom && a.left > 100 && Math.abs(c.left - a.left) < 2 && y0 === y1,
      `phone tabs (bar at y ${y0} / ${y1}): image ${JSON.stringify(a)} → text ${JSON.stringify(b)} → image ${JSON.stringify(c)}`);
    // Draw crop begun beside the image (the pane's margin above it): the rectangle starts at the image's corner
    await zoomBtn("#zoomFitBtn");
    await new Promise((r) => setTimeout(r, 300));
    const top0 = await page.evaluate(() => document.getElementById("img").getBoundingClientRect().top);
    await drawMode();
    const geo = await page.evaluate(() => ({ p: document.getElementById("imgPane").getBoundingClientRect().toJSON(), i: document.getElementById("img").getBoundingClientRect().toJSON(),
      bar: document.getElementById("cropBar").getBoundingClientRect().bottom }));
    const from = { x: geo.p.left + 4, y: geo.bar + 4 }, to = { x: geo.i.left + geo.i.width / 2, y: geo.i.top + geo.i.height / 2 };
    await page.mouse.move(from.x, from.y); await page.mouse.down(); await page.mouse.move(to.x, to.y, { steps: 5 }); await page.mouse.up();
    const [rc, ic] = await page.evaluate(() => ["rect", "img"].map((id) => document.getElementById(id).getBoundingClientRect().toJSON()));
    ok(geo.i.top > geo.bar + 8 && Math.abs(rc.left - ic.left) < 1.5 && Math.abs(rc.top - ic.top) < 1.5, `crop begun beside the image: rect at ${rc.left.toFixed(0)},${rc.top.toFixed(0)}, image at ${ic.left.toFixed(0)},${ic.top.toFixed(0)}`);
    // the phone's title bar holds Cancel + Crop: both on screen, no horizontal scroll
    const rs = await page.evaluate(() => { const r = document.getElementById("resetBtn").getBoundingClientRect(), d = document.getElementById("drawCropBtn").getBoundingClientRect(), b = document.getElementById("cropBar").getBoundingClientRect();
      return { row: Math.abs((r.top + r.bottom) / 2 - (d.top + d.bottom) / 2), end: Math.round(b.right - r.right), label: document.querySelector("#resetBtn span").getBoundingClientRect().width > 0,
        tight: document.getElementById("cropBar").classList.contains("tight") }; });
    // its label where it fits (the fonts decide: a Linux font is wider), else the icon alone — never a line of its own
    ok(rs.row < 2 && rs.end <= 12 && rs.label !== rs.tight, `phone: Reset on Draw's line, at the right end (${rs.end} px in), ${rs.label ? "with its label" : "its icon only"}`);
    if (rs.tight) {
      R.skip("phone, less room → Reset's icon only: this font shows the icon at full width already (the check above holds)");
    } else {
      // just too little room for the label (the free space beside Reset + 2 px, as a wider font gives): the icon alone,
      // still on Draw's line — measured, so it holds for any font
      const cut = await page.evaluate(() => {
        const free = document.getElementById("resetBtn").getBoundingClientRect().left - document.getElementById("pageSideSeg").getBoundingClientRect().right - 6;
        document.getElementById("cropBar").style.right = `${Math.ceil(free) + 2}px`;
        return Math.ceil(free) + 2;
      });
      await new Promise((r) => setTimeout(r, 200));
      const rs2 = await page.evaluate(() => { const r = document.getElementById("resetBtn").getBoundingClientRect(), d = document.getElementById("drawCropBtn").getBoundingClientRect();
        const out = { row: Math.abs((r.top + r.bottom) / 2 - (d.top + d.bottom) / 2), tight: document.getElementById("cropBar").classList.contains("tight") };
        document.getElementById("cropBar").style.right = ""; return out; });
      await new Promise((r) => setTimeout(r, 200));
      ok(rs2.row < 2 && rs2.tight, `phone, ${cut} px less room: Reset its icon only, still on Draw's line (${rs2.row.toFixed(1)} px apart)`);
    }
    const bar2 = await page.evaluate(() => ({ crop: document.getElementById("cropApplyBtn").getBoundingClientRect().right, vw: innerWidth, sw: document.documentElement.scrollWidth, shown: !document.getElementById("cropApplyBtn").hidden }));
    await page.screenshot({ path: `${OUT}/11d_phone_crop.png` });
    ok(bar2.shown && bar2.crop <= bar2.vw && bar2.sw <= bar2.vw && Math.abs(ic.top - top0) < 0.5, `phone crop: Crop at ${bar2.crop.toFixed(0)} of ${bar2.vw}, page ${bar2.sw} wide, the image stayed (top ${top0.toFixed(1)} → ${ic.top.toFixed(1)})`);
    await page.click("#cropCancelBtn");
    // the Text size gauge upright (a real page: the synthetic image has no measurable writing): its label in two
    // lines, closed by its ×
    const [ch2] = await Promise.all([page.waitForFileChooser(), page.click("#openBtn")]);
    await ch2.accept([path.join(REPO, "site", "examples", "humboldt_17r.jpg")]);
    await page.waitForFunction(() => /humboldt_17r/.test(document.getElementById("status").textContent) && document.getElementById("bakeInfo").textContent.startsWith("1176"));
    // the zoom stops where the image's layer would pass 6144 device px (a phone ended its page with a larger one)
    const fitPx = await page.evaluate(() => { const r = document.getElementById("img").getBoundingClientRect(); return Math.max(r.width, r.height) * devicePixelRatio; });
    for (let k = 0; k < 14; k++) await page.evaluate(() => document.getElementById("zoomInBtn").click());
    await new Promise((r) => setTimeout(r, 300));
    const layer = await page.evaluate(() => { const r = document.getElementById("img").getBoundingClientRect(); return { px: Math.round(Math.max(r.width, r.height) * devicePixelRatio), zoom: document.getElementById("zoomVal").textContent }; });
    const capPx = Math.min(6144, fitPx * 8);
    ok(Math.abs(layer.px - capPx) < 3, `phone zoom capped: ${layer.zoom}, the image ${layer.px} device px (fit ${Math.round(fitPx)}, cap ${Math.round(capPx)})`);
    await zoomBtn("#zoomFitBtn");
    await page.select("#scaleSel", "average");
    await page.waitForFunction(() => !document.getElementById("gauge").hidden, { timeout: 20000 });
    await new Promise((r) => setTimeout(r, 500));
    // the model size line follows the text size at once, not only when a read starts
    const infoAvg = await page.evaluate(() => document.getElementById("bakeInfo").textContent);
    // upright the image head is two lines: Text size + zoom, then the size line
    const head = await page.evaluate(() => {
      const r = (id) => document.getElementById(id).getBoundingClientRect();
      const sel = r("scaleSel"), zm = r("zoomSeg"), info = r("bakeInfo"), h = document.querySelector("#imgCard .cardhead").getBoundingClientRect();
      return { row1: Math.abs((sel.top + sel.bottom) / 2 - (zm.top + zm.bottom) / 2), below: info.top >= Math.max(sel.bottom, zm.bottom) - 1, h: Math.round(h.height) };
    });
    ok(head.row1 < 3 && head.below && head.h < 90, `phone image head in two lines: Text size beside the zoom (${head.row1.toFixed(1)} px apart), the size line below ${head.below}, ${head.h} px tall`);
    const lines = await page.evaluate(() => { const t = document.getElementById("gaugeText"); return Math.round(t.getBoundingClientRect().height / parseFloat(getComputedStyle(t).lineHeight)); });
    await page.screenshot({ path: `${OUT}/11c_phone_gauge.png` });
    await page.click("#gaugeClose");
    const closed = await page.evaluate(() => document.getElementById("gauge").hidden);
    await page.select("#scaleSel", "larger");
    await page.waitForFunction(() => !document.getElementById("gauge").hidden, { timeout: 20000 });
    await zoomBtn("#zoomInBtn");
    const byZoom = await page.evaluate(() => document.getElementById("gauge").hidden);
    await zoomBtn("#zoomFitBtn");
    const infoLarge = await page.evaluate(() => document.getElementById("bakeInfo").textContent);
    ok(/ · text /.test(infoAvg) && / · text /.test(infoLarge), `size line per text size before a read: average "${infoAvg}", larger "${infoLarge}"`);
    ok(lines >= 2 && lines <= 3 && closed && byZoom, `phone gauge: label in ${lines} lines (now · read · full size), closed by × → hidden ${closed}; shown again by a text size, hidden by a zoom ${byZoom}`);
    await page.select("#scaleSel", "");
    // the text: Fit takes every line whole, two fingers zoom it
    await page.click("#paneTabs [data-tab=text]");
    await page.click("#markupBtn");
    const long = "Agreement between British traders at Cape Coast Castle and Fante leaders about the exclusion of the French";
    await page.evaluate((t) => { document.getElementById("src").value = t; }, [long, "short line", long].join("\n"));
    await page.click("#markupBtn");
    await new Promise((r) => setTimeout(r, 400));
    const fit = await page.evaluate(() => { const e = document.getElementById("editor"); return { sw: e.scrollWidth, cw: e.clientWidth, fs: parseFloat(e.style.fontSize) }; });
    const er = await page.evaluate(() => document.getElementById("editor").getBoundingClientRect().toJSON());
    const cx = er.left + er.width / 2, cy = er.top + 120;
    const pcdp = await page.target().createCDPSession();
    await pcdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ x: cx - 30, y: cy, id: 21 }, { x: cx + 30, y: cy, id: 22 }] });
    for (let k = 1; k <= 6; k++) await pcdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [{ x: cx - 30 - 15 * k, y: cy, id: 21 }, { x: cx + 30 + 15 * k, y: cy, id: 22 }] });
    await pcdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    await new Promise((r) => setTimeout(r, 300));
    const zoomed = await page.evaluate(() => parseFloat(document.getElementById("editor").style.fontSize));
    ok(fit.sw <= fit.cw + 1 && zoomed > fit.fs * 2, `text: Fit ${fit.fs.toFixed(1)} px, content ${fit.sw} ≤ ${fit.cw}; pinched → ${zoomed.toFixed(1)} px`);
    await page.click("#fitBtn"); await page.click("#fitBtn"); // Fit off and on: the zoom from the start again
    await page.click("#paneTabs [data-tab=image]");
  }
  // a phone on its side: tab mode too — the tabs in the title bar, Read with the image beside the zoom, else in the bar
  {
    await page.setViewport({ width: 844, height: 390, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    await new Promise((r) => setTimeout(r, 600));
    if (await page.evaluate(() => document.getElementById("setupDlg").open)) await page.click("#suClose");
    const where = () => page.evaluate(() => ({
      stacked: document.body.classList.contains("stacked"),
      tabsInBar: document.getElementById("paneTabs").parentElement.classList.contains("titlebar"),
      readBeside: document.getElementById("readBtn").previousElementSibling === document.getElementById("zoomSeg"),
      oneRow: document.querySelector(".titlebar").getBoundingClientRect().height < 60,
    }));
    const a = await where();
    await page.screenshot({ path: `${OUT}/13_phone_landscape.png` });
    await page.click("#paneTabs [data-tab=text]");
    await new Promise((r) => setTimeout(r, 200));
    const b = await where();
    await page.screenshot({ path: `${OUT}/13b_phone_landscape_text.png` });
    ok(a.stacked && a.tabsInBar && a.readBeside && a.oneRow && !b.readBeside && b.oneRow, `phone on its side: image ${JSON.stringify(a)} · text ${JSON.stringify(b)}`);
    await page.click("#paneTabs [data-tab=image]");
    await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
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
