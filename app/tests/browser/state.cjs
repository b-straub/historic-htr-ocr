// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The control states (app.js applyState, the state table) + the crop tip, on the built site. The model comes from
// web/serve.py's /model/ (?modelBase=/model/): the real one with OCR_LITE_MODEL_DIR (its profile is kept in
// test-output/state-profile, so it is downloaded once), else a stand-in (lib.cjs fakeModel) — then the Read → Stop
// states, which need a read that really runs, are reported as SKIP.
//   node tests/browser/state.cjs
const fs = require("fs");
const path = require("path");
const { launch, serve, results, testImages, fakeModel, sleep, MODEL_DIR, REPO } = require("./lib.cjs");
const R = results("state");
const ok = R.ok;
const OUT = require("./lib.cjs").outDir("state");

// every control of the state table (app.js) by a short name
const C = {
  open: "#openBtn", rotL: '[data-op="rotateLeft"]', rotR: '[data-op="rotateRight"]', rot180: '[data-op="rotate180"]',
  contrast: '[data-op="autoContrast"]', pencil: '[data-op="pencil"]', crop: "#autoCropBtn", draw: "#drawCropBtn",
  find: "#autoCropGo", imgUndo: "#imgUndoBtn", reset: "#resetBtn", preset: "#presetSel", help: "#helpBtn",
  settings: "#settingsBtn", about: "#aboutBtn", read: "#readBtn", zoomOut: "#zoomOutBtn", zoomIn: "#zoomInBtn",
  fit: "#zoomFitBtn", fmt: "#fmtSeg button", gap: "#gapBtn", tab: "#tabBtn", undo: "#undoBtn", redo: "#redoBtn",
  markup: "#markupBtn", copy: "#copyBtn", save: "#saveBtn", saveTxt: '[data-save="txt"]', saveHtml: '[data-save="html"]',
  savePng: '[data-save="png"]',
};
const ALL = Object.keys(C);
// the expected enabled sets (the state table)
const ALWAYS = ["help", "about"];
const EDIT = ["fmt", "gap", "tab", "markup"];
const STEPS = ["rotL", "rotR", "rot180", "contrast", "pencil"];
const CROPMENU = ["crop", "draw", "find"];
const E = {
  noImg: [...ALWAYS, "open", "preset", "settings", ...EDIT],
  noModelImg: [...ALWAYS, "open", "preset", "settings", ...EDIT, ...STEPS, ...CROPMENU, "zoomIn", "save", "savePng"],
  img: [...ALWAYS, "open", "preset", "settings", ...EDIT, ...STEPS, ...CROPMENU, "zoomIn", "save", "savePng", "read"],
  imgZoomed: [...ALWAYS, "open", "preset", "settings", ...EDIT, ...STEPS, ...CROPMENU, "zoomIn", "zoomOut", "fit", "save", "savePng", "read"],
  crop: [...ALWAYS, "open", "preset", "settings", ...EDIT, ...CROPMENU, "zoomIn", "save", "savePng"],
  cropped: [...ALWAYS, "open", "preset", "settings", ...EDIT, ...STEPS, ...CROPMENU, "imgUndo", "reset", "zoomIn", "save", "savePng", "read"],
  // a read: About off (its re-layout during a read ended in crashes on the iPhone), Help stays
  starting: ["help", "zoomIn"],
  reading: ["help", "zoomIn", "read"],
  stopping: ["help", "zoomIn"],
  text: [...ALWAYS, "open", "preset", "settings", ...EDIT, ...STEPS, ...CROPMENU, "zoomIn", "save", "savePng", "read", "undo", "copy", "saveTxt", "saveHtml"],
  markupText: [...ALWAYS, "open", "preset", "settings", "markup", ...STEPS, ...CROPMENU, "zoomIn", "save", "savePng", "read", "copy", "saveTxt", "saveHtml"],
  markupEmpty: [...ALWAYS, "open", "preset", "settings", "markup", ...STEPS, ...CROPMENU, "zoomIn", "save", "savePng", "read"],
};

let srv = null;
let browser = null;
let BASE = null;
let BIG = null;
let SMALL = null;

(async () => {
  const real = MODEL_DIR !== null;
  srv = await serve(8802, { modelDir: real ? MODEL_DIR : fakeModel() });
  BASE = `${srv.base}?modelBase=/model/&gpuCheck=skip`;
  // the real model's profile is kept (a 1.4 GB download once); the stand-in's is fresh every run
  const profile = real ? path.join(REPO, "test-output", "state-profile") : fs.mkdtempSync(path.join(require("os").tmpdir(), "ocrlite-state-"));
  browser = await launch({ userDataDir: profile });
  ({ big: BIG, small: SMALL } = await testImages(browser));
  const errors = [];
  const prep = async (page) => {
    await page.setViewport({ width: 1440, height: 920, deviceScaleFactor: 1 });
    page.on("pageerror", (e) => { errors.push(e.message); console.log("pageerror", e.message); });
    // the snapshot of the enabled controls, in the page; the Read button's every state change records one
    await page.evaluateOnNewDocument((C) => {
      window.__snap = () => {
        const on = [];
        for (const [k, sel] of Object.entries(C)) {
          const el = document.querySelector(sel);
          if (el && !el.disabled) on.push(k);
        }
        const r = document.getElementById("readBtn");
        return { on, read: { state: r.dataset.state, label: r.textContent.trim(), title: r.title } };
      };
      window.__rec = [];
      document.addEventListener("DOMContentLoaded", () => {
        new MutationObserver(() => window.__rec.push(window.__snap()))
          .observe(document.getElementById("readBtn"), { attributes: true, attributeFilter: ["data-state"] });
      });
    }, C);
  };
  const snap = (page) => page.evaluate(() => window.__snap());
  const same = (got, want) => {
    const g = new Set(got), w = new Set(want);
    const extra = [...g].filter((k) => !w.has(k)), missing = [...w].filter((k) => !g.has(k));
    return { ok: extra.length === 0 && missing.length === 0, extra, missing };
  };
  const check = async (page, name, want, readTitle = null) => {
    const s = await snap(page);
    const d = same(s.on, want);
    ok(d.ok, `state ${name}: ${d.ok ? `${s.on.length} enabled` : `extra ${JSON.stringify(d.extra)} missing ${JSON.stringify(d.missing)}`}`);
    if (readTitle !== null) {
      ok(readTitle.test(s.read.title), `state ${name}: Read title "${s.read.title}"`);
    }
    return s;
  };
  const load = async (page) => {
    await page.goto(BASE, { waitUntil: "networkidle0" });
    await page.waitForFunction(() => document.getElementById("readBtn").title !== "Checking the model…" && document.getElementById("readBtn").title !== "", { timeout: 15000 });
    await sleep(800);
    if (await page.evaluate(() => document.getElementById("setupDlg").open)) {
      await page.click("#suClose");
    }
  };
  const open = async (page, img) => {
    const [chooser] = await Promise.all([page.waitForFileChooser(), page.click("#openBtn")]);
    await chooser.accept([img]);
    const w = path.basename(img) === path.basename(BIG) ? "3000" : "700";
    await page.waitForFunction((w) => !document.getElementById("stage").hidden && document.getElementById("bakeInfo").textContent.startsWith(`${w} ×`), {}, w);
    await sleep(300);
  };
  const hintShown = (page) => page.evaluate(() => !document.getElementById("cropHint").hidden);

  // ── no model (a fresh context: its own empty storage) ──
  {
    const ctx = await browser.createBrowserContext();
    const page = await ctx.newPage();
    await prep(page);
    await load(page);
    await check(page, "no model · no image", E.noImg, /^Set up the model first/);
    await open(page, BIG);
    await check(page, "no model · image", E.noModelImg, /^Set up the model first/);
    await ctx.close();
  }

  const page = await browser.newPage();
  // message boxes (the edit warning of a session) are answered and kept for the checks
  const dialogs = [];
  page.on("dialog", async (d) => { dialogs.push(d.message()); await d.accept(); });
  await prep(page);
  await load(page);
  // an earlier run killed mid-read (its Chrome gone before localStorage was flushed) leaves the crash marker / device
  // flag in this kept profile: cleared here, this test is not about them
  const stale = await page.evaluate(() => {
    const had = localStorage.getItem("ocrlite.unsupported") !== null;
    localStorage.removeItem("ocrlite.unsupported");
    localStorage.removeItem("ocrlite.crashMarker");
    localStorage.removeItem("ocrlite.cropHint.off");
    return had;
  });
  if (stale) {
    console.log("note: a stale device flag from a killed run cleared");
    await load(page);
  }
  // the model in this profile's storage (downloaded once from serve.py's model dir)
  if (/^Set up the model first/.test((await snap(page)).read.title)) {
    console.log("downloading the model into the profile (once)…");
    await page.click("#settingsBtn");
    await page.waitForFunction(() => document.getElementById("settingsDlg").open);
    await sleep(500);
    await page.click("#modelFromHub");
    for (let i = 0; ; i++) {
      const p = await page.evaluate(() => ({ t: document.getElementById("modelProgressText").textContent, e: document.getElementById("settingsErr").textContent }));
      if (p.t === "done" || p.e) {
        if (p.e) throw new Error(`download: ${p.e}`);
        break;
      }
      if (i > 1800) throw new Error(`download stuck: ${p.t}`);
      await sleep(500);
    }
    await page.click("#settingsCancel");
    await page.waitForFunction(() => document.getElementById("readBtn").title !== "" && !/^Set up the model/.test(document.getElementById("readBtn").title), { timeout: 15000 });
  }
  await check(page, "no image", E.noImg, /^Open an image first$/);

  // ── a fresh large image: the tip; Use whole page ──
  await open(page, BIG);
  ok(await hintShown(page), "crop tip shown on a fresh large image (3000 × 2000 → scaled)");
  await check(page, "image", E.img, /^Read the image/);
  await page.screenshot({ path: `${OUT}/hint_desktop_light.png` });
  // dialogs: modal — nothing changes behind them
  await page.click("#settingsBtn");
  await page.waitForFunction(() => document.getElementById("settingsDlg").open);
  await check(page, "image · Settings open", E.img);
  await page.click("#settingsCancel");
  await page.click("#zoomInBtn");
  await check(page, "image · zoomed", E.imgZoomed);
  await page.click("#zoomFitBtn");
  await page.click("#cropHintWhole");
  ok(!(await hintShown(page)), "Use whole page → tip gone");
  await page.click('[data-op="rotateLeft"]');
  await page.waitForFunction(() => document.getElementById("bakeInfo").textContent.startsWith("2000 ×"));
  ok(!(await hintShown(page)), "the tip stays gone for this image (a rotation does not bring it back)");
  await check(page, "image · one step", E.cropped);

  // ── reopen: the tip again; Draw crop → crop mode → pending → Cancel ──
  await open(page, BIG);
  ok(await hintShown(page), "reopened large image → the tip again (its own decision)");
  await check(page, "image (reopened, steps cleared)", E.img);
  await page.click("#cropHintDraw");
  ok(!(await hintShown(page)) && await page.evaluate(() => document.getElementById("stage").classList.contains("cropping")), "tip Draw crop → crop mode, tip gone");
  await check(page, "crop mode", E.crop, /^Apply or cancel the crop first/);
  {
    const r = await page.evaluate(() => document.getElementById("img").getBoundingClientRect().toJSON());
    await page.mouse.move(r.left + r.width * 0.2, r.top + r.height * 0.2);
    await page.mouse.down();
    await page.mouse.move(r.left + r.width * 0.7, r.top + r.height * 0.8, { steps: 5 });
    await page.mouse.up();
  }
  ok(await page.evaluate(() => !document.getElementById("cropPill").hidden), "a drawn rect is pending");
  await check(page, "pending crop", E.crop, /^Apply or cancel the crop first/);
  await page.click("#cropCancel");
  await check(page, "crop cancelled", E.img);

  // ── reopen: Find crop from the tip → a proposal (or nothing to trim) → Crop ──
  await open(page, BIG);
  ok(await hintShown(page), "reopened → tip");
  await page.click("#cropHintFind");
  await page.waitForFunction(() => /^Auto crop/.test(document.getElementById("status").textContent), { timeout: 30000 });
  const found = await page.evaluate(() => ({ st: document.getElementById("status").textContent, pending: !document.getElementById("cropPill").hidden, auto: document.getElementById("rect").classList.contains("auto") }));
  ok(!(await hintShown(page)), `tip Find crop → tip gone (${found.st})`);
  if (found.pending) {
    ok(found.auto, "Find crop → a dashed proposal");
    await check(page, "found crop pending", E.crop, /^Apply or cancel the crop first/);
    await page.click("#cropApply");
    await page.waitForFunction(() => document.getElementById("opsChips").textContent.includes("crop"));
    await check(page, "cropped", E.cropped);
  } else {
    ok(/nothing to trim/.test(found.st), `Find crop: ${found.st}`);
  }

  // ── reopen: a crop by hand from the menu → tip gone ──
  await open(page, BIG);
  ok(await hintShown(page), "reopened → tip");
  await page.click("#autoCropBtn");
  await page.click("#drawCropBtn");
  ok(!(await hintShown(page)), "menu Draw crop → tip gone");
  {
    const r = await page.evaluate(() => document.getElementById("img").getBoundingClientRect().toJSON());
    await page.mouse.move(r.left + r.width * 0.1, r.top + r.height * 0.1);
    await page.mouse.down();
    await page.mouse.move(r.left + r.width * 0.6, r.top + r.height * 0.7, { steps: 5 });
    await page.mouse.up();
  }
  await page.click("#cropApply");
  await page.waitForFunction(() => document.getElementById("opsChips").textContent.includes("crop"));
  ok(!(await hintShown(page)), "after a crop: no tip");
  await check(page, "cropped by hand", E.cropped);

  // ── reopen: Read → the tip goes; starting → reading → Stop → stopping → idle ──
  await open(page, BIG);
  ok(await hintShown(page), "reopened → tip");
  if (!real) {
    R.skip("Read → Stop: the tip on Read, the starting / reading / stopping states and the Read labels (6 checks) need a read that runs — set OCR_LITE_MODEL_DIR");
  } else {
  await page.evaluate(() => { window.__rec = []; });
  await page.click("#readBtn");
  ok(!(await hintShown(page)), "Read → tip gone");
  await page.waitForFunction(() => document.getElementById("readBtn").dataset.state === "reading", { timeout: 60000, polling: 50 }).catch(async (e) => {
    console.log("no reading state:", await page.evaluate(() => JSON.stringify({ st: document.getElementById("status").textContent, rec: window.__rec.map((r) => r.read) })));
    throw e;
  });
  await page.click("#readBtn");
  await page.waitForFunction(() => document.getElementById("readBtn").dataset.state === "idle", { timeout: 180000, polling: 100 });
  const rec = await page.evaluate(() => window.__rec);
  const seq = rec.map((r) => r.read.state);
  ok(JSON.stringify(seq) === JSON.stringify(["starting", "reading", "stopping", "idle"]), `Read → Stop: ${seq.join(" → ")} · status "${await page.evaluate(() => document.getElementById("status").textContent)}"`);
  for (const r of rec) {
    const want = r.read.state === "idle" ? null : E[r.read.state];
    if (want) {
      const d = same(r.on, want);
      ok(d.ok, `state ${r.read.state}: ${d.ok ? `${r.on.length} enabled · Read "${r.read.label}" (${r.read.title})` : `extra ${JSON.stringify(d.extra)} missing ${JSON.stringify(d.missing)}`}`);
    }
  }
  const lab = Object.fromEntries(rec.map((r) => [r.read.state, r.read.label]));
  ok(lab.reading === "Stop" && lab.stopping === "Stopping…" && lab.starting === "Read" && lab.idle === "Read", `Read labels ${JSON.stringify(lab)}`);

  // the app goes to the background while reading → the read stops as with Stop, the status says why. A real page
  // (OCR_LITE_READ_IMAGE): the stand-in image ends at once (end of text), before a stop could count
  if (!process.env.OCR_LITE_READ_IMAGE) {
    R.skip("background → read stopped: needs a real page that yields text — set OCR_LITE_READ_IMAGE");
  } else {
  const before = await page.evaluate(() => document.getElementById("bakeInfo").textContent);
  const [chooser] = await Promise.all([page.waitForFileChooser(), page.click("#openBtn")]);
  await chooser.accept([process.env.OCR_LITE_READ_IMAGE]);
  await page.waitForFunction((b) => document.getElementById("bakeInfo").textContent !== b, { timeout: 30000 }, before);
  await sleep(300);
  await page.click("#readBtn");
  await page.waitForFunction(() => document.getElementById("editor").textContent.length > 20, { timeout: 120000, polling: 100 });
  await page.evaluate(() => {
    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await page.waitForFunction(() => document.getElementById("readBtn").dataset.state === "idle", { timeout: 180000, polling: 100 });
  await page.evaluate(() => {
    delete document.visibilityState;
    document.dispatchEvent(new Event("visibilitychange"));
  });
  const hiddenStatus = await page.evaluate(() => document.getElementById("status").textContent);
  ok(/stopped: the app went to the background/.test(hiddenStatus), `background → read stopped: "${hiddenStatus}"`);
  }
  }

  // ── text: typed into the editor, then the markup view ──
  await page.evaluate(async () => {
    // the stopped read may have left text: start from an empty page
    document.getElementById("markupBtn").click();
    await new Promise((r) => setTimeout(r, 100));
    const s = document.getElementById("src");
    s.value = "";
    s.dispatchEvent(new Event("input"));
    document.getElementById("markupBtn").click();
    await new Promise((r) => setTimeout(r, 200));
  });
  await check(page, "image · no text", E.img);
  await page.click("#editor");
  await page.keyboard.type("Hallo Welt");
  await sleep(200);
  const warned = dialogs.filter((m) => /Tech preview/.test(m)).length;
  ok(warned === 1, `the first edit of the session warns once that edits are not kept (${warned} message box${warned === 1 ? "" : "es"})`);
  await check(page, "image · text", E.text);
  await page.click("#markupBtn");
  await sleep(200);
  await check(page, "markup view · text", E.markupText);
  await page.evaluate(() => { const s = document.getElementById("src"); s.select(); });
  await page.keyboard.press("Backspace");
  await sleep(100);
  await check(page, "markup view · no text", E.markupEmpty);
  await page.click("#markupBtn");
  await sleep(200);

  // ── a small image: no tip; Don't show again persists ──
  await open(page, SMALL);
  ok(!(await hintShown(page)), "small image (700 × 168, read at its size) → no tip");
  await open(page, BIG);
  ok(await hintShown(page), "large again → tip");
  await page.click("#cropHintNever");
  ok(!(await hintShown(page)), "Don't show again → tip gone");
  await open(page, BIG);
  ok(!(await hintShown(page)), "Don't show again: no tip on the next large image");
  await load(page);
  await open(page, BIG);
  ok(!(await hintShown(page)), "Don't show again: still none after a reload");
  await page.evaluate(() => { try { localStorage.removeItem("ocrlite.cropHint.off"); } catch {} });
  // screenshots: phone width (isMobile reloads the page), light + dark
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await load(page);
  await open(page, BIG);
  ok(await hintShown(page), "phone: tip on a fresh large image");
  const fit = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, vw: innerWidth, bar: document.getElementById("cropHint").getBoundingClientRect().toJSON() }));
  ok(fit.sw <= fit.vw && fit.bar.right <= fit.vw, `phone: the tip fits, no horizontal scroll (${fit.sw} / ${fit.vw})`);
  await page.screenshot({ path: `${OUT}/hint_phone_light.png` });
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
  await sleep(300);
  await page.screenshot({ path: `${OUT}/hint_phone_dark.png` });
  ok(errors.length === 0, `no page errors (${errors.length})`);
  await browser.close();
  srv.kill();
  R.done();
})().catch(async (e) => {
  console.log("ERROR", e);
  // closed, not killed: Chrome flushes the profile's localStorage (a killed one can leave the crash marker behind)
  if (browser) await browser.close();
  if (srv) srv.kill();
  process.exit(2);
});
