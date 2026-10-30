// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The published build as GitHub Pages serves it (web/serve.py --dist <site> --prefix /historic-htr-ocr/): the project
// pages and their links, the app under /app/ — its CSP, the setup to a stored model, About / Help / Settings, the
// service worker + manifest under the sub-path, and the production model URL — a real download from Hugging Face is
// started and cancelled only with OCR_LITE_HF_DOWNLOAD=1 (needs the network; every run would count as a download of the
// model on the Hub: not in CI, before a release by hand). The model for the setup comes from /model/ (?modelBase=/model/,
// localhost only): the real one with OCR_LITE_MODEL_DIR — then one image is read and must equal the native read
// (OCR_LITE_READ_IMAGE + OCR_LITE_READ_EXPECT, ocr-wgpu --example read --out) — else a stand-in, and the read is a SKIP.
//   node tests/browser/pages.cjs
const fs = require("fs");
const os = require("os");
const path = require("path");
const { launch, serve, outDir, results, fakeModel, sleep, MODEL_DIR, PREFIX } = require("./lib.cjs");
const R = results("pages");
const ok = R.ok;
const DIR = outDir("pages");
const IMG = process.env.OCR_LITE_READ_IMAGE ?? null;
const EXPECT = process.env.OCR_LITE_READ_EXPECT ?? null;
const FORBIDDEN = /\bkey\b/i; // the app has no key step (the model is open)

const cropTool = (page, id) => page.click(id);
(async () => {
  const real = MODEL_DIR !== null && IMG !== null && EXPECT !== null;
  const srv = await serve(8804, { modelDir: real ? MODEL_DIR : fakeModel() });
  const BASE = srv.base;
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "ocrlite-pages-"));
  fs.mkdirSync(path.join(DIR, "out"), { recursive: true });
  const browser = await launch({ userDataDir: profile });
  try {
    // ── the project pages: every page loads, every local link and image resolves ──
    {
      const p = await browser.newPage();
      const pages = ["", "markup.html", "training.html", "data.html", "metric.html", "outlook.html"];
      const bad = [];
      const seen = new Set();
      for (const name of pages) {
        const r = await p.goto(srv.root + name, { waitUntil: "load" });
        if (!r.ok()) {
          bad.push(`${name || "index"}: HTTP ${r.status()}`);
          continue;
        }
        const refs = await p.evaluate(() => [...document.querySelectorAll("a[href], img[src], link[href]")].map((e) => e.href || e.src));
        for (const u of refs) {
          const url = new URL(u);
          if (url.origin !== new URL(srv.root).origin || seen.has(url.pathname)) {
            continue;
          }
          seen.add(url.pathname);
          const h = await fetch(url.origin + url.pathname);
          if (!h.ok) {
            bad.push(`${name || "index"} → ${url.pathname}: HTTP ${h.status}`);
          }
        }
      }
      ok(bad.length === 0, `site pages: ${pages.length} pages, ${seen.size} local links / images resolve${bad.length ? ` — broken: ${bad.join(" · ")}` : ""}`);
      for (const ex of ["humboldt_48r", "humboldt_17r", "goethe_blb_6171314"]) {
        const r = await p.goto(`${srv.root}examples/${ex}.html`, { waitUntil: "load" });
        const t = await p.evaluate(() => document.body.innerText.length);
        ok(r.ok() && t > 200, `example ${ex}.html: HTTP ${r.status()}, ${t} chars of text`);
      }
      await p.close();
    }
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 860 });
    const errors = [];
    const infos = [];
    page.on("pageerror", (e) => { errors.push(e.message); console.log("pageerror", e.message); });
    page.on("console", (m) => { if (m.type() === "error") console.log("console.error", m.text()); if (m.type() === "info") infos.push(m.text()); });
    // the read's text as the worker streams it
    await page.evaluateOnNewDocument(() => {
      window.__text = "";
      const W = window.Worker;
      window.Worker = class extends W {
        set onmessage(f) { super.onmessage = (e) => { if (e.data && e.data.event === "text") window.__text += e.data.text; f(e); }; }
        get onmessage() { return super.onmessage; }
      };
    });
    const step = () => page.evaluate(() => {
      const d = document.getElementById("setupDlg");
      const s = [...d.querySelectorAll("[data-step]")].find((x) => !x.hidden);
      return d.open ? s.dataset.step : null;
    });
    const visibleText = () => page.evaluate(() => document.body.innerText);

    await page.goto(`${BASE}?modelBase=/model/&gpuCheck=skip`, { waitUntil: "networkidle0" });
    const meta = await page.evaluate(() => ({ csp: document.querySelector('meta[http-equiv="Content-Security-Policy"]')?.content ?? "", build: document.querySelector('meta[name="ocrlite-build"]')?.content ?? "" }));
    ok(/connect-src 'self' https:\/\/huggingface\.co/.test(meta.csp) && /"version"/.test(meta.build), `CSP + build stamp in index.html: ${meta.build}`);
    ok(infos.some((t) => /test model base \/model\//.test(t)), `?modelBase honoured on localhost: ${infos.find((t) => /model base/.test(t))}`);
    ok(!(await page.evaluate(() => document.getElementById("suSteps").textContent)).includes("Key"), `setup steps: ${await page.evaluate(() => [...document.querySelectorAll("#suSteps li")].map((l) => l.textContent).join(" · "))}`);
    ok(await page.evaluate(() => document.querySelector('[data-step="key"]') === null && document.getElementById("suKey") === null), "no key step in the page");

    // the setup: device check (skipped) → where → the model step downloads by itself
    await page.waitForFunction(() => document.getElementById("setupDlg").open, { timeout: 15000 });
    await page.waitForFunction(() => !document.getElementById("suGpuNext").hidden || !document.getElementById("suGpuAnyway").hidden, { timeout: 60000 });
    await page.click(await page.evaluate(() => (document.getElementById("suGpuNext").hidden ? "#suGpuAnyway" : "#suGpuNext")));
    await page.waitForFunction(() => !document.querySelector('[data-step="where"]').hidden);
    const where = await page.evaluate(() => document.querySelector('[data-step="where"]').innerText);
    ok(/Hugging Face/.test(where) && !FORBIDDEN.test(where), `where step: ${where.replace(/\s+/g, " ").slice(0, 140)}`);
    await page.screenshot({ path: path.join(DIR, "out", "1_where.png") });
    await page.click("#suInBrowser");
    await page.waitForFunction(() => !document.querySelector('[data-step="model"]').hidden, { timeout: 10000 });
    await page.waitForFunction(() => document.getElementById("suModelNote").textContent !== "", { timeout: 10000 });
    const note = await page.evaluate(() => document.getElementById("suModelNote").textContent);
    ok(/downloaded once from Hugging Face/.test(note) && !FORBIDDEN.test(note), `model step straight after where: ${note}`);
    const t0 = Date.now();
    for (let i = 0; ; i++) {
      const s = await step();
      if (s === "done") break;
      const e = await page.evaluate(() => document.getElementById("suModelErr").textContent);
      if (e) throw new Error(`download: ${e}`);
      if (i % 20 === 0) console.log(`  download: ${await page.evaluate(() => document.getElementById("suProgressText").textContent)}`);
      if (i > 1200) throw new Error("download stuck");
      await sleep(500);
    }
    const rec = await page.evaluate(async () => {
      const root = await navigator.storage.getDirectory();
      const names = [];
      for await (const [n] of (await root.getDirectoryHandle("models")).entries()) names.push(n);
      return { names: names.sort(), rec: JSON.parse(await (await (await root.getFileHandle("model-source.json")).getFile()).text()) };
    });
    ok(rec.names.join() === "mmproj-bf16.gguf,text-q8_0.gguf" && rec.rec.source === "hub", `downloaded in ${((Date.now() - t0) / 1000).toFixed(1)} s → ${JSON.stringify(rec)}`);
    await page.screenshot({ path: path.join(DIR, "out", "2_done.png") });
    await page.click("#suDone");
    await page.waitForFunction(() => document.getElementById("status").textContent === "Ready", { timeout: 10000 });
    ok(true, "status Ready after the setup");
    // Settings → Verify model: the stored files' MD5 against the release — "intact" for the real model, "damaged" for the
    // stand-in (its bytes are not the release's)
    await page.evaluate(() => document.getElementById("settingsBtn").click());
    await page.waitForFunction(() => document.getElementById("settingsDlg").open, { timeout: 10000 });
    const tv = Date.now();
    await page.evaluate(() => document.getElementById("modelVerify").click());
    await page.waitForFunction(() => /intact|damaged/.test(document.getElementById("modelProgressText").textContent) || document.getElementById("settingsErr").textContent !== "", { timeout: 180000, polling: 500 });
    const verdict = await page.evaluate(() => document.getElementById("modelProgressText").textContent || document.getElementById("settingsErr").textContent);
    ok(real ? /intact/.test(verdict) : /damaged/.test(verdict), `Verify model (${real ? "the real model" : "a stand-in"}, ${((Date.now() - tv) / 1000).toFixed(1)} s): ${verdict}`);
    await page.evaluate(() => document.getElementById("settingsDlg").close());
    // the status line ends with the check's result, not the worker's last "checking …" line
    const after = await page.evaluate(() => document.getElementById("status").textContent);
    ok(/intact|damaged/.test(after) && !/checking/i.test(after), `status line after the check: "${after}"`);

    // a read with the real model: must equal the native read
    if (!real) {
      R.skip("a real read (5 checks: it runs, its text equals the native read, fit for the read, a scaled read, Read ready after it) — set OCR_LITE_MODEL_DIR, OCR_LITE_READ_IMAGE, OCR_LITE_READ_EXPECT");
    } else {
    const [chooser] = await Promise.all([page.waitForFileChooser(), page.click("#openBtn")]);
    await chooser.accept([IMG]);
    await page.waitForFunction(() => !document.getElementById("stage").hidden);
    await page.click("#readBtn");
    await page.waitForFunction(() => document.getElementById("readBtn").classList.contains("stop"), { timeout: 10000 });
    await page.waitForFunction(() => !document.getElementById("readBtn").classList.contains("stop"), { timeout: 300000, polling: 200 });
    // the status line follows the button (ocr:done awaits the editor first)
    await page.waitForFunction(() => /tok\/s|stopped|token limit/.test(document.getElementById("status").textContent), { timeout: 10000, polling: 100 });
    const st = await page.evaluate(() => document.getElementById("status").textContent);
    const text = await page.evaluate(() => window.__text);
    const want = fs.readFileSync(EXPECT, "utf8");
    const norm = (t) => t.split("\n").map((l) => l.trimEnd()).join("\n").trim();
    ok(/tok\/s/.test(st), `read ran: ${st}`);
    ok(norm(text) === norm(want), `read == ${path.basename(EXPECT)} (line ends trimmed): ${text.length} chars, first line "${text.split("\n")[0]}"`);
    fs.writeFileSync(path.join(DIR, "out", "read.txt"), text);
    await page.screenshot({ path: path.join(DIR, "out", "3_read.png") });
    // the Scale: Auto finds the text crop, the slider to its low end (9 px letters) — the read on the scaled input
    await cropTool(page, "#textCropBtn");
    await page.waitForFunction(() => /^Text crop/.test(document.getElementById("status").textContent), { timeout: 30000 });
    await page.click("#cropApplyBtn");
    await page.waitForFunction(() => !document.getElementById("scaleSlider").disabled, { timeout: 30000 });
    await page.evaluate(() => { const s = document.getElementById("scaleSlider"); s.value = s.min; s.dispatchEvent(new Event("input")); });
    const sizeLine = await page.evaluate(() => `${document.getElementById("bakeInfo").textContent} · ${document.getElementById("scaleInfo").textContent}`);
    await page.evaluate(() => { window.__text = ""; });
    await page.click("#zoomInBtn"); await page.click("#zoomInBtn"); // a zoomed image goes back to fit for the read
    await page.click("#readBtn");
    await page.waitForFunction(() => document.getElementById("readBtn").classList.contains("stop"), { timeout: 10000 });
    await page.waitForFunction(() => !document.getElementById("readBtn").classList.contains("stop"), { timeout: 300000, polling: 200 });
    // the status line follows the button (ocr:done awaits the editor first)
    await page.waitForFunction(() => /tok\/s|stopped|token limit/.test(document.getElementById("status").textContent), { timeout: 10000, polling: 100 });
    const fitForRead = await page.evaluate(() => document.getElementById("zoomVal").textContent);
    ok(fitForRead === "100 %", `the zoomed image back to fit for the read: ${fitForRead}`);
    const scaled = await page.evaluate(() => ({ info: `${document.getElementById("bakeInfo").textContent} · ${document.getElementById("scaleInfo").textContent}`, st: document.getElementById("status").textContent, text: window.__text }));
    // the read's model input is the one the size line showed before it
    ok(scaled.info === sizeLine && /· 9 px$/.test(scaled.info) && /tok\/s/.test(scaled.st) && scaled.text.length > 0, `Scale read: ${scaled.info} · ${scaled.text.length} chars`);
    fs.writeFileSync(path.join(DIR, "out", "read_scaled.txt"), scaled.text);
    await page.screenshot({ path: path.join(DIR, "out", "3b_read_scaled.png") });
    // after it Read is ready for the next image — and after a reload
    const readState = () => page.evaluate(() => ({ disabled: document.getElementById("readBtn").disabled, why: document.getElementById("readBtn").title, status: document.getElementById("status").textContent }));
    const reopen = async () => {
      const [ch] = await Promise.all([page.waitForFileChooser(), page.click("#openBtn")]);
      await ch.accept([IMG]);
      await sleep(1500);
    };
    await reopen();
    const next = await readState();
    await page.reload({ waitUntil: "networkidle0" });
    await sleep(1500);
    await reopen();
    const reloaded = await readState();
    ok(!next.disabled && !reloaded.disabled, `Read after a scaled read: next image ${JSON.stringify(next)} · after a reload ${JSON.stringify(reloaded)}`);
    }

    // About / Help / Settings: the open model, no key or preview text
    await page.click("#aboutBtn");
    const about = await page.evaluate(() => ({ t: document.getElementById("aboutDlg").innerText, links: [...document.querySelectorAll("#aboutDlg a")].map((a) => a.href) }));
    ok(!FORBIDDEN.test(about.t) && about.links.includes("https://huggingface.co/b-straub/LightOnOCR-3-1B-Historic-German-Handwriting") && /model in browser storage \(LightOnOCR-3-1B · Historic German Handwriting\)/.test(about.t) && about.links.includes("https://github.com/b-straub/historic-htr-ocr"), "About: HF card + GitHub linked, no key text");
    ok(/CC BY-SA 4\.0/.test(about.t) && !/Planned release/.test(about.t), "About: CC BY-SA 4.0, not 'planned'");
    await page.screenshot({ path: path.join(DIR, "out", "4_about.png") });
    await page.keyboard.press("Escape");
    await page.click("#helpBtn");
    const help = await page.evaluate(() => document.getElementById("helpDlg").innerText);
    ok(!FORBIDDEN.test(help) && /Hugging Face/.test(help) && /offline/i.test(help), "Help: setup + offline texts of the open model");
    await page.screenshot({ path: path.join(DIR, "out", "5_help.png") });
    await page.keyboard.press("Escape");
    await page.click("#settingsBtn");
    await page.waitForFunction(() => document.getElementById("settingsDlg").open);
    await sleep(800);
    const set = await page.evaluate(() => ({ t: document.getElementById("settingsDlg").innerText, btn: document.getElementById("modelFromHub").textContent, src: document.getElementById("modelSourceWeb").textContent }));
    ok(!FORBIDDEN.test(set.t) && set.btn === "Download from Hugging Face" && set.src === "from Hugging Face", `Settings: "${set.btn}" · "${set.src}"`);
    await page.screenshot({ path: path.join(DIR, "out", "6_settings.png") });
    await page.click("#settingsCancel");
    ok(!FORBIDDEN.test(await visibleText()), "page text: no key text");

    // the PWA under the sub-path
    const sw = await page.evaluate(async () => { const r = await navigator.serviceWorker.getRegistration(); return r ? r.scope : null; });
    ok(sw === BASE, `service worker scope ${sw}`);
    const man = await page.evaluate(async () => { const l = document.querySelector('link[rel="manifest"]'); const r = await fetch(l.href); return { ok: r.ok, m: await r.json(), url: l.href }; });
    ok(man.ok && man.url === `${BASE}manifest.webmanifest` && new URL(man.m.start_url, man.url).href === BASE, `manifest under the sub-path: start_url → ${new URL(man.m.start_url, man.url).href}`);

    // no ?modelBase: the real Hugging Face URL — the download runs (needs the network; only with OCR_LITE_HF_DOWNLOAD=1:
    // each run counts as a download on the Hub)
    if (process.env.OCR_LITE_HF_DOWNLOAD !== "1") {
      R.skip("production base (no ?modelBase): the HF download runs — set OCR_LITE_HF_DOWNLOAD=1 (it counts as a download on the Hub)");
    } else {
    const ctx = await browser.createBrowserContext();
    const p2 = await ctx.newPage();
    const fetched = [];
    p2.on("console", (m) => { if (m.type() === "error") fetched.push(m.text()); });
    await p2.goto(`${BASE}?gpuCheck=skip`, { waitUntil: "networkidle0" });
    await p2.waitForFunction(() => document.getElementById("setupDlg").open, { timeout: 15000 });
    await p2.waitForFunction(() => !document.getElementById("suGpuNext").hidden || !document.getElementById("suGpuAnyway").hidden, { timeout: 60000 });
    await p2.click(await p2.evaluate(() => (document.getElementById("suGpuNext").hidden ? "#suGpuAnyway" : "#suGpuNext")));
    await p2.waitForFunction(() => !document.querySelector('[data-step="where"]').hidden);
    await p2.click("#suInBrowser");
    // the real Hugging Face model: the download runs — 2 % is proof, then Cancel
    await p2.waitForFunction(() => document.getElementById("suProgress").value > 0.02 || document.getElementById("suModelErr").textContent !== "", { timeout: 120000 });
    const dl = await p2.evaluate(() => ({ v: document.getElementById("suProgress").value, err: document.getElementById("suModelErr").textContent, line: document.getElementById("suProgressText").textContent }));
    ok(dl.err === "" && dl.v > 0.02, `production base (no ?modelBase): HF download runs — ${dl.line}${dl.err}`);
    await p2.click("#suCancel").catch(() => {});
    await p2.close();
    await ctx.close();
    }
    ok(errors.length === 0, `no page errors (${errors.length})`);
  } finally {
    await browser.close();
    srv.kill();
    fs.rmSync(profile, { recursive: true, force: true });
  }
  R.done();
})().catch((e) => { console.log("ERROR", e.message); process.exitCode = 1; });
