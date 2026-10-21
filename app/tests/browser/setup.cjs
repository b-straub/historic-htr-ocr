// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The first-run setup + help (ui/setup.js, bridge-web.js setup) on the built site, with a stand-in model (lib.cjs
// fakeModel) from web/serve.py's /model/ (?modelBase=/model/) and the device check skipped (?gpuCheck=skip — both
// honoured on localhost only): install hints, the download with progress + cancel, the done step, help, a removed
// model, an installed app (standalone), too little storage, the phone layout.
//   node tests/browser/setup.cjs
const fs = require("fs");
const os = require("os");
const path = require("path");
const { launch, serve, outDir, results, testImages, fakeModel, sleep } = require("./lib.cjs");
const R = results("setup");
const ok = R.ok;
const OUT = outDir("setup");
let srv = null;

(async () => {
  srv = await serve(8803, { modelDir: fakeModel() });
  const BASE = `${srv.base}?modelBase=/model/&gpuCheck=skip`;
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "ocrlite-setup-"));
  const browser = await launch({ userDataDir: profile });
  const { small: IMG } = await testImages(browser);
  const prep = async (page, { standalone = false, quota = null } = {}) => {
    await page.setViewport({ width: 1280, height: 860 });
    page.on("pageerror", (e) => console.log("pageerror", e.message));
    page.on("console", (m) => { if (m.type() === "error") console.log("console.error", m.text()); });
    await page.evaluateOnNewDocument((standalone, quota) => {
      if (standalone) Object.defineProperty(navigator, "standalone", { value: true });
      if (quota !== null) navigator.storage.estimate = async () => ({ quota, usage: 0 });
    }, standalone, quota);
  };
  const step = (page) => page.evaluate(() => {
    const d = document.getElementById("setupDlg");
    const s = [...d.querySelectorAll("[data-step]")].find((x) => !x.hidden);
    return d.open ? s.dataset.step : null;
  });
  const waitStep0 = (page, name, timeout = 30000) => page.waitForFunction((n) => {
    const d = document.getElementById("setupDlg");
    const s = [...d.querySelectorAll("[data-step]")].find((x) => !x.hidden);
    return d.open && s.dataset.step === n;
  }, { timeout }, name);
  // the device check comes first (ui/setup.js gpu step): passed (green → Continue) on the way to `name`
  const waitStep = async (page, name, timeout = 30000) => {
    const at = await page.waitForFunction((n) => {
      const d = document.getElementById("setupDlg");
      const s = [...d.querySelectorAll("[data-step]")].find((x) => !x.hidden);
      if (!d.open) return false;
      if (s.dataset.step === n) return n;
      return s.dataset.step === "gpu" && !document.getElementById("suGpuNext").hidden ? "gpu" : false;
    }, { timeout }, name);
    if ((await at.jsonValue()) === "gpu") {
      await page.click("#suGpuNext");
      await waitStep0(page, name, timeout);
    }
  };
  const text = (page, id) => page.evaluate((id) => document.getElementById(id).textContent, id);
  const status = (page) => text(page, "status");
  const shot = (page, n) => page.screenshot({ path: path.join(OUT, `${n}.png`) });

  // ── platform table (ui/setup.js platformOf) ──
  const page = await browser.newPage();
  await prep(page);
  await page.goto(BASE, { waitUntil: "networkidle0" });
  const table = [
    ["Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15", 5, "ios"],
    ["Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1", 5, "ios"],
    ["Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/130.0 Mobile/15E148 Safari/604.1", 5, "ios"],
    ["Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15", 0, "mac-safari"],
    ["Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36", 0, "chrome"],
    ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36", 0, "chrome"],
    ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0", 10, "edge"],
    ["Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36", 5, "chrome"],
    ["Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:140.0) Gecko/20100101 Firefox/140.0", 0, "firefox"],
    ["Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0", 0, "firefox"],
    ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 OPR/120.0", 0, "other"],
    ["Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Epiphany/46 Safari/605.1.15", 0, "other"],
  ];
  // the built setup module (content-hashed name): the one the page loaded
  const got = await page.evaluate(async (t) => {
    const url = performance.getEntriesByType("resource").map((e) => e.name).find((n) => /\/setup\.[0-9a-f]{12}\.js$/.test(n));
    const m = await import(url);
    return t.map(([ua, n]) => m.platformOf(ua, n));
  }, table);
  table.forEach(([ua, n, want], i) => ok(got[i] === want, `platformOf(${ua.slice(13, 60)}…, ${n}) = ${got[i]} (want ${want})`));

  // ── fresh profile → the setup opens; after the device check the "where" step ──
  await waitStep(page, "where");
  ok(await page.evaluate(() => [...document.querySelectorAll("#suSteps li")].map((l) => l.dataset.for).join() === "gpu,where,model,done"), "fresh profile: setup open at 'where', steps Device · Install · Model · Read");
  ok(/No model in this browser yet/.test(await status(page)), `status under the setup: ${await status(page)}`);
  await shot(page, "1_where");
  await page.click("#suToInstall");
  await waitStep(page, "install");
  const hint = await page.evaluate(() => ({ main: document.querySelector("#suHint b").textContent, others: [...document.querySelectorAll("#suOtherHints b")].map((b) => b.textContent), btn: !document.getElementById("suInstallBtn").hidden }));
  // the Install button shows once Chrome offers its prompt (beforeinstallprompt: manifest + service worker — its timing
  // is Chrome's), so it is reported, not required
  ok(hint.main === "Chrome" && hint.others.length === 5, `install step: ${JSON.stringify(hint)}`);
  await page.click(".suother summary");
  await shot(page, "2_install");
  await page.click("#suInstallBack");
  await waitStep(page, "where");
  await page.click("#suInBrowser");
  // the model step → the download starts on its own; throttled so the progress can be seen; Cancel
  const cdp = await page.createCDPSession();
  await cdp.send("Network.enable");
  await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 0, downloadThroughput: 8e6, uploadThroughput: 8e6 });
  await waitStep(page, "model");
  await page.waitForFunction(() => /downloading text-q8_0\.gguf · \d+ \/ 63 MB \(\d+ %\)/.test(document.getElementById("suProgressText").textContent) && document.getElementById("suProgress").value > 0.1, { timeout: 30000 });
  ok(true, `download started on its own: ${await text(page, "suProgressText")}`);
  await shot(page, "4_download");
  ok(await page.evaluate(() => getComputedStyle(document.getElementById("suFilesBtn")).display === "none"), "model step: no \"I have the files\"");
  await page.click("#suCancel");
  await page.waitForFunction(() => document.getElementById("suModelErr").textContent !== "");
  const opfs = () => page.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    const names = [];
    try { for await (const [n] of (await root.getDirectoryHandle("models")).entries()) names.push(n); } catch {}
    return names.sort();
  });
  ok((await text(page, "suModelErr")).startsWith("Download cancelled") && (await opfs()).length === 0, `cancel: "${await text(page, "suModelErr")}", OPFS ${JSON.stringify(await opfs())}`);
  await shot(page, "4b_cancelled");
  await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
  await page.click("#suDownload");
  await waitStep(page, "done");
  ok(JSON.stringify(await opfs()) === JSON.stringify(["mmproj-bf16.gguf", "text-q8_0.gguf"]), `download complete → done step, OPFS ${JSON.stringify(await opfs())}`);
  await shot(page, "5_done");
  await page.click("#suDone");
  await page.waitForFunction(() => document.getElementById("status").textContent === "Ready");
  ok(true, "setup closed, status Ready");
  // Read allowed (the fake model then fails to parse — the gate passed and the read started)
  await page.evaluate(() => { window.__st = []; new MutationObserver(() => window.__st.push(document.getElementById("status").textContent)).observe(document.getElementById("status"), { childList: true, characterData: true, subtree: true }); });
  const [chooser] = await Promise.all([page.waitForFileChooser(), page.click("#openBtn")]);
  await chooser.accept([IMG]);
  await page.waitForFunction(() => !document.getElementById("stage").hidden);
  await page.click("#readBtn");
  await page.waitForFunction(() => window.__st.some((s) => s.startsWith("Starting")) && !document.getElementById("readBtn").classList.contains("stop"), { timeout: 60000 });
  ok(!(await page.evaluate(() => window.__st.some((s) => /Read needs the model/.test(s)))), `Read allowed after the setup: ${JSON.stringify(await page.evaluate(() => window.__st.slice(-2)))}`);

  // ── About + help ──
  await page.click("#aboutBtn");
  ok(await page.evaluate(() => document.getElementById("aboutDlg").open && /Hugging Face|LightOnOCR/.test(document.getElementById("aboutDlg").innerText)), "About opens");
  await page.keyboard.press("Escape");
  await page.click("#helpBtn");
  await page.waitForFunction(() => document.getElementById("helpDlg").open);
  const help = await page.evaluate(() => { const d = document.getElementById("helpDlg"); return { text: d.innerText, setupBtn: !document.getElementById("helpSetupBtn").closest(".dlgfoot").hidden }; });
  ok(/Hugging Face/.test(help.text) && help.setupBtn, "help: the setup text, Run setup again shown");
  await shot(page, "6_help");
  await page.click("#helpSetupBtn");
  await waitStep(page, "where");
  ok(!(await page.evaluate(() => document.getElementById("helpDlg").open)), "Run setup again → setup open at 'where', help closed");
  await page.click("#suInBrowser");
  await waitStep(page, "model");
  await sleep(500); const mnote = await text(page, "suModelNote"); ok(mnote.startsWith("The model is already") && (await page.evaluate(() => !document.getElementById("suNext").hidden)), `model present: "${mnote}" + Next, nothing downloaded`);
  await page.click("#suClose");

  // ── model removed → reload → setup opens; closed without a model → Read refused with the setup link ──
  await page.evaluate(async () => { const root = await navigator.storage.getDirectory(); await root.removeEntry("models", { recursive: true }); try { await root.removeEntry("model-source.json"); } catch {} });
  await page.reload({ waitUntil: "networkidle0" });
  await waitStep(page, "where");
  await page.click("#suClose");
  // Read is disabled without the model (app.js applyState), its title says why; the status line links
  // the setup
  await page.waitForFunction(() => /^Set up the model first/.test(document.getElementById("readBtn").title));
  const link = await page.evaluate(() => { const a = document.querySelector("#status a"); return a ? a.textContent : null; });
  const rb = await page.evaluate(() => ({ dis: document.getElementById("readBtn").disabled, title: document.getElementById("readBtn").title }));
  ok(rb.dis && link === "Set up OCR Lite", `Read disabled without the model ("${rb.title}"), status: "${await status(page)}"`);
  await shot(page, "7_read_refused");
  await page.click("#status a");
  await waitStep(page, "where");
  ok(true, "the status link reopens the setup");

  // ── phone width + dark ──
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
  await sleep(300);
  const fit = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, vw: innerWidth, dlg: document.getElementById("setupDlg").getBoundingClientRect().width }));
  ok(fit.sw <= fit.vw && fit.dlg <= fit.vw, `phone: no horizontal scroll ${JSON.stringify(fit)}`);
  await shot(page, "8_phone_dark_where");
  await page.close();

  // ── standalone (an installed app): no install step, straight to the model ──
  const ctx = await browser.createBrowserContext();
  const p2 = await ctx.newPage();
  await prep(p2, { standalone: true });
  await p2.goto(BASE, { waitUntil: "networkidle0" });
  await waitStep(p2, "model");
  ok(await p2.evaluate(() => document.querySelector('#suSteps li[data-for="where"]').className === "done"), "standalone: no install step, opens at the model step");
  await shot(p2, "9_standalone_model");
  await p2.close();
  await ctx.close();

  // ── less storage than the model needs: said plainly, no download ──
  const ctx2 = await browser.createBrowserContext();
  const p3 = await ctx2.newPage();
  await prep(p3, { quota: 9e8 });
  await p3.goto(BASE, { waitUntil: "networkidle0" });
  await waitStep(p3, "where");
  await p3.click("#suInBrowser");
  await waitStep(p3, "model");
  await p3.waitForFunction(() => document.getElementById("suModelErr").textContent !== "");
  ok(/offers only 0\.9 GB .* needs 1\.5 GB/.test(await text(p3, "suModelErr")) && (await p3.evaluate(() => document.getElementById("suProgress").hidden)), `quota: ${await text(p3, "suModelErr")}`);
  await shot(p3, "10_quota");
  await p3.close();
  await ctx2.close();

  await browser.close();
  srv.kill();
  fs.rmSync(profile, { recursive: true, force: true });
  R.done();
})().catch(async (e) => { console.log("ERROR", e.message); process.exitCode = 1; if (srv) srv.kill("SIGTERM"); process.exit(1); });
