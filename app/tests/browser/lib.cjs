// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Shared parts of the browser tests (headless Chrome via puppeteer-core):
//   site        the built site (app/ci.sh site → <repo>/_site), served by web/serve.py --dist as GitHub Pages serves it,
//               under the Pages sub-path /historic-htr-ocr/ — the app at …/app/
//   chrome      $CHROME, else the platform's usual Google Chrome / Chromium
//   model       $OCR_LITE_MODEL_DIR (a folder with text-q8_0.gguf + mmproj-bf16.gguf): the parts that need the real model
//               run only with it — without it they are reported as SKIP, never passed silently
//   results     every check prints PASS / FAIL / SKIP; done() prints the totals and sets the exit code (FAIL → 1)
const { spawn, execFileSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const puppeteer = require("puppeteer-core");

const APP = path.resolve(__dirname, "../..");
const REPO = path.dirname(APP);
const SERVE = path.join(APP, "web", "serve.py");
const SITE = process.env.OCR_LITE_SITE ?? path.join(REPO, "_site");
const PREFIX = "/historic-htr-ocr/";
const MODEL_DIR = process.env.OCR_LITE_MODEL_DIR ?? null;
const OUT_ROOT = process.env.OCR_LITE_TEST_OUT ?? path.join(REPO, "test-output");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/// one test's checks and their totals
function results(name) {
  const n = { pass: 0, fail: 0, skip: 0 };
  return {
    ok(c, m) {
      console.log(`${c ? "PASS" : "FAIL"} ${m}`);
      n[c ? "pass" : "fail"]++;
    },
    skip(m) {
      console.log(`SKIP ${m}`);
      n.skip++;
    },
    done() {
      console.log(`${name}: ${n.pass} passed, ${n.fail} failed, ${n.skip} skipped`);
      if (n.fail > 0) {
        process.exitCode = 1;
      }
    },
  };
}

function chrome() {
  if (process.env.CHROME) {
    return process.env.CHROME;
  }
  const mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  if (fs.existsSync(mac)) {
    return mac;
  }
  for (const name of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]) {
    try {
      return execFileSync("which", [name], { encoding: "utf8" }).trim();
    } catch {
      // not this one
    }
  }
  throw new Error("no Chrome found — set CHROME to its executable");
}

async function launch(extra = {}) {
  const args = ["--enable-unsafe-webgpu", "--window-size=1440,920"];
  if (process.platform === "linux") {
    // CI runners (Ubuntu 24.04) restrict the user namespaces Chrome's sandbox needs
    args.push("--no-sandbox");
  }
  return puppeteer.launch({ executablePath: chrome(), headless: true, args, ...extra });
}

/// web/serve.py --dist <site> --prefix /historic-htr-ocr/ on `port` → { base (the app's URL), kill() }
async function serve(port, { modelDir = null, dataDir = null } = {}) {
  if (!fs.existsSync(path.join(SITE, "app", "index.html"))) {
    throw new Error(`${SITE}/app/index.html missing — build the site first (app/ci.sh site)`);
  }
  const args = [SERVE, "--port", String(port), "--dist", SITE, "--prefix", PREFIX];
  if (modelDir) {
    args.push("--model-dir", modelDir);
  }
  if (dataDir) {
    args.push("--data-dir", dataDir);
  }
  const srv = spawn("python3", args, { stdio: ["ignore", "ignore", "inherit"] });
  const root = `http://127.0.0.1:${port}${PREFIX}`;
  for (let i = 0; ; i++) {
    try {
      const r = await fetch(`${root}app/`);
      if (r.ok) {
        break;
      }
    } catch {
      // not up yet
    }
    if (i > 100) {
      srv.kill();
      throw new Error("serve.py did not start");
    }
    await sleep(100);
  }
  return { root, base: `${root}app/`, kill: () => srv.kill() };
}

/// a fresh output folder for screenshots of the test `name`
function outDir(name) {
  const d = path.join(OUT_ROOT, name);
  fs.rmSync(d, { recursive: true, force: true });
  fs.mkdirSync(d, { recursive: true });
  return d;
}

/// a stand-in model: the two file names with junk content (text 60 MiB, sparse) — enough for the setup's download,
/// progress and cancel; a read on it fails at the load
function fakeModel() {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "ocrlite-fake-model-"));
  const fd = fs.openSync(path.join(d, "text-q8_0.gguf"), "w");
  fs.ftruncateSync(fd, 60 * 1024 * 1024);
  fs.closeSync(fd);
  fs.writeFileSync(path.join(d, "mmproj-bf16.gguf"), "not a gguf file\n");
  return d;
}

/// two synthetic test images as PNG files (drawn in the browser): big 3000 × 2000 — paper with a left-to-right shading
/// (darker at the left), a frame and strokes of "ink"; small 700 × 168 — a line of strokes. → { big, small }
async function testImages(browser) {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "ocrlite-images-"));
  const page = await browser.newPage();
  const png = await page.evaluate(() => {
    const draw = (w, h, shade) => {
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      const x = c.getContext("2d");
      const g = x.createLinearGradient(0, 0, w, 0);
      g.addColorStop(0, shade ? "rgb(150,140,120)" : "rgb(235,230,220)");
      g.addColorStop(0.35, "rgb(222,214,198)");
      g.addColorStop(1, "rgb(238,232,220)");
      x.fillStyle = g;
      x.fillRect(0, 0, w, h);
      x.strokeStyle = "rgb(40,30,25)";
      x.lineWidth = Math.max(2, h / 200);
      let seed = 7;
      const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
      const rows = Math.max(1, Math.floor(h / 120));
      for (let r = 0; r < rows; r++) {
        const y = h * (0.2 + 0.6 * (rows === 1 ? 0.5 : r / (rows - 1)));
        x.beginPath();
        for (let px = w * 0.15; px < w * 0.85; px += 18) {
          x.lineTo(px, y + (rnd() - 0.5) * 30);
        }
        x.stroke();
      }
      return c.toDataURL("image/png").split(",")[1];
    };
    return { big: draw(3000, 2000, true), small: draw(700, 168, false) };
  });
  await page.close();
  const big = path.join(d, "big.png");
  const small = path.join(d, "small.png");
  fs.writeFileSync(big, Buffer.from(png.big, "base64"));
  fs.writeFileSync(small, Buffer.from(png.small, "base64"));
  return { big, small };
}

/// the setup's visible step, or null when it is closed
const setupStep = (page) => page.evaluate(() => {
  const d = document.getElementById("setupDlg");
  const s = [...d.querySelectorAll("[data-step]")].find((x) => !x.hidden);
  return d.open && s ? s.dataset.step : null;
});

module.exports = { APP, REPO, SITE, PREFIX, MODEL_DIR, sleep, results, launch, serve, outDir, fakeModel, testImages, setupStep };
