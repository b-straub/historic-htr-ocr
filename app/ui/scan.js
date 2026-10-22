// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The read's "scanner" over the page image (a little fun while the read is prepared),
// driven by the ocr:phase events (ocr-decode Phase):
//   encodingImage                 a glowing line sweeps top → bottom, again and again; behind it the model's real patch
//                                 grid (the bake plan: scaled image + white pad, `grid` px tiles in the model's input)
//                                 lights up tile by tile as the line passes and settles to a faint grid
//   loadingText, readingPrompt    the grid fades, the page outline pulses gently — unless the engine encodes the image
//                                 inside its prefill (llama.cpp: encodeMs null), then the line keeps sweeping
//   decoding / done / error       the overlay goes away
// prefers-reduced-motion → a thin static progress bar under the image instead (one step per phase).
// Costs no GPU time the engine needs: a transform on the line and a small 2D canvas redrawn at ≤ 30 fps on this thread.

const SWEEP_MS = 2600;   // one pass of the line over the page
const GLOW_MS = 520;     // a tile's afterglow once the line passed it
const BASE = 0.07;       // a passed tile's resting brightness (the faint grid)
const PEAK = 0.42;
const FRAME_MS = 33;

export function scanOverlay(stage, img) {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const root = document.createElement("div");
  root.className = "scan";
  root.hidden = true;
  root.innerHTML = '<canvas class="scan-grid"></canvas><div class="scan-line"></div><div class="scan-outline"></div><div class="scan-bar"><span></span></div>';
  stage.insertBefore(root, stage.querySelector("#rect"));
  const canvas = root.querySelector(".scan-grid");
  const line = root.querySelector(".scan-line");
  const ctx = canvas.getContext("2d");

  let raf = 0;
  let t0 = 0;
  let last = 0;
  let tiles = [];       // { x, y, w, h, at } in CSS px; `at` = when in a sweep the line reaches it (ms)
  let geom = "";        // the size + plan the tiles were built for
  let tint = "91, 140, 255";
  let fadeTimer = 0;
  let plan = null;      // the bake plan being swept

  // the model's patch grid in the image's displayed scale: the image is the plan's scaled image, the model input adds
  // the white pad (left / top) — tile (i, j) covers model px [i·g, (i+1)·g) → displayed (i·g − left) · s, clipped
  function layout(plan) {
    const W = img.clientWidth, H = img.clientHeight;
    const key = `${W}×${H}:${JSON.stringify(plan)}`;
    if (key === geom) return;
    geom = key;
    // zoomed in, the image can be 8× the pane: the canvas resolution is capped (≤ 4 MP), the tiles are soft glows anyway
    const dpr = Math.min(window.devicePixelRatio || 1, Math.sqrt(4e6 / Math.max(1, W * H)));
    canvas.width = Math.max(1, Math.round(W * dpr));
    canvas.height = Math.max(1, Math.round(H * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const s = W / plan.scaledW, g = plan.grid;
    const cols = (plan.scaledW + plan.left + plan.right) / g, rows = (plan.scaledH + plan.top + plan.bottom) / g;
    tiles = [];
    for (let j = 0; j < rows; j++) {
      const y0 = Math.max(0, (j * g - plan.top) * s), y1 = Math.min(H, ((j + 1) * g - plan.top) * s);
      if (y1 <= y0) continue;
      for (let i = 0; i < cols; i++) {
        const x0 = Math.max(0, (i * g - plan.left) * s), x1 = Math.min(W, ((i + 1) * g - plan.left) * s);
        if (x1 <= x0) continue;
        // a little scatter so the tiles light one by one, not as whole rows
        const jitter = (Math.random() - 0.3) * (y1 - y0) * 1.6;
        tiles.push({ x: x0, y: y0, w: x1 - x0, h: y1 - y0, at: (Math.min(H, Math.max(0, (y0 + y1) / 2 + jitter)) / H) * SWEEP_MS });
      }
    }
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (now - last < FRAME_MS) return;
    last = now;
    layout(plan);   // follows zoom / pan: the overlay is laid out in the image's displayed size
    const H = img.clientHeight;
    const t = now - t0, phase = t % SWEEP_MS, sweeps = Math.floor(t / SWEEP_MS);
    line.style.transform = `translateY(${(phase / SWEEP_MS) * H}px)`;
    ctx.clearRect(0, 0, img.clientWidth, H);
    for (const p of tiles) {
      if (sweeps === 0 && phase < p.at) continue; // not reached yet on the first pass
      const since = phase >= p.at ? phase - p.at : phase + SWEEP_MS - p.at;
      const a = BASE + PEAK * Math.exp(-since / GLOW_MS);
      ctx.fillStyle = `rgba(${tint}, ${a.toFixed(3)})`;
      ctx.fillRect(p.x + 0.5, p.y + 0.5, p.w - 1, p.h - 1);
    }
  }

  function stopLoop() {
    cancelAnimationFrame(raf);
    raf = 0;
  }

  function sweep(p) {
    clearTimeout(fadeTimer);
    root.classList.remove("prompt");
    plan = p;
    layout(plan);
    if (!raf) {
      t0 = performance.now();
      last = 0;
      raf = requestAnimationFrame(frame);
    }
  }

  function bar(step) {
    root.classList.add("still");
    root.querySelector(".scan-bar span").style.width = `${step}%`;
  }

  function show() {
    const acc = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
    const m = /^#([0-9a-f]{6})$/i.exec(acc);
    if (m) {
      const n = parseInt(m[1], 16);
      tint = `${n >> 16}, ${(n >> 8) & 255}, ${n & 255}`;
    }
    root.hidden = false;
  }

  function stop() {
    stopLoop();
    clearTimeout(fadeTimer);
    root.hidden = true;
    root.classList.remove("prompt", "still");
    geom = "";
  }

  // p = the ocr:phase payload, bake = the bake plan of the image being read (S.prev.bake)
  function phase(p, bake) {
    if (!img.clientWidth || !bake) { stop(); return; }
    show();
    const inPrefill = p.phase === "readingPrompt" && p.encodeMs === null; // llama.cpp: the image is encoded in here
    if (reduce.matches) {
      if (p.phase === "encodingImage" || inPrefill) bar(20);
      else if (p.phase === "loadingText" || p.phase === "readingPrompt") bar(55);
      else if (p.phase === "decoding") bar(85);
      else stop();
      return;
    }
    if (p.phase === "encodingImage" || inPrefill) {
      sweep(bake);
    } else if (p.phase === "loadingText" || p.phase === "readingPrompt") {
      if (root.classList.contains("prompt")) return; // loadingText → readingPrompt: the fade already runs
      root.classList.add("prompt");                // CSS: grid + line fade out, outline pulses
      fadeTimer = setTimeout(stopLoop, 900);       // the canvas is invisible by then
    } else {
      stop();
    }
  }

  return { phase, stop };
}
