// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The browser build's end-to-end check: main-thread wasm opens + bakes the image exactly as the app does, the
// worker loads the model (HTTP ranges from the dev server's /model/, or OPFS — optionally filled from /model/ first)
// and reads it; the read is compared with the native ocr-wgpu read (`expect`). Every line also goes to POST /log so a
// headless run reports to the dev server's terminal. Ends with "SELFTEST PASS" / "SELFTEST FAIL".
// Every read loads the weights and frees them after it: each read logs its load (parse · weights: file reads · GPU
// copy) and the GPU memory alive after it — more than IDLE_MAX there fails the selftest (weights left on the GPU).
// lose=1 | encodingImage | readingPrompt | decoding: after the read(s), the lost-device check — the GPU device is
// destroyed as that phase of a read starts (1 = decoding), as a crashed GPU process loses it: that read must fail with the reason (not hang), the page must hear gpuFatal, and a further read
// must fail at once with the same reason.
import init, * as ocr from "./pkg/ocr_web.js";

const q = new URLSearchParams(location.search);
const out = document.getElementById("out");
function log(line) {
  out.textContent += line + "\n";
  console.log(line);
  fetch("/log", { method: "POST", body: line }).catch(() => {});
}

const worker = new Worker("worker.js", { type: "module" });
let next = 1;
const waiting = new Map();
let text = "";
let fatal = null;       // the worker's gpuFatal event
let onPhase = null;     // the lost-device check's hook on the phase events
worker.onmessage = (e) => {
  const m = e.data;
  if (m.event === "text") { text += m.text; return; }
  if (m.event === "step") { log(`  step ${m.step}`); return; }
  if (m.event === "gpuFatal") { fatal = m; log(`  gpuFatal ${m.kind}: ${m.message}`); return; }
  if (m.event === "progress") { if (m.done === m.total || m.done % (256 << 20) < (32 << 20)) log(`  ${m.label} ${m.name ?? ""} ${(m.done / 1e6).toFixed(0)} / ${(m.total / 1e6).toFixed(0)} MB`); return; }
  if (m.event === "status") { log(`  worker: ${m.text}`); return; }
  if (m.event === "loaded") return;
  if (m.event === "phase") {
    log(`  phase ${JSON.stringify(m.phase)}`);
    if (onPhase) onPhase(m.phase);
    return;
  }
  const w = waiting.get(m.id);
  waiting.delete(m.id);
  if (m.ok) w.resolve(m.result); else w.reject(new Error(m.error));
};
const call = (cmd, args = {}, transfer = []) => new Promise((resolve, reject) => {
  const id = next++;
  waiting.set(id, { resolve, reject });
  worker.postMessage({ id, cmd, ...args }, transfer);
});
// the selftest is a dev page (never in dist): it reads the model by HTTP ranges or fills OPFS from /model/ itself, so
// it allows "url" + "stream" for its own worker — the app's MODEL_SOURCES (web/config.js) stays "hub" only
worker.postMessage({ cmd: "sources", sources: ["url", "stream"] });
const mb = (b) => `${(b / 2 ** 20).toFixed(1)} MiB`;
const IDLE_MAX = 2 ** 20; // the GPU memory allowed between reads: the kernels' parameter ring (0.3 MiB), no weights
const memLine = (g) => `gpu memory: resident ${mb(g.residentBytes)} · read peak ${mb(g.peakBytes)} in ${g.peakPhase} · ${g.phases.map((p) => `${p.phase} ${mb(p.peak.weights + p.peak.kvCache + p.peak.work + p.peak.readback)} (kv ${mb(p.peak.kvCache)} · work ${mb(p.peak.work)})`).join(" | ")}`;

// the lost-device check (lose=…): destroy the device as the phase `at` of a read starts → { pass, verdict }
async function loseDevice(source, b, preset, at) {
  const read = () => call("read", { source, rgb: b.rgb.slice(), w: b.w, h: b.h, preset, seed: b.seed });
  const hang = (ms) => new Promise((resolve) => setTimeout(() => resolve("HUNG"), ms));
  fatal = null;
  text = "";
  onPhase = (p) => {
    if (p.phase !== at) return;
    onPhase = null;
    log(`  destroying the GPU device as ${at} starts`);
    call("debugLoseDevice").catch((e) => log(`  debugLoseDevice failed: ${e.message}`));
  };
  const first = await Promise.race([read().then(() => "FINISHED", (e) => e.message), hang(60000)]);
  log(`  the read during the loss: ${first}`);
  // device.lost resolves after the readback's rejection (Chrome: a few hundred ms later): wait for the page to hear it
  for (let t = 0; t < 100 && fatal === null; t++) await new Promise((r) => setTimeout(r, 100));
  const second = await Promise.race([read().then(() => "FINISHED", (e) => e.message), hang(10000)]);
  log(`  a further read: ${second}`);
  // the read during the loss fails (in Chrome its readback's rejection comes before device.lost: "GPU readback
  // failed …"); the lost reason reaches the page (gpuFatal) and every later read fails with it
  const ok = fatal !== null && fatal.message.startsWith("GPU device lost") && first.startsWith("GPU ") && second === fatal.message;
  return { pass: ok, verdict: ok ? `surfaced (the read: ${first} · gpuFatal ${fatal.kind}: ${fatal.message} · a further read fails with it)` : `NOT surfaced as expected (gpuFatal ${fatal ? fatal.kind : "never"})` };
}

(async () => {
  try {
    log(`selftest · ${navigator.userAgent}`);
    log(`navigator.gpu: ${"gpu" in navigator ? "yes" : "NO"} · crossOriginIsolated: ${self.crossOriginIsolated}`);
    await init();
    const st = JSON.parse(ocr.startup(null));
    const presetName = q.get("preset") ?? st.presets.defaultName;
    const preset = [...st.presets.shipped, ...st.presets.user].find((p) => p.name === presetName);
    if (!preset) throw new Error(`no preset ${presetName}`);
    const imgUrl = q.get("image") ?? "/data/small_a.png";
    const bytes = new Uint8Array(await (await fetch(imgUrl)).arrayBuffer());
    const opened = JSON.parse(ocr.open_image(bytes, imgUrl.split("/").pop()));
    const b = ocr.bake("[]");
    log(`image ${opened.stem} ${opened.w}×${opened.h} → baked ${b.w}×${b.h} · preset ${preset.name} · seed ${b.seed}`);

    const source = q.get("source") === "opfs" ? { kind: "opfs" } : { kind: "http", base: `${location.origin}/model` };
    if (source.kind === "opfs" && q.get("fill")) {
      const t = performance.now();
      const r = await call("download", { base: `${location.origin}/model`, record: { source: "url", url: `${location.origin}/model/` } });
      log(`OPFS filled from /model/ in ${((performance.now() - t) / 1000).toFixed(1)} s: ${r.map((f) => `${f.name} ${mb(f.size)}`).join(", ")}`);
    }
    if (source.kind === "opfs") log(`OPFS: ${(await call("list")).map((f) => `${f.name} ${mb(f.size)}`).join(", ")}`);
    const m0 = await call("memory");
    log(`worker memory before the engine: wasm ${mb(m0.wasmBytes)}${m0.jsHeapUsed ? ` · JS heap ${mb(m0.jsHeapUsed)}` : ""}`);
    const info = await call("load", { source });
    log(`engine open for ${info.source} in ${(info.openMs / 1000).toFixed(2)} s (device + kernels + tokenizer; ${mb(info.modelBytes)} of tensors, loaded per read) on ${info.adapter} (${info.backend}, compute ${info.compute} = ${info.computeLabel}, weights ${info.weights}, subgroups ${info.subgroups ? "on" : "off"}, shader checks ${info.shaderChecks ? "on" : "off"}) · maxStorageBufferBindingSize ${mb(info.maxStorageBufferBindingSize)} · maxBufferSize ${mb(info.maxBufferSize)}`);
    log(`gpu memory with the engine open: ${mb(info.idleBytes)}`);
    const m1 = await call("memory");
    log(`worker memory with the engine open: wasm ${mb(m1.wasmBytes)}${m1.jsHeapUsed ? ` · JS heap ${mb(m1.jsHeapUsed)}` : ""}`);
    let idleOk = true;

    // repeat=N: N reads on the loaded engine (the first read vs the warm ones); the comparison uses the last read
    const repeat = Math.max(1, Number(q.get("repeat") ?? 1));
    for (let run = 1; run <= repeat; run++) {
      text = "";
      const rgb = run === repeat && !q.get("lose") ? b.rgb : b.rgb.slice(); // lose=1 reads it again after
      const t0 = performance.now();
      const stats = await call("read", { source, rgb, w: b.w, h: b.h, preset, seed: b.seed }, [rgb.buffer]);
      const wall = performance.now() - t0;
      const m2 = await call("memory");
      log(`read${repeat > 1 ? ` ${run}/${repeat}` : ""}: ${stats.imageTokens} image + ${stats.promptTokens} prompt tokens · encode ${stats.encodeMs.toFixed(0)} ms · text prefill ${stats.prefillMs.toFixed(0)} ms · ${stats.outTokens} tokens in ${stats.decodeMs.toFixed(0)} ms (${(stats.outTokens / (stats.decodeMs / 1000)).toFixed(1)} tok/s, token choice ${stats.tokenChoice}) · eog ${stats.stoppedOnEog} · wall ${(wall / 1000).toFixed(1)} s`);
      const L = stats.load;
      log(`  load ${stats.loadMs.toFixed(0)} ms: headers ${L.parseMs.toFixed(0)} ms · weights ${L.ms.toFixed(0)} ms (file reads ${L.readMs.toFixed(0)} · gpu copy ${L.waitMs.toFixed(0)}) · weights ${mb(L.weightsBytes)} (vision ${mb(L.visionWeightsBytes)} · text ${mb(L.textWeightsBytes)}) · load peak ${mb(L.loadPeakBytes)}`);
      log(`worker memory after read: wasm ${mb(m2.wasmBytes)}${m2.jsHeapUsed ? ` · JS heap ${mb(m2.jsHeapUsed)}` : ""}`);
      log(memLine(stats.gpuMemory));
      log(`gpu memory after read ${run}: ${mb(stats.idleBytes)}${stats.idleBytes > IDLE_MAX ? ` — MORE than ${mb(IDLE_MAX)}: weights left on the GPU` : ""}`);
      if (stats.idleBytes > IDLE_MAX) idleOk = false;
    }
    log("----- read -----");
    log(text);
    log("----------------");
    const expect = q.get("expect");
    let verdict = "read only (no expect= given)";
    let pass = idleOk;
    if (expect) {
      const want = await (await fetch(expect)).text();
      if (want === text) {
        verdict = `identical to ${expect} (${[...text].length} chars)`;
      } else {
        const a = [...text], w = [...want];
        let i = 0;
        while (i < a.length && i < w.length && a[i] === w[i]) i++;
        verdict = `differs from ${expect} at char ${i} of ${a.length} / ${w.length}: …${a.slice(Math.max(0, i - 10), i + 30).join("")} vs …${w.slice(Math.max(0, i - 10), i + 30).join("")}`;
        pass = false;
      }
    }
    if (!idleOk) verdict += ` · GPU memory left after a read above ${mb(IDLE_MAX)}`;
    if (q.get("lose")) {
      const lost = await loseDevice(source, b, preset, q.get("lose") === "1" ? "decoding" : q.get("lose"));
      verdict += ` · lost device: ${lost.verdict}`;
      pass = pass && lost.pass;
    }
    log(`SELFTEST ${pass ? "PASS" : "FAIL"}: ${verdict}`);
  } catch (e) {
    log(`SELFTEST FAIL: ${e instanceof Error ? e.message : e}`);
  }
})();
