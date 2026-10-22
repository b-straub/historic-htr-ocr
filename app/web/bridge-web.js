// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The host bridge of the browser build — the interface ui/app.js talks to:
//   commands  served by ocr-web's wasm on this thread (image work) or by the engine worker (worker.js: model + read);
//             errors reject with the message string
//   events    ocr:status / ocr:phase / ocr:text / ocr:done / ocr:error from the worker's messages
//   files     Open = a file input, drag-drop = HTML5 drop, Save = a download (Blob URL)
//   settings  { preset, modelUrl } + the user's presets in localStorage; the model lives in OPFS (Settings: download
//             it, see what is stored, delete) — the other sources (upload the two files, download them from a folder
//             URL, `?model=<url>` streaming by HTTP range requests) are switched off: web/config.js MODEL_SOURCES
//   model     the model's Hugging Face repo (web/config.js MODEL). A first start with no model in OPFS says so; the
//             setup and Settings offer the download
//   device    a lost GPU device / GPU error (the worker's gpuFatal) and the crash marker (below): a device that ran out
//             of memory is flagged, never retried silently
//   setup     the first-run setup's host side (ui/setup.js): the device check (gpu-check.js), the model's presence,
//             the download with progress and cancel
import init, * as ocr from "./pkg/ocr_web.js";
import { osOf, runGpuCheck } from "./gpu-check.js";
import { LINKS, MODEL, MODEL_BASE as HUB_BASE, MODEL_FROM, MODEL_SOURCES } from "./config.js";

const $ = (id) => document.getElementById(id);
const SETTINGS = "ocrlite.settings";
const USER_PRESETS = "ocrlite.userPresets";
const MODEL_FILES = ["text-q8_0.gguf", "mmproj-bf16.gguf"];
const LOCALHOST = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
// the model folder: the Hugging Face repo's resolve/main/ (web/config.js). ?modelBase=<folder> replaces it ONLY on
// localhost — the build tested against local model files (web/serve.py's /model/); anywhere else it is ignored (and
// said so on the console)
const MODEL_BASE = (() => {
  const asked = new URLSearchParams(location.search).get("modelBase");
  if (asked === null) {
    return HUB_BASE;
  }
  if (!LOCALHOST) {
    console.info(`OCR Lite: ?modelBase=${asked} ignored — a test override, honoured only on localhost`);
    return HUB_BASE;
  }
  console.info(`OCR Lite: test model base ${asked} (?modelBase=, localhost only) instead of ${HUB_BASE}`);
  return asked;
})();
// ?gpuCheck=skip (localhost only, like ?modelBase=): the setup's device check passes without running — the browser
// tests on a machine without a GPU (CI) walk the setup this way; anywhere else the parameter is ignored
const SKIP_GPU_CHECK = (() => {
  const asked = new URLSearchParams(location.search).get("gpuCheck");
  if (asked === null) {
    return false;
  }
  if (asked !== "skip" || !LOCALHOST) {
    console.info(`OCR Lite: ?gpuCheck=${asked} ignored — a test override (skip), honoured only on localhost`);
    return false;
  }
  console.info("OCR Lite: the device check is skipped (?gpuCheck=skip, localhost only)");
  return true;
})();
// where a model may come from: web/config.js MODEL_SOURCES
const allows = (source) => MODEL_SOURCES.includes(source);

// no WebGPU: the app still starts, so the setup's device check can say why and what to do (gpu-check.js); Read is refused
// (blockedNote)
const NO_WEBGPU = "gpu" in navigator ? null : "OCR Lite needs WebGPU — this browser has none (Chrome / Edge 113+, Safari 26+; on Linux Chrome: enable it in chrome://flags)";
await init();

const modelParam = new URLSearchParams(location.search).get("model");
if (modelParam !== null && !allows("stream")) {
  console.info(`OCR Lite: ?model=${modelParam} ignored — streaming the model is switched off (MODEL_SOURCES: ${MODEL_SOURCES.join(", ")}); the model comes from this browser's storage`);
}
const httpModel = allows("stream") ? modelParam : null;
const modelSource = () => (httpModel ? { kind: "http", base: new URL(httpModel, location.href).href } : { kind: "opfs" });

// ── storage (localStorage may be unavailable: then nothing persists, every read is the default) ──────────
function load(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function store(key, value) {
  try { localStorage.setItem(key, value); } catch (e) { throw new Error(`settings not saved: ${e.message}`); }
}

// ── crash marker + unsupported device (on an iPhone, Safari died right after the model load and the console showed
// nothing — iOS kills a process that runs out of memory, the page hears nothing). No browser tells a page
// how much GPU memory is FREE (there is no such API; Safari has not even navigator.deviceMemory), so running out can
// only be flagged after it happened:
//   marker       before each heavy GPU step (model load, image encode, prefill, decode) { step, time, build } goes to
//                localStorage; it is cleared when the read ends, done or failed normally. Still set at the next start =
//                the tab or its GPU process was killed during that step.
//   unsupported  that marker, or a lost GPU device / GPU out-of-memory error (the worker's gpuFatal), marks this
//                device "unsupported": the notice says why, and Read stays refused until "Try anyway" clears it.
// Without localStorage (private mode, blocked site data) no marker can be kept: Settings says so.
const CRASH_MARKER = "ocrlite.crashMarker";
const UNSUPPORTED = "ocrlite.unsupported";
// the engine's GPU buffers at the peak of a read, by ocr-wgpu's ledger (gpu.rs Buf), default preset. The load is
// STAGED (engine.rs load_and_read): vision weights 790.5 MiB + the encoder's work, then — the vision weights freed —
// text weights 604.1 MiB + KV cache + work. MEASURED (native M3, F16Coop, humboldt 44v, 1204 × 1540, 1893 tokens):
// peak 1158 MiB in the decode (before staging 1949 MiB), vision 1113. DERIVED from the earlier unstaged measurements
// (the same phases minus the other weight set): the largest test page's encode on F16 1984 → 1379 MiB = 1.45 GB; a
// read that runs to its 7024-token cap (the full KV cache) 2293 → 1503 MiB = 1.58 GB.
const GPU_NEED = "about 1.4 GB (1.6 GB for a read that runs to its token cap)";
const STEP = { encodingImage: "image encode", loadingText: "model load (text)", readingPrompt: "prefill", decoding: "decode" };
let markerError = null;  // why the crash marker cannot be kept (storage unavailable), shown in Settings
let currentStep = null;  // the heavy GPU step running now (null: none)
let gpuDead = null;      // the engine's death this session (gpuFatal): every further read fails with it
function buildId() {
  const meta = document.querySelector('meta[name="ocrlite-build"]');
  return meta ? JSON.parse(meta.content).git : "dev (web/serve.py)";
}
function keep(key, value) {
  try {
    if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, JSON.stringify(value));
  } catch (e) {
    markerError = e.message;
    console.error(`OCR Lite: ${key} not kept: ${e.message}`);
  }
}
function setMarker(step) {
  currentStep = step;
  keep(CRASH_MARKER, { step, time: Date.now(), build: buildId() });
}
function clearMarker() {
  currentStep = null;
  keep(CRASH_MARKER, null);
}
let unsupported = JSON.parse(load(UNSUPPORTED) ?? "null");
function markUnsupported(u) {
  unsupported = u;
  keep(UNSUPPORTED, u);
}
// a marker left by the last run: it never finished its step
{
  const m = JSON.parse(load(CRASH_MARKER) ?? "null");
  if (m) {
    markUnsupported({ ...m, cause: "killed" });
    keep(CRASH_MARKER, null);
  }
}
/// the crash of the run before (the crash marker / a lost GPU) as one message for the page's message box, and the device
/// cleared: the next read tries again (null: none)
export function takeCrashNote() {
  if (!unsupported) {
    return null;
  }
  const u = unsupported;
  markUnsupported(null);
  const when = new Date(u.time).toLocaleString();
  if (u.cause === "killed" && u.step === MEMORY_TEST_STEP) {
    return `This device does not have enough memory for OCR Lite: the device check's memory test (${when}) needed ${(MEMORY_TEST_BYTES / 2 ** 30).toFixed(1)} GB of GPU memory and the system ended the app.\n\nOCR Lite runs on Macs and iPads with Apple silicon and on recent iPhones (experimental). You can still try — the next read tries again.`;
  }
  const what = u.cause === "killed"
    ? `OCR Lite stopped during ${u.step} (${when}) — most likely this device ran out of memory.`
    : `OCR Lite lost the GPU during ${u.step} (${when}): ${u.why}.`;
  return `${what}\n\nReading in the browser with WebGPU is work in progress: on phones and tablets OCR Lite is not production-ready yet — a crash loses the text on the right, so copy or save what you edited. Real editing, with the work kept, is the next major step.\n\nThe next read tries again.`;
}
function tryAnyway() {
  markUnsupported(null);
  return "Device notice cleared — Read tries again";
}
function unsupportedNote(u) {
  const when = new Date(u.time).toLocaleString();
  const what = u.cause === "killed"
    ? `The last run stopped during ${u.step} without finishing (${when}, build ${u.build}) — most likely this device ran out of memory.`
    : `The last run lost the GPU during ${u.step} (${when}): ${u.why}.`;
  return { text: `${what} OCR Lite needs ${GPU_NEED} of GPU memory (an iPad / Mac with Apple silicon, or a desktop GPU).`, kind: "err", link: { label: "Try anyway", run: tryAnyway } };
}
/// why Read is refused now (null: it is not): the unsupported-device notice (with "Try anyway"), else a dead engine
export function blockedNote() {
  if (NO_WEBGPU !== null) return { text: NO_WEBGPU, kind: "err", link: null };
  if (unsupported) return unsupportedNote(unsupported);
  if (gpuDead) return { text: `${gpuDead} — reload the page to start again`, kind: "err", link: null };
  return null;
}
function gpuFatal(m) {
  gpuDead = m.message;
  if (m.kind === "lost" || m.kind === "outOfMemory") {
    markUnsupported({ step: currentStep ?? "no read (idle)", time: Date.now(), build: buildId(), cause: "gpu", why: m.message });
  }
  clearMarker();
  drawMemory();
  emit("ocr:error", `${m.message} — reload the page to start again`);
}

// ── the engine worker ────────────────────────────────────────────────
// a worker is ended and a new one started in the middle of each read (commands.read): startWorker sets one up
let worker = startWorker();
function startWorker() {
  const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  // first message: the sources the worker may store / read the model from (before any command — messages keep their
  // order)
  w.postMessage({ cmd: "sources", sources: MODEL_SOURCES });
  w.onmessage = onWorkerMessage;
  w.onerror = (e) => emit("ocr:error", `engine worker: ${e.message}`);
  return w;
}
const listeners = new Map();
const waiting = new Map();
let nextId = 1;
function emit(event, payload) {
  for (const cb of listeners.get(event) ?? []) cb({ payload });
}
function onWorkerMessage(e) {
  const m = e.data;
  if (m.event === "text") { emit("ocr:text", m.text); return; }
  if (m.event === "status") { emit("ocr:status", m.text); return; }
  if (m.event === "phase") {
    if (!(m.phase.phase in STEP)) throw new Error(`worker: unknown phase ${m.phase.phase}`);
    setMarker(STEP[m.phase.phase]);
    emit("ocr:phase", m.phase);
    return;
  }
  if (m.event === "step") { setMarker(m.step); return; }
  if (m.event === "gpuFatal") { gpuFatal(m); return; }
  if (m.event === "progress") { progress(m); return; }
  if (m.event === "loaded") { engineLoaded(m.info); return; }
  const w = waiting.get(m.id);
  waiting.delete(m.id);
  if (m.ok) w.resolve(m.result); else w.reject(m.error);
}
// the update's reload (pwa.js reloadNow): the engine worker ends before the page goes — its wasm heap and its GPU
// device (the weights are freed after each read anyway) are not left to be released behind the new page's start
window.addEventListener("ocrlite:release", () => worker.terminate());
function work(cmd, args = {}, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, ...args }, transfer);
  });
}

// model load / upload / download progress: the status bar while reading, the setup while it stores the model, the
// Settings bar otherwise
let progressTo = null; // the setup's callback ({ line, fraction }) while it stores the model
function progress(m) {
  const pct = m.total ? Math.round((100 * m.done) / m.total) : 0;
  const line = `${m.label === "load" ? "loading the model" : m.label === "upload" ? `storing ${m.name}` : `downloading ${m.name}`} · ${(m.done / 1e6).toFixed(0)}${m.total ? ` / ${(m.total / 1e6).toFixed(0)} MB (${pct} %)` : " MB"}`;
  if (m.label === "load") { emit("ocr:status", line); return; }
  if (progressTo) {
    progressTo({ line, fraction: m.total ? m.done / m.total : 0 });
    return;
  }
  $("modelProgress").hidden = false;
  $("modelProgress").value = m.total ? m.done / m.total : 0;
  $("modelProgressText").textContent = line;
}

// ── commands ──────────────────────────────────────────────────────────
let presets = null;
let previewUrl = null;
const user = () => load(USER_PRESETS);

function settings() {
  const s = JSON.parse(load(SETTINGS) ?? "null") ?? {};
  return { preset: s.preset ?? presets.defaultName, modelUrl: s.modelUrl ?? "" };
}

const commands = {
  startup() {
    const st = JSON.parse(ocr.startup(user()));
    presets = st.presets;
    return { ...st, settings: settings() };
  },
  preview({ ops, sharp }) {
    const p = ocr.preview(JSON.stringify(ops), sharp);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(new Blob([p.jpeg], { type: "image/jpeg" }));
    return { url: previewUrl, w: p.w, h: p.h, bake: JSON.parse(p.bake) };
  },
  read({ ops, preset }) {
    const p = [...presets.shipped, ...presets.user].find((x) => x.name === preset);
    if (!p) throw new Error(`no preset '${preset}'`);
    const blocked = blockedNote();
    if (blocked) throw new Error(blocked.text);
    if (!gatePassed) throw new Error("this read did not pass modelGate first");
    gatePassed = false;
    const b = ocr.bake(JSON.stringify(ops));
    const source = modelSource();
    stopAsked = false;
    // the staged read over two workers: worker A loads the vision weights, encodes, hands the embeddings back and is
    // ended; a new worker B loads the text weights and reads on — iOS kept counting freed GPU memory against the page
    // (the iPhone died at its 2048 MB in the text load), a dead worker's memory has to go
    work("encode", { source, rgb: b.rgb, w: b.w, h: b.h }, [b.rgb.buffer]).then(async (enc) => {
      worker.terminate();
      worker = startWorker();
      if (stopAsked) {
        return { ...enc.stats, imageTokens: enc.stats.nImg, promptTokens: 0, prefillMs: 0, decodeMs: 0, outTokens: 0, stoppedOnEog: false, cancelled: true, idleBytes: 0 };
      }
      const r = await work("readEncoded", { source, emb: enc.emb, encodeMs: enc.stats.encodeMs, preset: p, seed: b.seed }, [enc.emb.buffer]);
      // one read's stats from both halves: the loads (+ the new worker's engine start) added, the peak the larger
      const top = r.gpuMemory.peakBytes >= enc.stats.gpuMemory.peakBytes ? r.gpuMemory : { ...r.gpuMemory, peakBytes: enc.stats.gpuMemory.peakBytes, peakPhase: enc.stats.gpuMemory.peakPhase };
      return {
        ...r,
        loadMs: enc.stats.loadMs + r.restartMs + r.loadMs,
        load: { ...r.load, ms: enc.stats.load.ms + r.load.ms, visionWeightsBytes: enc.stats.load.visionWeightsBytes, weightsBytes: enc.stats.load.visionWeightsBytes + r.load.textWeightsBytes },
        gpuMemory: { ...top, phases: [...enc.stats.gpuMemory.phases, ...r.gpuMemory.phases] },
      };
    }).then(
      (stats) => {
        clearMarker();
        lastRead = stats;
        drawMemory();
        emit("ocr:done", stats);
      },
      (err) => {
        clearMarker();
        emit("ocr:error", gpuDead ? `${gpuDead} — reload the page to start again` : err);
      },
    );
    return JSON.parse(b.plan);
  },
  auto_crop({ ops, opts }) {
    return JSON.parse(ocr.auto_crop(JSON.stringify(ops), JSON.stringify(opts)));
  },
  cancel() {
    stopAsked = true; // a Stop during the encode: the second half does not start
    worker.postMessage({ cmd: "cancel" });
  },
  render({ text }) {
    return ocr.render(text);
  },
  save_settings({ settings: s }) {
    if (![...presets.shipped, ...presets.user].some((p) => p.name === s.preset)) throw new Error(`no preset '${s.preset}'`);
    store(SETTINGS, JSON.stringify({ preset: s.preset, modelUrl: s.modelUrl ?? "" }));
  },
  save_user_presets({ user: list }) {
    const next = JSON.parse(ocr.check_user_presets(JSON.stringify(list)));
    const sel = settings().preset;
    if (![...next.shipped, ...next.user].some((p) => p.name === sel)) throw new Error(`the selected preset '${sel}' would be gone — pick another first`);
    store(USER_PRESETS, JSON.stringify(next.user));
    presets = next;
    return next;
  },
};

export async function invoke(cmd, args = {}) {
  const f = commands[cmd];
  if (!f) throw `no command ${cmd}`;
  try {
    return await f(args);
  } catch (e) {
    throw e instanceof Error ? e.message : String(e);
  }
}

export async function listen(event, cb) {
  if (!listeners.has(event)) listeners.set(event, []);
  listeners.get(event).push(cb);
}

// ── files ─────────────────────────────────────────────────────────────
const picker = Object.assign(document.createElement("input"), { type: "file", accept: "image/png,image/jpeg,image/tiff,image/jxl,.tif,.tiff,.jxl" });

export function pickImage() {
  return new Promise((resolve) => {
    picker.value = "";
    picker.onchange = () => resolve(picker.files[0] ?? null);
    picker.oncancel = () => resolve(null);
    picker.click();
  });
}

export async function openImage(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  try {
    return JSON.parse(ocr.open_image(bytes, file.name));
  } catch (e) {
    throw e instanceof Error ? e.message : String(e);
  }
}

export function onDrop({ enter, leave, drop }) {
  let depth = 0;
  window.addEventListener("dragenter", (e) => { e.preventDefault(); if (depth++ === 0) enter(); });
  window.addEventListener("dragleave", () => { if (--depth === 0) leave(); });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    depth = 0;
    leave();
    const f = e.dataTransfer?.files?.[0];
    if (f) drop(f);
  });
}

function download(name, blob) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

export async function saveAs(what, { stem, ops, text }) {
  const name = `${stem || "read"}.${what}`;
  if (what === "txt") download(name, new Blob([text], { type: "text/plain;charset=utf-8" }));
  if (what === "html") download(name, new Blob([ocr.document(stem, text)], { type: "text/html;charset=utf-8" }));
  if (what === "png") download(name, new Blob([ocr.baked_png(JSON.stringify(ops))], { type: "image/png" }));
  return `${name} (download)`;
}

// ── before every read: modelGate ────────────────────────────────────
// the model in this browser must come from a source MODEL_SOURCES lists
let gatePassed = false; // modelGate passed for the next read (commands.read consumes it: no read skips the check)
/// may the next read run? → { ok, note } (ok: note null; refused: { text, kind, link } for the status bar). ok marks
/// the next read as checked (commands.read refuses one that was not).
export async function modelGate() {
  gatePassed = false;
  if (httpModel) {
    // ?model=<folder> streams by range requests: nothing stored to check
    gatePassed = true;
    return { ok: true, note: null };
  }
  const { record } = await work("source");
  if (record !== null && !fromAllowedSources(record)) {
    return { ok: false, note: { text: FOREIGN_MODEL, kind: "err", link: null } };
  }
  gatePassed = true;
  return { ok: true, note: null };
}

/// every stored model file came from a source MODEL_SOURCES lists (the worker's record: per file, or the record's own
/// source for a record without `files`)
function fromAllowedSources(record) {
  const sources = record.files ? Object.values(record.files) : [record.source];
  return sources.every(allows);
}
const FOREIGN_MODEL = `The model in this browser did not come from ${MODEL_FROM} and is not used`;

// ── settings: the model in OPFS ───────────────────────────────────────
// the loaded engine's formats (the worker's "loaded" event): weights = how the files store them, compute = what the
// GPU computes in — two different things (bf16 vision weights are computed in f16)
let engine = null;
let backendName = "WebGPU";
const formatLine = () => (engine ? `weights: ${engine.weights} · compute: ${engine.computeLabel}${engine.adapter ? ` on ${engine.adapter}` : ""}` : "weights / compute: shown once the model is loaded (first read)");
function engineLoaded(info) {
  engine = info;
  $("backendInfo").textContent = engineLabel(backendName);
  $("modelFormatWeb").textContent = formatLine();
  drawMemory();
}
// the GPU memory the engine counted (ocr-wgpu's ledger): between reads (no weights — each read loads them and frees
// them after it), the last read's weights and peak
let stopAsked = false; // a Stop since the read started (commands.read: between its two workers)
let lastRead = null; // the worker's stats of the last finished read (ReadStats + loadMs, load, idleBytes)
function drawMemory() {
  const parts = [];
  if (engine) {
    const idle = lastRead ? lastRead.idleBytes : engine.idleBytes;
    parts.push(`GPU memory between reads: ${mb(idle)} (each read loads the weights and frees them after it)`);
    if (lastRead) {
      const g = lastRead.gpuMemory;
      parts.push(`last read: weights ${mb(lastRead.load.weightsBytes)} · peak ${mb(g.peakBytes)} (in its ${g.peakPhase}) · load ${(lastRead.loadMs / 1000).toFixed(1)} s`);
    } else {
      parts.push("weights + peak: shown after the first read");
    }
  } else {
    parts.push(`GPU memory: shown once the model is loaded — a full page needs ${GPU_NEED}`);
  }
  if (gpuDead) parts.push(`the engine is dead: ${gpuDead}`);
  if (markerError) parts.push(`crash marker off (browser storage: ${markerError})`);
  $("gpuMemoryWeb").textContent = parts.join(" · ");
}

export function engineLabel(backend) {
  backendName = backend;
  const where = httpModel ? `model streamed from ${httpModel}` : `model in browser storage (${MODEL.name})`;
  return engine ? `${backend} · ${where} · weights: ${engine.weights} · compute: ${engine.computeLabel}` : `${backend} · ${where}`;
}

const mb = (b) => `${(b / 2 ** 20).toFixed(1)} MiB`;

async function drawStore() {
  const t = $("opfsTable");
  let files;
  try {
    files = await work("list");
  } catch (e) {
    t.innerHTML = "";
    $("opfsQuota").textContent = `browser storage unavailable: ${e}`;
    return;
  }
  t.innerHTML = "<tr><th>File</th><th>Size</th><th></th></tr>";
  for (const name of new Set([...MODEL_FILES, ...files.map((f) => f.name)])) {
    const f = files.find((x) => x.name === name);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${name}</td><td>${f ? mb(f.size) : "— missing"}</td><td></td>`;
    if (f) {
      const b = Object.assign(document.createElement("button"), { type: "button", className: "btn small", textContent: "Delete" });
      b.onclick = async () => {
        try { await work("delete", { name }); } catch (e) { $("settingsErr").textContent = String(e); }
        drawStore();
        modelChanged();
      };
      tr.lastChild.append(b);
    }
    t.append(tr);
  }
  const est = navigator.storage.estimate ? await navigator.storage.estimate() : null;
  const persisted = navigator.storage.persisted ? await navigator.storage.persisted() : false;
  $("opfsQuota").textContent = `${est ? `${mb(est.usage)} used of ${mb(est.quota)} · ` : ""}${persisted ? "kept (persistent storage granted)" : "the browser may evict it under storage pressure"}`;
  const { record } = await work("source");
  $("modelSourceWeb").textContent = record === null ? "" : sourceLine(record);
  if (record !== null && !fromAllowedSources(record)) {
    $("modelSourceWeb").textContent += ` · not used: only a model downloaded from ${MODEL_FROM} reads — download it`;
  }
}
/// Settings: where the model in this browser came from
function sourceLine(record) {
  return record.source === "hub" ? `from ${MODEL_FROM}` : record.source === "url" ? `from ${record.url}` : record.source === "upload" ? "uploaded" : "of unknown origin";
}

// the model in this browser changed (stored / deleted): the page re-asks startupNote for its status line (else, after
// a download, the status would still say "No model in this browser")
function modelChanged() {
  window.dispatchEvent(new Event("ocrlite:model-changed"));
}
// storing the model (Settings and the setup): ask for persistent storage first — it keeps the browser from evicting
// 1.4 GB — then the job; the page hears of the new model
async function storeModel(job) {
  if (navigator.storage.persist) {
    await navigator.storage.persist();
  }
  await job();
  modelChanged();
}
// the switched-off sources' controls (MODEL_SOURCES) are not shown at all
document.querySelector('label[for="modelFiles"]').hidden = !allows("upload");
$("modelUrl").hidden = !allows("url");
$("modelDownload").hidden = !allows("url");
$("modelOtherSources").hidden = !allows("upload") && !allows("url");
async function busy(job) {
  $("settingsErr").textContent = "";
  for (const id of ["modelFiles", "modelDownload", "modelFromHub"]) $(id).disabled = true;
  try {
    await storeModel(job);
    $("modelProgressText").textContent = "done";
  } catch (e) {
    $("settingsErr").textContent = String(e);
    $("modelProgressText").textContent = "";
  } finally {
    $("modelProgress").hidden = true;
    for (const id of ["modelFiles", "modelDownload", "modelFromHub"]) $(id).disabled = false;
    drawStore();
  }
}

$("modelFiles").onchange = () => {
  if (!allows("upload")) {
    $("settingsErr").textContent = `uploading model files is switched off — download the model from ${MODEL_FROM}`;
    return;
  }
  const files = [...$("modelFiles").files];
  $("modelFiles").value = "";
  const wrong = files.filter((f) => !MODEL_FILES.includes(f.name));
  if (wrong.length) { $("settingsErr").textContent = wrongFiles(wrong); return; }
  busy(() => work("upload", { files }));
};
const wrongFiles = (wrong) => `${wrong.map((f) => f.name).join(", ")}: the model files are ${MODEL_FILES.join(" + ")}`;
/// both model files from a folder URL into OPFS (Settings and the setup). The download streams each file straight
/// into OPFS (worker.js download — also from the Hugging Face repo: its resolve URLs answer a browser fetch with CORS,
/// through the redirect to its CDN)
async function downloadModel(url) {
  const base = new URL(url.endsWith("/") ? url : `${url}/`, location.href);
  // the model folder (or its localhost test override) → "hub", else "url"
  const own = base.href === new URL(MODEL_BASE, location.href).href;
  const record = own ? { source: "hub" } : { source: "url", url: base.href };
  if (!allows(record.source)) {
    throw new Error(`${base.href}: downloading from another folder is switched off — the model comes from ${MODEL_FROM}`);
  }
  // the hub repo may name the files differently (web/config.js MODEL.remoteFiles); stored under MODEL_FILES either way.
  // A test override (?modelBase=) holds MODEL_FILES itself
  const remote = own && base.href === new URL(HUB_BASE, location.href).href ? MODEL.remoteFiles : null;
  await work("download", { base: base.href, record, remote });
}
function downloadFrom(url) {
  busy(() => downloadModel(url));
}
$("modelDownload").onclick = () => {
  const base = $("modelUrl").value.trim();
  if (!base) { $("settingsErr").textContent = "enter the URL of a folder holding both model files"; return; }
  downloadFrom(base);
};
$("modelFromHub").textContent = `Download from ${MODEL_FROM}`;
$("modelFromHub").onclick = () => downloadFrom(MODEL_BASE);

/// version / commit / build time for the About box: stamped into index.html by web/dist.py; the dev server has none
// the memory after a read, for the breadcrumb (an iPad PWA killed by iOS leaves only that log): the wasm heaps of the
// worker and of this page (they only grow) + the engine's GPU ledger (between reads, and the last read's peak)
export async function memoryNote() {
  const w = await work("memory");
  const parts = [`wasm worker ${mb(w.wasmBytes)}`, `page ${mb(ocr.wasm_memory_bytes())}`];
  if (lastRead) {
    parts.push(`GPU between reads ${mb(lastRead.idleBytes)}`, `peak ${mb(lastRead.gpuMemory.peakBytes)} (${lastRead.gpuMemory.peakPhase})`);
  }
  return parts.join(" · ");
}

export async function buildInfo() {
  const meta = document.querySelector('meta[name="ocrlite-build"]');
  return meta ? JSON.parse(meta.content) : { version: "dev", git: "working tree (web/serve.py)", built: null };
}

/// the status bar's note at startup instead of "Ready" (null: Ready) — { text, kind, link, setup? }; setup: the model is
/// not in this browser yet, the page links its setup (ui/setup.js, which gets the model)
export async function startupNote() {
  if (NO_WEBGPU !== null) return { text: NO_WEBGPU, kind: "err", link: null, setup: true };
  if (!httpModel) {
    let missing;
    try {
      missing = await modelMissing();
    } catch (e) {
      return { text: `browser storage unavailable: ${e}`, kind: "err", link: null };
    }
    if (missing) {
      const { record } = await work("source");
      return { text: record !== null && !fromAllowedSources(record) ? FOREIGN_MODEL : "No model in this browser yet", kind: "err", link: null, setup: true };
    }
  }
  if (unsupported) return unsupportedNote(unsupported);
  if (httpModel) return null;
  // the source check at start (a pass here does not count for a read: each read checks again)
  const gate = await modelGate();
  gatePassed = false;
  return gate.note;
}

/// both model files in this browser's storage, from a source MODEL_SOURCES allows? (never asked when the model
/// streams: ?model=) — a model stored from a switched-off source counts as missing: the setup downloads the hub's
async function modelMissing() {
  const files = await work("list");
  if (!MODEL_FILES.every((n) => files.some((f) => f.name === n))) {
    return true;
  }
  const { record } = await work("source");
  return !fromAllowedSources(record);
}

// ── the first-run setup's host side (ui/setup.js) ─────────────────────
// No state is kept for it: the setup is due exactly while the model is missing from this browser's storage.
const MODEL_ROOM = 1.5e9; // the two files (1.4 GB) + headroom: less free storage for this site → the setup says so
// the device check's memory test (iPhone / iPad): the larger half of a read — the image encode measured on the iPad's F16
// path at 1408 MiB — plus about a tenth; the crash marker's step while it runs
const MEMORY_TEST_BYTES = 1536 * 2 ** 20;
const MEMORY_TEST_STEP = "device check: memory test";
// the last device check's result — a per-viewer convenience (shown while the next check runs), never a gate: the check
// runs again whenever the setup opens
const GPU_CHECK = "ocrlite.gpuCheck";
export const setup = {
  /// the device check (gpu-check.js) → its verdict; onStep(verdict) once the adapter passed, before the benchmark
  async gpuCheck(onStep) {
    if (SKIP_GPU_CHECK) {
      return { verdict: "ok", headline: "Device check skipped (test: ?gpuCheck=skip on localhost)", detected: [], problems: [], notes: [], hints: [] };
    }
    const env = {
      ua: navigator.userAgent,
      maxTouchPoints: navigator.maxTouchPoints,
      secure: window.isSecureContext,
      flagged: unsupported ? unsupportedNote(unsupported).text : null,
    };
    const r = await runGpuCheck(navigator.gpu, env, onStep);
    // iPhone / iPad: the memory of the larger half of a read, held once — in the engine worker, ended after it (its
    // memory goes with it); iOS killing the page here is caught by the crash marker before the model's download
    const os = osOf(env.ua, env.maxTouchPoints);
    if ((os === "iphone" || os === "ipad") && r.verdict !== "reject") {
      setMarker(MEMORY_TEST_STEP);
      try {
        const m = await work("memoryTest", { bytes: MEMORY_TEST_BYTES });
        r.notes.push(`Memory test passed: ${(m.bytes / 2 ** 30).toFixed(1)} GB of GPU memory held (the larger half of a read, with room to spare) in ${(m.ms / 1000).toFixed(1)} s.`);
      } catch (e) {
        r.notes.push(`Memory test failed: ${e instanceof Error ? e.message : e}`);
      } finally {
        clearMarker();
        worker.terminate();
        worker = startWorker();
      }
    }
    try {
      localStorage.setItem(GPU_CHECK, JSON.stringify({ time: Date.now(), verdict: r.verdict, cls: r.cls ?? null, headline: r.headline }));
    } catch (e) {
      console.error(`OCR Lite: the device check's result not kept: ${e.message}`);
    }
    return r;
  },
  /// this browser has no WebGPU at all: the setup opens at start to say why (its device check)
  noWebGpu: NO_WEBGPU !== null,
  /// the last device check's { time, verdict, cls, headline }, or null (none, or storage unavailable)
  lastGpuCheck() {
    return JSON.parse(load(GPU_CHECK) ?? "null");
  },
  /// the setup is due: the model is not in this browser's storage (never with ?model=: it streams)
  async modelMissing() {
    if (httpModel) {
      return false;
    }
    return modelMissing();
  },
  /// where the model comes from, for the setup's texts + the project's links
  modelFrom: MODEL_FROM,
  modelName: MODEL.name,
  links: LINKS,
  /// free storage for this site → { free, need } in bytes, or null when the browser does not say
  async room() {
    if (!navigator.storage.estimate) {
      return null;
    }
    const est = await navigator.storage.estimate();
    return { free: est.quota - est.usage, need: MODEL_ROOM };
  },
  /// both files from the model folder → resolves when stored; rejects with "download cancelled" after cancel(), or the
  /// download's error
  download(onProgress) {
    return withProgress(onProgress, () => downloadModel(MODEL_BASE));
  },
  /// the user's own model files may be stored (MODEL_SOURCES "upload"): the setup shows "I have the files" only then
  canUpload: allows("upload"),
  /// model files picked by the user (both, or the one still missing)
  upload(files, onProgress) {
    if (!allows("upload")) {
      return Promise.reject(new Error(`uploading model files is switched off — the model comes from ${MODEL_FROM}`));
    }
    const wrong = files.filter((f) => !MODEL_FILES.includes(f.name));
    if (wrong.length) {
      return Promise.reject(new Error(wrongFiles(wrong)));
    }
    return withProgress(onProgress, () => work("upload", { files }));
  },
  cancel() {
    worker.postMessage({ cmd: "cancelDownload" });
  },
  modelFiles: MODEL_FILES,
};
async function withProgress(onProgress, job) {
  progressTo = onProgress;
  try {
    await storeModel(job);
  } finally {
    progressTo = null;
  }
}

export function settingsOpen(s) {
  $("modelUrl").value = s.modelUrl;
  $("modelFormatWeb").textContent = formatLine();
  drawMemory();
  $("modelProgressText").textContent = "";
  drawStore();
}

export function settingsCollect(s) {
  return { ...s, modelUrl: $("modelUrl").value.trim() };
}
