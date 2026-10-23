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
// the GPU memory the engine holds: its fixed memory pool (ocr-wgpu gpu.rs Pool::fix, allocated when the engine starts:
// 24 × 64 MiB + the largest tensor's block = 1693.6 MiB, the Chrome breadcrumb "GPU pool") + the 32 MiB upload buffer
// + the readback (32 MiB) ≈ 1758 MiB ≈ 1.8 GB. The largest page (1540 × 1540) peaks at 1533 MiB in use inside it
// (native, F16 path).
const GPU_NEED = "about 1.8 GB";
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
    return `This device does not have enough memory for OCR Lite: the device check's memory test (${when}) needed ${(MEMORY_TEST_BYTES / 1e9).toFixed(1)} GB of GPU memory and the system ended the app.\n\nOCR Lite runs on Macs and iPads with Apple silicon and on recent iPhones (experimental). You can still try — the next read tries again.`;
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
// the app was away (app.js wentAway → awayFromApp) since the engine last started: a GPU loss then is the OS's doing
// (a frozen / backgrounded page), no sign of the device — it is not marked unsupported; the app restarts the engine
// (resetEngine) and the read when it is back
let wasAway = false;
export function awayFromApp() {
  wasAway = true;
}
function gpuFatal(m) {
  gpuDead = m.message;
  if ((m.kind === "lost" || m.kind === "outOfMemory") && !wasAway && document.visibilityState !== "hidden") {
    markUnsupported({ step: currentStep ?? "no read (idle)", time: Date.now(), build: buildId(), cause: "gpu", why: m.message });
  }
  clearMarker();
  drawMemory();
  emit("ocr:error", `${m.message} — reload the page to start again`);
}

// ── the engine worker ────────────────────────────────────────────────
// one worker for every read (it keeps the engine's memory pool); a new one only after the device check's memory test
// and when the app is back from the background (resetEngine): startWorker sets one up
let worker = startWorker();
// the worker ended and a new one started: after the device check's memory test (its GPU memory goes with the worker,
// before the engine opens in the new one)
function freshWorker() {
  worker.terminate();
  worker = startWorker();
}
/// the engine started anew after the app was away (app.js cameBack): the worker — its wasm, GPU device and memory pool —
/// ended, a new one started (it opens the engine at the next read); what the old one still owed fails, the engine's death
/// is forgotten
export function resetEngine() {
  for (const w of waiting.values()) {
    w.reject("the engine was restarted (the app was in the background)");
  }
  waiting.clear();
  freshWorker();
  gpuDead = null;
  wasAway = false;
  clearMarker();
  drawMemory();
}
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
    // one worker, one GPU device: the engine keeps its memory pools from read to read (ocr-wgpu gpu.rs Pool) — nothing
    // freed and allocated anew between or inside reads (iOS kept counting freed GPU memory against the page)
    work("read", { source: modelSource(), rgb: b.rgb, w: b.w, h: b.h, preset: p, seed: b.seed }, [b.rgb.buffer]).then(
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
let lastRead = null; // the worker's stats of the last finished read (ReadStats + loadMs, load, idleBytes)
function drawMemory() {
  const parts = [];
  if (engine) {
    if (lastRead) {
      parts.push(`GPU memory held by the engine: ${mb(lastRead.poolBytes)} (its memory pool, kept from read to read; each read loads its weights into it)`);
    } else {
      parts.push(`GPU memory: ${mb(engine.idleBytes)} before the first read`);
    }
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

/// the model check: both stored files' MD5 against the release (config.js MODEL.md5) → { ok, damaged: [names], text }
export async function verifyModel() {
  const got = await work("verifyModel");
  const damaged = Object.keys(MODEL.md5).filter((name) => got[name] !== MODEL.md5[name]);
  return damaged.length === 0
    ? { ok: true, damaged, text: "The stored model is intact (both files match the release)." }
    : { ok: false, damaged, text: `The stored model is damaged: ${damaged.join(", ")} differs from the release. Delete the model (Settings) and download it again.` };
}
$("modelVerify").onclick = async () => {
  $("settingsErr").textContent = "";
  $("modelVerify").disabled = true;
  $("modelProgressText").textContent = "checking the stored model…";
  try {
    const r = await verifyModel();
    $("modelProgressText").textContent = r.text;
  } catch (e) {
    $("settingsErr").textContent = String(e);
    $("modelProgressText").textContent = "";
  } finally {
    $("modelVerify").disabled = false;
  }
};

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
    parts.push(`GPU pool ${mb(lastRead.poolBytes)}`, `peak in use ${mb(lastRead.gpuMemory.peakBytes)} (${lastRead.gpuMemory.peakPhase})`);
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
// the device check's verdicts, worst to best (a retry keeps the better); facts: the browser has WebGPU at all (no
// retry without it — nothing to warm up)
const RANK = { reject: 0, pending: 1, slow: 2, ok: 3 };
const facts = () => NO_WEBGPU === null;
// the device check's memory test (iPhone / iPad): what the engine holds on the GPU from its first read on — the fixed
// memory pool (ocr-wgpu gpu.rs Pool::fix: 24 × 64 MiB + the largest tensor's 157.6 MiB block = 1693.6 MiB) + the 32 MiB
// upload buffer + the 32 MiB readback ≈ 1758 MiB (GPU_NEED "about 1.8 GB"); the crash marker's step while it runs
const MEMORY_TEST_BYTES = 1760 * 2 ** 20;
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
    // a first check right after a fresh install often failed or came out slow (the browser's GPU process and the test
    // shaders cold); a second one, all anew (adapter, device), passed (user): one more run by itself, the better counts
    let r = await runGpuCheck(navigator.gpu, env, onStep);
    if (r.verdict !== "ok" && facts(r)) {
      await new Promise((res) => setTimeout(res, 500));
      const again = await runGpuCheck(navigator.gpu, env, onStep);
      if (RANK[again.verdict] >= RANK[r.verdict]) {
        r = again;
      }
    }
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
        freshWorker();
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
