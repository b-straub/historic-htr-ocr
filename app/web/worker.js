// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// OCR Lite's engine worker (browser build): ocr-web's wasm engine on WebGPU + the model files.
// The model is never held in JS or wasm memory as a whole: the engine asks for byte ranges (TensorSource::read_at) and
// the reader here fills the engine's staging chunk in place —
//   OPFS  a FileSystemSyncAccessHandle (dedicated workers only) read straight into the view on wasm memory
//   HTTP  synchronous Range requests (allowed in workers), one per chunk — the selftest's path, and later the stream
//         from a model URL without storing it
// OPFS I/O is plain JS (far simpler than Rust bindings); Rust only parses and uploads.
// Weights per read (Safari bills a page's WebGPU memory to it): the engine (device + kernels + tokenizer) is opened
// once per model source and kept; every read opens the files again, the engine loads the weights, reads and frees them
// (a failed or cancelled read too), the files are closed — between reads no weights are on the GPU and no OPFS handle
// is open.
// Messages in: { id, cmd, … } — reply { id, ok, result | error }; events out: { event: "status" | "phase" | "text" |
// "progress" | "step" | "gpuFatal", … } (phase: ocr-decode's Phase — { phase: "encodingImage" | "loadingText" | "readingPrompt" |
// "decoding", loadMs, encodeMs?, prefillMs? }; step: { step: "model load" } as each read's weight upload starts — the
// page's crash marker, bridge-web.js; gpuFatal: { kind, message } — the GPU device was lost or a GPU error went
// uncaught: the engine is dead, every further read fails with that message).
// The model's source record: OPFS root model-source.json { source: "hub" | "upload" | "url", url?, files: { <name>:
// source } } — rewritten whenever a model file is stored or deleted, removed with the last model file.
// The sources allowed (web/config.js MODEL_SOURCES, the page's first message { cmd: "sources", sources }): upload /
// download / an HTTP model refused for a source not listed — and all of them before the page said.
import init, * as ocr from "./pkg/ocr_web.js";

const MODEL_FILES = ["text-q8_0.gguf", "mmproj-bf16.gguf"];
const DIR = "models";
const SOURCE = "model-source.json"; // in the OPFS root, beside DIR (DIR holds only the model files)
let opened = null; // the source key of the open engine ("opfs" | "http:<base>")
let sources = null; // the model sources the page allows ({ cmd: "sources" }); null: not told yet — every source refused

/// a model source the page has not allowed → an error
function allowSource(source) {
  if (sources === null || !sources.includes(source)) {
    throw new Error(`the model source "${source}" is switched off (allowed: ${sources === null ? "none yet" : sources.join(", ")})`);
  }
}

const post = (m) => self.postMessage(m);
const status = (text) => post({ event: "status", text });
// ocr-wgpu's fatal GPU events (ocr-web calls this global): at once to the page, a read running or not
self.ocrGpuFatal = (kind, message) => post({ event: "gpuFatal", kind, message });

async function modelDir() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(DIR, { create: true });
}

async function opfsList() {
  const dir = await modelDir();
  const out = [];
  for await (const [name, h] of dir.entries()) {
    if (h.kind === "file") out.push({ name, size: (await h.getFile()).size });
  }
  return out.sort((a, b) => a.name.localeCompare(b.name));
}

function closeEngine() {
  ocr.close_engine();
  opened = null;
}

async function opfsDelete(name) {
  if (opened === "opfs") closeEngine();
  await (await modelDir()).removeEntry(name);
  await recordDelete();
}

/// the model files present in OPFS (names only)
async function modelNames() {
  return (await opfsList()).map((f) => f.name).filter((n) => MODEL_FILES.includes(n));
}

/// the stored source record, or null when there is none
async function readSource() {
  const root = await navigator.storage.getDirectory();
  let fh;
  try {
    fh = await root.getFileHandle(SOURCE);
  } catch (e) {
    if (e.name === "NotFoundError") {
      return null;
    }
    throw e;
  }
  return JSON.parse(await (await fh.getFile()).text());
}

async function writeSource(record) {
  const root = await navigator.storage.getDirectory();
  const h = await (await root.getFileHandle(SOURCE, { create: true })).createSyncAccessHandle();
  try {
    const b = new TextEncoder().encode(JSON.stringify(record));
    h.truncate(0);
    h.write(b, { at: 0 });
    h.flush();
  } finally {
    h.close();
  }
}

async function removeSource() {
  const root = await navigator.storage.getDirectory();
  try {
    await root.removeEntry(SOURCE);
  } catch (e) {
    if (e.name !== "NotFoundError") {
      throw e;
    }
  }
}

/// the record that governs the model in OPFS: the stored one; a model WITHOUT a record counts as "unknown" (no source
/// allows it: the setup downloads the model again). null: no model file at all.
async function effectiveSource() {
  const names = await modelNames();
  if (!names.length) {
    return { record: null, names };
  }
  const stored = await readSource();
  return { record: stored ?? { source: "unknown" }, names };
}

// where one stored file came from (a record without `files`: the record's own source)
const fileSource = (record, name) => (record.files && name in record.files ? record.files[name] : record.source);

/// after the model file `name` was stored from `next` ({ source: "hub" } | { source: "upload" } |
/// { source: "url", url }): the record rewritten — after each file, so a store that fails half way leaves a record
/// for what is there
async function recordStore(name, next) {
  const { record: prev, names } = await effectiveSource();
  const files = {};
  for (const n of names) {
    files[n] = n === name ? next.source : fileSource(prev, n);
  }
  await writeSource({ ...next, files });
}

/// after a model file was deleted: the record for what is left (none left: no record)
async function recordDelete() {
  const { record: prev, names } = await effectiveSource();
  if (!names.length) {
    await removeSource();
    return;
  }
  const files = {};
  for (const n of names) {
    files[n] = fileSource(prev, n);
  }
  await writeSource({ ...prev, files });
}

/// write a byte stream into OPFS as `name` (removed again if anything fails); progress events per MB
async function opfsWrite(name, stream, total, label) {
  if (!MODEL_FILES.includes(name)) throw new Error(`${name}: the model files are ${MODEL_FILES.join(" + ")}`);
  if (opened === "opfs") closeEngine();
  const dir = await modelDir();
  const h = await (await dir.getFileHandle(name, { create: true })).createSyncAccessHandle();
  let at = 0, shown = 0;
  try {
    h.truncate(0);
    const reader = stream.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      let off = 0;
      while (off < value.length) off += h.write(value.subarray(off), { at: at + off });
      at += value.length;
      if (at - shown >= 1 << 20) { shown = at; post({ event: "progress", label, name, done: at, total }); }
    }
    if (total && at !== total) throw new Error(`${name}: ${at} of ${total} bytes arrived`);
    h.flush();
  } catch (e) {
    h.close();
    await dir.removeEntry(name);
    throw e;
  }
  h.close();
  post({ event: "progress", label, name, done: at, total: at });
  return at;
}

let downloadAbort = null; // the running download's AbortController (the setup's Cancel: { cmd: "cancelDownload" })

async function download(base, name, signal, remoteName) {
  const url = `${base.replace(/\/$/, "")}/${remoteName}`;
  const r = await fetch(url, { signal });
  // a Hugging Face repo that is gone or not public answers 401 / 404 — say so in words instead of "HTTP 401"
  if (!r.ok && (r.status === 401 || r.status === 404) && new URL(url).hostname === "huggingface.co") {
    throw new Error(`The model is not available on Hugging Face right now — please try again later. (${name}: HTTP ${r.status})`);
  }
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = Number(r.headers.get("Content-Length")) || 0;
  return opfsWrite(name, r.body, total, "download");
}

/// a reader for the engine: (offset, view) → bytes written into the view; counts the model bytes for the progress bar
function counted(read, name, total, state) {
  return (off, view) => {
    const n = read(off, view);
    state.done += n;
    if (state.done - state.shown >= 32 << 20) {
      state.shown = state.done;
      post({ event: "progress", label: "load", name, done: state.done, total });
    }
    return n;
  };
}

function httpReader(url) {
  return (off, view) => {
    const x = new XMLHttpRequest();
    x.open("GET", url, false);
    x.responseType = "arraybuffer";
    x.setRequestHeader("Range", `bytes=${off}-${off + view.length - 1}`);
    x.send();
    if (x.status !== 206) throw new Error(`${url}: HTTP ${x.status} for a range request (the server must answer 206)`);
    const b = new Uint8Array(x.response);
    view.set(b);
    return b.length;
  };
}

async function httpSize(url) {
  const r = await fetch(url, { method: "HEAD" });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const n = Number(r.headers.get("Content-Length"));
  if (!n) throw new Error(`${url}: no Content-Length`);
  return n;
}

const sourceKey = (src) => (src.kind === "opfs" ? "opfs" : `http:${src.base}`);

/// the model files of `src` opened: [{ name, size, read }] + close() (the OPFS sync access handles — held only while
/// the engine opens or a read loads its weights)
async function openFiles(src) {
  const files = [];
  const handles = [];
  const close = () => { for (const h of handles) h.close(); };
  try {
    if (src.kind === "opfs") {
      const dir = await modelDir();
      for (const name of MODEL_FILES) {
        let fh;
        try { fh = await dir.getFileHandle(name); } catch { throw new Error(`${name} is not in this browser's storage — download the model first (Help → Run setup again, or Settings → Model)`); }
        const h = await fh.createSyncAccessHandle();
        handles.push(h);
        files.push({ name, size: h.getSize(), read: (off, view) => h.read(view, { at: off }) });
      }
    } else {
      for (const name of MODEL_FILES) {
        const url = `${src.base.replace(/\/$/, "")}/${name}`;
        files.push({ name, size: await httpSize(url), read: httpReader(url) });
      }
    }
  } catch (e) {
    close();
    throw e;
  }
  return { files, close };
}

/// open the engine for the model of `src` (OPFS { kind: "opfs" } or HTTP ranges { kind: "http", base }) unless it is
/// open already: the device + kernels + tokenizer, no weights → its info (null: already open)
async function ensureEngine(src) {
  if (src.kind === "http") {
    allowSource("stream");
  }
  const key = sourceKey(src);
  if (opened === key && ocr.engine_open()) return null;
  closeEngine();
  const t0 = performance.now();
  const { files, close } = await openFiles(src);
  try {
    const [t, m] = files;
    const info = JSON.parse(await ocr.open_engine(t.size, t.read, m.size, m.read));
    info.openMs = performance.now() - t0;
    info.source = key;
    opened = key;
    return info;
  } finally {
    close();
  }
}

/// `run(readFiles)` with the model files of `src` opened (and closed after it); `step` = the crash marker's step
async function withFiles(src, step, run) {
  const { files, close } = await openFiles(src);
  try {
    const total = files.reduce((a, f) => a + f.size, 0);
    const state = { done: 0, shown: 0 };
    post({ event: "step", step });
    const [t, m] = files;
    return await run(new ocr.ReadFiles(t.size, counted(t.read, t.name, total, state), m.size, counted(m.read, m.name, total, state)));
  } finally {
    close();
  }
}

/// one read: the files opened, the weights loaded by the engine for this read and freed after it, the files closed
async function readOnce(src, { rgb, w, h, preset, seed }) {
  const { files, close } = await openFiles(src);
  try {
    const total = files.reduce((a, f) => a + f.size, 0);
    const state = { done: 0, shown: 0 };
    status(`loading the model (${(total / 1e9).toFixed(2)} GB) from ${src.kind === "opfs" ? "browser storage" : src.base}`);
    post({ event: "step", step: "model load" });
    const [t, m] = files;
    const rf = new ocr.ReadFiles(t.size, counted(t.read, t.name, total, state), m.size, counted(m.read, m.name, total, state));
    const json = await ocr.read(rf, rgb, w, h, JSON.stringify(preset), seed, (phase) => post({ event: "phase", phase: JSON.parse(phase) }), (piece) => post({ event: "text", text: piece }));
    return JSON.parse(json);
  } finally {
    close();
  }
}

/// the device check's memory test (iPhone / iPad): `bytes` of GPU memory allocated in blocks and written (a shader
/// writes every word — untouched memory may not count), held until the GPU is done → { bytes, blocks, ms }. Run in this
/// worker, which the page ends after it: iOS keeps counting freed GPU memory against its process; a device
/// short of memory is killed here, before the model's download (the page's crash marker says so at the next start)
async function memoryTest(bytes) {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) {
    throw new Error("memory test: no GPU adapter");
  }
  const device = await adapter.requestDevice({
    requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize },
  });
  const t0 = performance.now();
  const block = Math.min(256 << 20, adapter.limits.maxBufferSize, adapter.limits.maxStorageBufferBindingSize);
  const module = device.createShaderModule({
    code: `@group(0) @binding(0) var<storage, read_write> b: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = arrayLength(&b);
  for (var i = id.x; i < n; i += 65535u * 256u) { b[i] = i | 1u; }
}`,
  });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bufs = [];
  try {
    for (let done = 0; done < bytes; done += block) {
      const size = Math.min(block, bytes - done);
      const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
      bufs.push(buf);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] }));
      pass.dispatchWorkgroups(65535);
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    return { bytes, blocks: bufs.length, ms: performance.now() - t0 };
  } finally {
    for (const b of bufs) {
      b.destroy();
    }
    device.destroy();
  }
}

async function memory() {
  const m = { wasmBytes: ocr.wasm_memory_bytes() };
  if (performance.memory) m.jsHeapUsed = performance.memory.usedJSHeapSize;
  return m;
}

const handlers = {
  async list() { return opfsList(); },
  async delete({ name }) { return opfsDelete(name); },
  async upload({ files }) {
    allowSource("upload");
    const out = [];
    for (const f of files) {
      out.push({ name: f.name, size: await opfsWrite(f.name, f.stream(), f.size, "upload") });
      await recordStore(f.name, { source: "upload" });
    }
    return out;
  },
  // record: { source: "hub" } for the model's Hugging Face repo, else { source: "url", url } — decided by the page
  // (bridge-web.js downloadModel); remote: the repo's file names when they differ (web/config.js MODEL.remoteFiles)
  // a cancelled download removes the file it was writing (opfsWrite); a file stored before it stays
  async download({ base, record, remote }) {
    if (!record) {
      throw new Error("download: no source record (bridge-web.js downloadModel decides it)");
    }
    allowSource(record.source);
    const out = [];
    downloadAbort = new AbortController();
    try {
      for (const name of MODEL_FILES) {
        out.push({ name, size: await download(base, name, downloadAbort.signal, remote ? remote[name] : name) });
        await recordStore(name, record);
      }
    } catch (e) {
      if (downloadAbort.signal.aborted) {
        throw new Error("download cancelled");
      }
      throw e;
    } finally {
      downloadAbort = null;
    }
    return out;
  },
  async source() { return effectiveSource(); },
  async load({ source }) { return ensureEngine(source); },
  async memory() { return memory(); },
  async memoryTest({ bytes }) { return memoryTest(bytes); },
  // the lost-device selftest (selftest.html?lose=1): destroy the engine's GPU device, also during a read
  async debugLoseDevice() { ocr.debug_destroy_device(); },
  async read({ source, rgb, w, h, preset, seed }) {
    ocr.reset_cancel(); // queued: a Stop from now on counts (also during the model load)
    const info = await ensureEngine(source);
    if (info) post({ event: "loaded", info });
    return readOnce(source, { rgb, w, h, preset, seed });
  },
  // the staged read split over two workers (bridge-web.js): this worker encodes and is ended after it; a new one
  // reads on from the embeddings — what a dead worker held cannot stay counted against the second half
  async encode({ source, rgb, w, h }) {
    const info = await ensureEngine(source);
    if (info) post({ event: "loaded", info });
    return withFiles(source, "model load (vision)", async (rf) => {
      const r = await ocr.encode(rf, rgb, w, h, (phase) => post({ event: "phase", phase: JSON.parse(phase) }));
      return { emb: r.emb, stats: JSON.parse(r.stats) };
    });
  },
  async readEncoded({ source, emb, encodeMs, preset, seed }) {
    ocr.reset_cancel();
    post({ event: "step", step: "engine restart" });
    const t0 = performance.now();
    const info = await ensureEngine(source);
    if (info) post({ event: "loaded", info });
    const restartMs = performance.now() - t0;
    const stats = await withFiles(source, "model load (text)", async (rf) =>
      JSON.parse(await ocr.read_encoded(rf, emb, encodeMs, JSON.stringify(preset), seed, (phase) => post({ event: "phase", phase: JSON.parse(phase) }), (piece) => post({ event: "text", text: piece }))));
    return { ...stats, restartMs };
  },
};

const ready = init();
self.onmessage = async (e) => {
  const { id, cmd, ...args } = e.data;
  if (cmd === "cancel") { await ready; ocr.cancel(); return; }
  if (cmd === "sources") {
    sources = args.sources;
    return;
  }
  if (cmd === "cancelDownload") {
    if (downloadAbort) {
      downloadAbort.abort();
    }
    return;
  }
  try {
    await ready;
    const h = handlers[cmd];
    if (!h) throw new Error(`worker: no command ${cmd}`);
    post({ id, ok: true, result: await h(args) });
  } catch (err) {
    post({ id, ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};
