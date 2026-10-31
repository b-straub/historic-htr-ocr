// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The device check of the browser build — WebGPU tested in advance, rejected or shown with hints (an RTX 2070 notebook
// read at 7 tok/s in Chrome, most likely on its integrated GPU; the M3 decodes ≈ 84 tok/s): the setup's
// first step (ui/setup.js) and Help → Check this device. Plain WebGPU on the page, no model:
//   1  navigator.gpu        missing → rejected, with the hint for this browser (HTTPS, Safari 26, Firefox, Linux flags)
//   2  the adapter          requestAdapter({ powerPreference: "high-performance" }) like the engine (ocr-wgpu gpu.rs
//                           Gpu::create); none → rejected; a fallback (software) adapter → rejected: far too slow
//   3  limits + shader-f16  what the engine needs (NEEDS below, from the code); short → rejected with the numbers;
//                           no shader-f16, or too little workgroup memory for the f16 kernels → runs (the f32 path,
//                           computePath), said: slower, more GPU memory
//   4  micro-benchmark      ≈ 1 s: an int8 matvec over 96 MiB (memory-bound like the decode, its q8_0 matvecs) → GB/s,
//                           and the cost of a small dependent dispatch (the decode runs ≈ 200 per token) → the decode's
//                           tokens/s ESTIMATED against the same benchmark on the M3 (REF below)
//   5  hints                an integrated GPU on Windows / Linux, a slow result, an Intel Mac: how to switch to the
//                           discrete GPU
// judge() is pure (unit-tested in Node: tests/gpu_check.mjs); runGpuCheck() gathers the facts in the browser. The
// texts: env.t / t = a translator of ui/i18n.js (the page passes its current one; the Node test translator("en")).
// Nothing here gates a read: the read path keeps its own checks (the engine's limits pre-check, the crash marker).

/// what the engine needs from an adapter — the numbers from the code (ocr-wgpu requests the adapter's own limits)
export const NEEDS = {
  // the largest weight bound whole to a kernel: token_embd.weight, q8_0 151936 × 1024 = 165 306 368 bytes
  // (engine.rs: largest_tensor → Gpu::check_fits against both limits, before anything is uploaded)
  maxBufferSize: 165306368,
  maxStorageBufferBindingSize: 165306368,
  // workgroup memory: attention_tile.wgsl (the vision attention on the f16 path) holds Q 8192 + K 2048 + V 2048 +
  // S 4096 + Red 512 + Cr 256 = 17 152 bytes — over WebGPU's 16 384 default (less → the engine takes the f32 path,
  // gpu.rs Compute::choose_caps, which counts it from the kernels; its test compute_choice_matches_gpu_check_js keeps
  // this number equal); the f32 path's largest is matmul128.wgsl with f32 tiles: 2 × 512 × 16 = 16 384
  workgroupStorageF16: 17152,
  workgroupStorageF32: 16384,
  // @workgroup_size(256, 1, 1) (the decode matvecs, norms, choose_*) and (16, 16, 1) (matmul128)
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 16,
  // dmv_qkv.wgsl / qkv_post.wgsl bind 6 storage buffers
  maxStorageBuffersPerShaderStage: 6,
};
/// a full-size SQUARE page (1540 × 1540 = 110 × 110 patches) binds the vision FFN's gate / up products whole:
/// 12 100 × 4096 f32 = 198 246 400 bytes (vision.rs run / run_f16 — the F16 browser path keeps them f32); a portrait
/// page (980 × 1540) 126 MB. Below this the check passes with a note: the read names the limit (gpu.rs Batch).
export const SQUARE_PAGE_BINDING = 198246400;

/// the reference: this benchmark on the M3 (10-core GPU, 24 GB) in headless Chrome 155, measured at normal
/// priority (5 runs of benchmark(): 93.1 · 93.3 · 90.7 · 91.4 · 90.6 GB/s, 1.29 · 1.20 · 1.13 · 1.14 · 1.19 µs a
/// dispatch — the medians here; the LM head's own matvec there ≈ 94 GB/s), paired with Chrome's decode there (a full
/// 980 × 1540 page: 84.2 tok/s, 11.9 ms a token)
export const REF = { gbps: 91.4, dispatchUs: 1.2, tokS: 84 };
/// dispatches per decoded token: 28 layers × 7 on the f16 paths (text.rs, the fused decode step) + embed, norm, LM head,
/// the three token-choice kernels; the f32 path keeps 18 per layer
export const DECODE_DISPATCHES = { f16: 28 * 7 + 6, f32: 28 * 18 + 6 };
/// est. tokens/s → class: ok ≥ 25 · slow ≥ 8 · very slow below (a 900-token page: ≤ 36 s · ≤ 2 min · longer)
export const CLASS = { ok: 25, slow: 8 };
/// estimates beyond this are shown as "150+": the reference is one Apple GPU, a far faster GPU is not extrapolated
export const EST_CAP = 150;

// ── the environment ──────────────────────────────────────────────────

/// the operating system from the user agent (iPadOS Safari says "Macintosh": a Mac with a touch screen is an iPad)
/// → "windows" | "mac" | "ipad" | "iphone" | "android" | "chromeos" | "linux" | "other"
export function osOf(ua, maxTouchPoints) {
  if (/iPhone|iPod/.test(ua)) {
    return "iphone";
  }
  if (/iPad/.test(ua) || (/Macintosh/.test(ua) && maxTouchPoints > 1)) {
    return "ipad";
  }
  if (/Windows NT/.test(ua)) {
    return "windows";
  }
  if (/Android/.test(ua)) {
    return "android";
  }
  if (/CrOS/.test(ua)) {
    return "chromeos";
  }
  if (/Macintosh/.test(ua)) {
    return "mac";
  }
  if (/Linux|X11/.test(ua)) {
    return "linux";
  }
  return "other";
}

/// the browser and its major version → { name: "edge" | "chrome" | "firefox" | "safari" | "other", major }; every
/// browser on iPadOS / iOS is WebKit underneath: "safari" with the system's version
export function browserOf(ua) {
  const m = (re) => {
    const r = re.exec(ua);
    return r ? Number(r[1]) : 0;
  };
  if (/iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && /Version\/[\d.]+.*Safari\//.test(ua) && !/Chrome\//.test(ua))) {
    const os = /OS (\d+)_/.exec(ua);
    const v = m(/Version\/(\d+)/);
    return { name: "safari", major: v > 0 ? v : os ? Number(os[1]) : 0 };
  }
  if (/Firefox\//.test(ua)) {
    return { name: "firefox", major: m(/Firefox\/(\d+)/) };
  }
  if (/Edg\//.test(ua)) {
    return { name: "edge", major: m(/Edg\/(\d+)/) };
  }
  if (/OPR\/|SamsungBrowser\//.test(ua)) {
    return { name: "other", major: 0 };
  }
  if (/Chrome\//.test(ua)) {
    return { name: "chrome", major: m(/Chrome\/(\d+)/) };
  }
  return { name: "other", major: 0 };
}

// ── the adapter ──────────────────────────────────────────────────────

/// the adapter's words in one lower-case string (vendor, architecture, device, description — Chrome leaves device and
/// description empty unless its WebGPU developer features are on)
function words(info) {
  return [info.vendor, info.architecture, info.device, info.description].filter((s) => typeof s === "string" && s !== "").join(" ").toLowerCase();
}

/// a short name for the adapter, from what it reports: "intel · gen-12lp", "apple · metal-3", "NVIDIA GeForce …"
export function adapterName(info, t) {
  if (typeof info.description === "string" && info.description !== "") {
    return info.description;
  }
  const parts = [info.vendor, info.architecture].filter((s) => typeof s === "string" && s !== "");
  return parts.length ? parts.join(" · ") : t("g.unnamed");
}

/// software rendering (the adapter says it is a fallback, or names a CPU rasterizer)
export function isSoftware(info, fallbackFlag) {
  return fallbackFlag === true || info.isFallbackAdapter === true || /swiftshader|llvmpipe|lavapipe|softpipe|microsoft basic render/.test(words(info));
}

/// integrated or discrete, from the adapter's words → "apple" | "integrated" | "discrete" | "unknown"
///   apple      Apple silicon (one GPU — nothing to switch)
///   integrated Intel except Arc (architecture xe-hpg / alchemist / battlemage or "Arc" in the name); AMD when the name
///              is the APU's ("Radeon(TM) Graphics", "Radeon Vega 8 Graphics", "Radeon 780M") or the architecture an
///              APU's; Qualcomm / ARM / Imagination (one GPU on the chip)
///   discrete   NVIDIA; Intel Arc; AMD with a card's name (RX …, Radeon Pro …)
///   unknown    AMD without a name (Chrome hides it): could be either
export function adapterKind(info) {
  const w = words(info);
  const vendor = (info.vendor ?? "").toLowerCase();
  if (vendor === "apple" || /apple/.test(w)) {
    return "apple";
  }
  if (vendor === "nvidia" || /nvidia|geforce|quadro|rtx/.test(w)) {
    return "discrete";
  }
  if (vendor === "intel" || /\bintel\b/.test(w)) {
    return /\barc\b|xe-hpg|alchemist|battlemage/.test(w) ? "discrete" : "integrated";
  }
  if (vendor === "amd" || /\bamd\b|radeon|\bati\b/.test(w)) {
    if (/\brx\s?\d|radeon pro|firepro|instinct/.test(w)) {
      return "discrete";
    }
    if (/radeon\(tm\) graphics|radeon graphics|vega \d+ graphics|radeon \d{3}m\b|gcn-5|vega|rdna-3\.5/.test(w) && !/\brx\b/.test(w)) {
      return "integrated";
    }
    return "unknown";
  }
  if (/qualcomm|adreno|\barm\b|mali|imagination|powervr/.test(w)) {
    return "integrated";
  }
  return "unknown";
}

/// the path the engine computes on (ocr-wgpu gpu.rs Compute::choose_caps) → "f16" | "f32" (no shader-f16) |
/// "f32-workgroup" (shader-f16, but less workgroup memory than the f16 kernels need: the f32 path instead)
export function computePath(limits, f16) {
  if (!f16) {
    return "f32";
  }
  const have = limits.maxComputeWorkgroupStorageSize;
  if (typeof have !== "number") {
    throw new Error("the adapter reports no maxComputeWorkgroupStorageSize");
  }
  return have >= NEEDS.workgroupStorageF16 ? "f16" : "f32-workgroup";
}

/// the adapter's limits against NEEDS for its compute path (computePath) → { fails: [{ name, have, need }], notes: [text] }
export function checkLimits(limits, f16, t) {
  const need = [
    ["maxBufferSize", NEEDS.maxBufferSize],
    ["maxStorageBufferBindingSize", NEEDS.maxStorageBufferBindingSize],
    ["maxComputeWorkgroupStorageSize", computePath(limits, f16) === "f16" ? NEEDS.workgroupStorageF16 : NEEDS.workgroupStorageF32],
    ["maxComputeInvocationsPerWorkgroup", NEEDS.maxComputeInvocationsPerWorkgroup],
    ["maxComputeWorkgroupSizeX", NEEDS.maxComputeWorkgroupSizeX],
    ["maxComputeWorkgroupSizeY", NEEDS.maxComputeWorkgroupSizeY],
    ["maxStorageBuffersPerShaderStage", NEEDS.maxStorageBuffersPerShaderStage],
  ];
  const fails = [];
  for (const [name, n] of need) {
    const have = limits[name];
    if (typeof have !== "number") {
      throw new Error(`the adapter reports no ${name}`);
    }
    if (have < n) {
      fails.push({ name, have, need: n });
    }
  }
  const notes = [];
  if (fails.length === 0 && limits.maxStorageBufferBindingSize < SQUARE_PAGE_BINDING) {
    notes.push(t("g.squareNote", { have: mb(limits.maxStorageBufferBindingSize, t), need: mb(SQUARE_PAGE_BINDING, t) }));
  }
  return { fails, notes };
}

const mb = (b, t) => `${t.fixed(b / 1048576, 1)} MB`;
/// a limit's value for the user: buffer sizes in MB, workgroup memory in bytes, counts as they are
function limitText(name, v, t) {
  if (name === "maxBufferSize" || name === "maxStorageBufferBindingSize") {
    return mb(v, t);
  }
  if (name === "maxComputeWorkgroupStorageSize") {
    return t("g.bytes", { n: v });
  }
  return String(v);
}

// ── the speed estimate ───────────────────────────────────────────────

/// decode tokens/s estimated from this device's benchmark against the reference's: the reference's time per token
/// split into its dispatch part (DECODE_DISPATCHES × its dispatch cost) and the rest — the rest scaled by the bandwidth
/// ratio (the decode streams the q8_0 weights + the KV cache each token), the dispatch part replaced by this device's
/// own. f16 = the path the engine will take (no shader-f16: the f32 path, ≈ 2.5× the dispatches).
export function estimateTokS(bench, f16, ref = REF) {
  if (!(bench.gbps > 0) || !(ref.gbps > 0)) {
    throw new Error("estimateTokS: a benchmark without bandwidth");
  }
  const refTok = 1 / ref.tokS;
  const refDispatch = (DECODE_DISPATCHES.f16 * ref.dispatchUs) / 1e6;
  if (refDispatch >= refTok) {
    throw new Error("estimateTokS: the reference's dispatch cost exceeds its time per token");
  }
  const rest = (refTok - refDispatch) * (ref.gbps / bench.gbps);
  const dispatch = ((f16 ? DECODE_DISPATCHES.f16 : DECODE_DISPATCHES.f32) * bench.dispatchUs) / 1e6;
  return 1 / (rest + dispatch);
}

/// est. tokens/s → "ok" | "slow" | "very slow"
export function speedClass(tokS) {
  if (tokS >= CLASS.ok) {
    return "ok";
  }
  if (tokS >= CLASS.slow) {
    return "slow";
  }
  return "very slow";
}

/// "~42" / "150+"
export function tokSText(tokS) {
  return tokS > EST_CAP ? `${EST_CAP}+` : `~${Math.round(tokS)}`;
}

// ── hints ────────────────────────────────────────────────────────────

/// why there is no navigator.gpu, for this browser → [{ title, text }]
export function noWebGpuHints(env) {
  const t = env.t;
  const os = osOf(env.ua, env.maxTouchPoints);
  const b = browserOf(env.ua);
  if (!env.secure) {
    return [{ title: t("g.insecure.title"), text: t("g.insecure") }];
  }
  if (b.name === "safari") {
    const where = t(os === "mac" ? "g.where.mac" : "g.where.ios");
    if (b.major < 26) {
      return [{ title: `Safari ${b.major > 0 ? b.major : t("g.safari.old")}`, text: `${t("g.safari.update", { where })}${t(os === "mac" ? "g.safari.orChrome" : "g.safari.engine")}` }];
    }
    return [{ title: `Safari ${b.major}`, text: t(os === "mac" ? "g.safari.offMac" : "g.safari.offIos") }];
  }
  if (b.name === "firefox") {
    return [{ title: `Firefox ${b.major}`, text: t("g.firefox") }];
  }
  if (b.name === "chrome" || b.name === "edge") {
    const label = b.name === "chrome" ? "Chrome" : "Edge";
    const flags = b.name === "chrome" ? "chrome://flags" : "edge://flags";
    if (b.major > 0 && b.major < 113) {
      return [{ title: `${label} ${b.major}`, text: t("g.chrome.update", { label }) }];
    }
    if (os === "linux") {
      return [{ title: t("g.linux.title", { label }), text: t("g.linux", { flags, label }) }];
    }
    if (os === "android") {
      return [{ title: t("g.android.title", { label }), text: t("g.android") }];
    }
    return [{ title: `${label} ${b.major}`, text: t("g.blocked", { page: b.name === "chrome" ? "chrome://gpu" : "edge://gpu" }) }];
  }
  return [{ title: t("g.other.title"), text: t("g.other") }];
}

/// how to put the browser on the discrete GPU / keep the GPU fast — for this OS + adapter → [{ title, text }]
export function speedHints(os, kind, vendor, t) {
  const hint = (k) => ({ title: t(`g.${k}.title`), text: t(`g.${k}`) });
  if (os === "windows") {
    return [hint("win"), hint("nv"), hint("power")];
  }
  if (os === "linux") {
    return [hint("linuxGpu")];
  }
  if (os === "mac" && vendor !== "apple" && kind !== "apple") {
    return [hint("intelMac")];
  }
  return [];
}

// ── the verdict ──────────────────────────────────────────────────────

/// the check's verdict from the gathered facts (pure — tests/gpu_check.mjs):
///   facts  { gpu: navigator.gpu present, adapter: null | { info, fallback, f16, limits }, bench: null | { gbps,
///          dispatchUs }, error?: text (the probe failed) }
///   env    { ua, maxTouchPoints, secure, flagged: null | text (the host's unsupported-device notice), t: a translator }
/// → { verdict: "ok" | "slow" | "reject" | "pending" (passed so far — the benchmark is next), cls?, headline,
///     detected: [text], problems: [text], notes: [text], hints: [{ title, text }], adapter?, kind?, estTokS? }
export function judge(facts, env, ref = REF) {
  const t = env.t;
  const os = osOf(env.ua, env.maxTouchPoints);
  const r = { verdict: "reject", headline: "", detected: [], problems: [], notes: [], hints: [] };
  // Firefox is not supported: where it has WebGPU, its timers are too coarse for the benchmark (100 ms steps under its
  // fingerprinting protection) and the model's storage is often refused — rejected before any GPU work
  const browser = browserOf(env.ua);
  if (browser.name === "firefox") {
    r.headline = t("g.firefoxNo");
    r.hints = [{ title: `Firefox ${browser.major}`, text: t("g.firefox") }];
    return r;
  }
  if (os === "iphone") {
    r.notes.push(t("g.iphone"));
  }
  if (!facts.gpu) {
    r.headline = t("g.noWebGpu");
    r.hints = noWebGpuHints(env);
    return r;
  }
  if (facts.error) {
    r.headline = t("g.failed");
    r.problems.push(facts.error);
    return r;
  }
  if (facts.adapter === null) {
    r.headline = t("g.noAdapter");
    r.hints = [{ title: t("g.driver.title"), text: t("g.driver") }];
    if (os === "linux") {
      r.hints.push(noWebGpuHints(env)[0]);
    }
    return r;
  }
  const a = facts.adapter;
  const kind = adapterKind(a.info);
  r.adapter = adapterName(a.info, t);
  r.kind = kind;
  r.detected.push(`${t("g.gpu", { name: r.adapter })}${kind === "integrated" ? t("g.integrated") : kind === "discrete" ? t("g.discrete") : ""}`);
  if (isSoftware(a.info, a.fallback)) {
    r.headline = t("g.software");
    r.hints = [{ title: t("g.accel.title"), text: t("g.accel") }];
    return r;
  }
  const lim = checkLimits(a.limits, a.f16, t);
  if (lim.fails.length) {
    r.headline = t("g.limits", { names: lim.fails.map((f) => f.name).join(", ") });
    for (const f of lim.fails) {
      r.problems.push(t("g.limit", { name: f.name, have: limitText(f.name, f.have, t), need: limitText(f.name, f.need, t) }));
    }
    r.hints = speedHints(os, kind, (a.info.vendor ?? "").toLowerCase(), t);
    return r;
  }
  r.notes.push(...lim.notes);
  const path = computePath(a.limits, a.f16);
  if (path === "f32") {
    r.notes.push(t("g.f32"));
  }
  if (path === "f32-workgroup") {
    r.notes.push(t("g.f32wg"));
  }
  if (env.flagged !== null && os === "iphone") {
    r.headline = t("g.iphoneOom");
    r.problems.push(env.flagged);
    return r;
  }
  if (env.flagged !== null) {
    r.notes.push(t("g.earlier", { text: env.flagged }));
  }
  if (facts.bench === null) {
    r.verdict = "pending";
    r.headline = t("g.measuring");
    return r;
  }
  const est = estimateTokS(facts.bench, path === "f16", ref);
  r.estTokS = est;
  r.cls = speedClass(est);
  r.detected.push(t("g.speed", { est: tokSText(est), gbps: t.fixed(facts.bench.gbps, 0), us: t.fixed(facts.bench.dispatchUs, 1), ref: ref.tokS }));
  const integratedPc = kind === "integrated" && (os === "windows" || os === "linux");
  const intelMac = os === "mac" && kind !== "apple";
  if (r.cls === "ok") {
    r.verdict = "ok";
    r.headline = t("g.ok", { gpu: r.adapter, est: tokSText(est) });
    if (integratedPc || intelMac) {
      r.notes.push(t("g.integratedOk"));
      r.hints = speedHints(os, kind, (a.info.vendor ?? "").toLowerCase(), t);
    }
    return r;
  }
  r.verdict = "slow";
  const page = Math.round(900 / est);
  r.headline = r.cls === "slow"
    ? t("g.slow", { est: tokSText(est), s: page })
    : t("g.verySlow", { est: tokSText(est), min: Math.max(1, Math.round(page / 60)) });
  if (integratedPc || intelMac) {
    r.notes.push(t("g.integratedSlow"));
  }
  r.hints = speedHints(os, kind, (a.info.vendor ?? "").toLowerCase(), t);
  return r;
}

// ── the probe + benchmark (browser) ──────────────────────────────────

// int8 weights, 1024 a row (64 × vec4<u32>), x 1024 f32 — the decode's q8_0 matvec in kind (dmv.wgsl: 8 groups of 32
// lanes, 2 rows a group); each byte unpacked and multiplied once, the row's sum reduced in workgroup memory
const MATVEC = `
@group(0) @binding(0) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<vec2<f32>, 256>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let lane = li & 31u;
  let row0 = (wg.x * 8u + (li >> 5u)) * 2u;
  var acc = vec2<f32>(0.0);
  for (var r = 0u; r < 2u; r++) {
    var s = 0.0;
    for (var j = 0u; j < 2u; j++) {
      let i = lane + 32u * j;
      let w = W[(row0 + r) * 64u + i];
      s += dot(unpack4x8snorm(w.x), X[4u * i]) + dot(unpack4x8snorm(w.y), X[4u * i + 1u]) + dot(unpack4x8snorm(w.z), X[4u * i + 2u]) + dot(unpack4x8snorm(w.w), X[4u * i + 3u]);
    }
    acc[r] = s;
  }
  red[li] = acc;
  workgroupBarrier();
  for (var h = 16u; h > 0u; h >>= 1u) {
    if (lane < h) {
      red[li] += red[li + h];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    Y[row0] = red[li].x;
    Y[row0 + 1u] = red[li].y;
  }
}`;
// the weights' bytes, written on the GPU (no 96 MiB from the CPU): a hash per word
const FILL = `
@group(0) @binding(0) var<storage, read_write> W: array<vec4<u32>>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  W[i] = vec4<u32>(i * 2654435761u, i * 2246822519u + 1u, i * 3266489917u + 2u, i * 668265263u + 3u);
}`;
// one small dependent step: the dispatch cost (each reads what the previous one wrote)
const TICK = `
@group(0) @binding(0) var<storage, read_write> C: array<u32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_index) li: u32) {
  C[li] = C[li] + 1u;
}`;

const BENCH_ROWS = 98304; // × 1024 bytes = 96 MiB: inside WebGPU's default 128 MiB binding, far past any GPU cache
const BENCH_BYTES = BENCH_ROWS * 1024;
const MV_N = 8;
const TICKS = 400;
const WARM_S = 0.4;

/// the benchmark on `device` → { gbps, dispatchUs }: warm-up, then pairs of submissions — MV_N matvecs and 1 — until
/// ≈ 0.8 s are spent (at least 3 pairs); bandwidth = (MV_N − 1) × the bytes / (the fastest MV_N − the fastest 1): the
/// difference takes out the submit → done latency, the fastest of each because a GPU shared with other work only ever
/// runs slower. The dispatch cost the same way: TICKS dependent small dispatches against 1, 3 pairs.
export async function benchmark(device) {
  const q = device.queue;
  const w = device.createBuffer({ size: BENCH_BYTES, usage: GPUBufferUsage.STORAGE });
  const x = device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const y = device.createBuffer({ size: BENCH_ROWS * 4, usage: GPUBufferUsage.STORAGE });
  const c = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE });
  try {
    q.writeBuffer(x, 0, new Float32Array(1024).map((_, i) => ((i % 17) - 8) / 8));
    const pipe = (code) => device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
    const [fill, matvec, tick] = await Promise.all([pipe(FILL), pipe(MATVEC), pipe(TICK)]);
    const bind = (p, bufs) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const bgFill = bind(fill, [w]);
    const bgMv = bind(matvec, [w, x, y]);
    const bgTick = bind(tick, [c]);
    const run = async (n, pipeline, bg, groups) => {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      for (let i = 0; i < n; i++) {
        pass.dispatchWorkgroups(groups);
      }
      pass.end();
      const t0 = performance.now();
      q.submit([enc.finish()]);
      await q.onSubmittedWorkDone();
      return (performance.now() - t0) / 1000;
    };
    await run(1, fill, bgFill, BENCH_BYTES / 16 / 256);
    // warm-up ≥ WARM_S of untimed matvecs: a GPU idling at low clocks (and the browser's first submissions) made the
    // first check after opening fail or read slow — "Check again" then passed
    for (let warm = 0; warm < WARM_S;) {
      warm += await run(MV_N, matvec, bgMv, BENCH_ROWS / 16);
    }
    const mv = { one: Infinity, many: Infinity };
    let spent = 0;
    for (let i = 0; spent < 0.8 || i < 3; i++) {
      const one = await run(1, matvec, bgMv, BENCH_ROWS / 16);
      const many = await run(MV_N, matvec, bgMv, BENCH_ROWS / 16);
      mv.one = Math.min(mv.one, one);
      mv.many = Math.min(mv.many, many);
      spent += one + many;
      if (many > 1.5) {
        break; // a very slow GPU: one pair says enough
      }
    }
    await run(TICKS, tick, bgTick, 1);
    const tk = { one: Infinity, many: Infinity };
    for (let i = 0; i < 3; i++) {
      tk.one = Math.min(tk.one, await run(1, tick, bgTick, 1));
      tk.many = Math.min(tk.many, await run(TICKS, tick, bgTick, 1));
    }
    if (!(mv.many > mv.one) || !(tk.many > tk.one)) {
      throw new Error(`timing without resolution (matvec ${mv.one} / ${mv.many} s, dispatch ${tk.one} / ${tk.many} s)`);
    }
    return { gbps: ((MV_N - 1) * BENCH_BYTES) / (mv.many - mv.one) / 1e9, dispatchUs: ((tk.many - tk.one) / (TICKS - 1)) * 1e6 };
  } finally {
    for (const b of [w, x, y, c]) {
      b.destroy();
    }
  }
}

/// gather the facts in this browser and judge them; onStep(verdict "pending") is called once the adapter passed,
/// before the benchmark runs. gpu = navigator.gpu (undefined: none); env as judge's
export async function runGpuCheck(gpu, env, onStep) {
  const facts = { gpu: gpu !== undefined && gpu !== null, adapter: null, bench: null };
  if (!facts.gpu || browserOf(env.ua).name === "firefox") {
    return judge(facts, env);
  }
  let adapter;
  try {
    adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch (e) {
    return judge({ ...facts, error: `requestAdapter failed: ${e instanceof Error ? e.message : e}` }, env);
  }
  if (adapter === null) {
    return judge(facts, env);
  }
  const info = adapter.info ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  facts.adapter = { info, fallback: adapter.isFallbackAdapter === true, f16: adapter.features.has("shader-f16"), limits: adapter.limits };
  const first = judge(facts, env);
  if (first.verdict !== "pending") {
    return first;
  }
  onStep(first);
  let device;
  try {
    device = await adapter.requestDevice();
  } catch (e) {
    return judge({ ...facts, error: `requestDevice failed: ${e instanceof Error ? e.message : e}` }, env);
  }
  // a short pause after the device is up, then the benchmark; a failed or not-green first run is run ONCE more
  // (the better of the two counts): the first run after a page load was often low
  try {
    await new Promise((r) => setTimeout(r, 300));
    let error = null;
    for (let attempt = 1; attempt <= 2; attempt++) {
      let bench;
      try {
        bench = await benchmark(device);
      } catch (e) {
        error = e;
        continue;
      }
      facts.bench = facts.bench === null ? bench : { gbps: Math.max(facts.bench.gbps, bench.gbps), dispatchUs: Math.min(facts.bench.dispatchUs, bench.dispatchUs) };
      if (judge(facts, env).verdict === "ok") {
        break;
      }
    }
    if (facts.bench === null) {
      return judge({ ...facts, error: `the GPU benchmark failed: ${error instanceof Error ? error.message : error}` }, env);
    }
  } finally {
    device.destroy();
  }
  return judge(facts, env);
}
