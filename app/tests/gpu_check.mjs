// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// web/gpu-check.js's pure part in Node: adapter info → integrated / discrete, limits → ok / reject, bandwidth →
// est. tokens/s → class, the verdicts and hints.   node tests/gpu_check.mjs
import * as g from "../web/gpu-check.js";

let failed = 0;
function ok(c, m) {
  console.log(`${c ? "PASS" : "FAIL"} ${m}`);
  if (!c) {
    failed++;
  }
}

const UA = {
  winChrome: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
  winEdge: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0",
  winChrome110: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
  linuxChrome: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
  macChrome: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36",
  macSafari18: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15",
  macSafari26: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15",
  iphone18: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
  iphone26: "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1",
  ipadChrome: "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/130.0 Mobile/15E148 Safari/604.1",
  firefox: "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0",
  android: "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36",
};

// ── os / browser ──
for (const [ua, tp, os] of [[UA.winChrome, 0, "windows"], [UA.linuxChrome, 0, "linux"], [UA.macChrome, 0, "mac"], [UA.macSafari26, 5, "ipad"], [UA.iphone26, 5, "iphone"], [UA.ipadChrome, 5, "ipad"], [UA.android, 5, "android"]]) {
  ok(g.osOf(ua, tp) === os, `osOf ${ua.slice(13, 45)}… ${tp} → ${g.osOf(ua, tp)} (want ${os})`);
}
for (const [ua, name, major] of [[UA.winEdge, "edge", 141], [UA.winChrome, "chrome", 141], [UA.firefox, "firefox", 140], [UA.macSafari18, "safari", 18], [UA.iphone26, "safari", 26], [UA.ipadChrome, "safari", 17]]) {
  const b = g.browserOf(ua);
  ok(b.name === name && b.major === major, `browserOf → ${b.name} ${b.major} (want ${name} ${major})`);
}

// ── adapter info → kind ──
const kinds = [
  [{ vendor: "apple", architecture: "metal-3" }, "apple"],
  [{ vendor: "intel", architecture: "gen-12lp" }, "integrated"],
  [{ vendor: "intel", architecture: "gen-9", description: "Intel(R) UHD Graphics 630" }, "integrated"],
  [{ vendor: "intel", architecture: "xe-lpg", description: "Intel(R) Iris(R) Xe Graphics" }, "integrated"],
  [{ vendor: "intel", architecture: "xe-hpg", description: "" }, "discrete"],
  [{ vendor: "intel", architecture: "", description: "Intel(R) Arc(TM) A770 Graphics" }, "discrete"],
  [{ vendor: "nvidia", architecture: "turing" }, "discrete"],
  [{ vendor: "nvidia", architecture: "", description: "NVIDIA GeForce RTX 2070 with Max-Q Design" }, "discrete"],
  [{ vendor: "amd", architecture: "", description: "AMD Radeon(TM) Graphics" }, "integrated"],
  [{ vendor: "amd", architecture: "", description: "AMD Radeon(TM) Vega 8 Graphics" }, "integrated"],
  [{ vendor: "amd", architecture: "", description: "AMD Radeon 780M" }, "integrated"],
  [{ vendor: "amd", architecture: "rdna-2", description: "AMD Radeon RX 6800 XT" }, "discrete"],
  [{ vendor: "amd", architecture: "rdna-2", description: "" }, "unknown"],
  [{ vendor: "qualcomm", architecture: "adreno-7xx" }, "integrated"],
];
for (const [info, want] of kinds) {
  ok(g.adapterKind(info) === want, `adapterKind ${JSON.stringify(info)} → ${g.adapterKind(info)} (want ${want})`);
}
ok(g.isSoftware({ vendor: "google", architecture: "swiftshader" }, false), "SwiftShader = software");
ok(g.isSoftware({ vendor: "", description: "Microsoft Basic Render Driver" }, false), "Microsoft Basic Render = software");
ok(g.isSoftware({ vendor: "intel" }, true), "isFallbackAdapter = software");
ok(!g.isSoftware({ vendor: "apple", architecture: "metal-3" }, false), "Apple = hardware");
ok(g.adapterName({ vendor: "intel", architecture: "gen-12lp", description: "" }) === "intel · gen-12lp", "adapterName without description");

// ── limits ──
const M3 = { maxBufferSize: 4294967292, maxStorageBufferBindingSize: 4294967292, maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 1024, maxComputeWorkgroupSizeX: 1024, maxComputeWorkgroupSizeY: 1024, maxStorageBuffersPerShaderStage: 10 };
const WEBGPU_DEFAULT = { maxBufferSize: 268435456, maxStorageBufferBindingSize: 134217728, maxComputeWorkgroupStorageSize: 16384, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 256, maxStorageBuffersPerShaderStage: 8 };
ok(g.checkLimits(M3, true).fails.length === 0 && g.checkLimits(M3, true).notes.length === 0, "M3 limits: ok, no note");
const def = g.checkLimits(WEBGPU_DEFAULT, true);
ok(def.fails.map((f) => f.name).join() === "maxStorageBufferBindingSize", `WebGPU defaults + f16: only the binding fails (16 KB workgroup memory → the f32 path): ${JSON.stringify(def.fails)}`);
ok(g.checkLimits({ ...WEBGPU_DEFAULT, maxStorageBufferBindingSize: 2147483648 }, false).fails.length === 0, "defaults + 2 GB binding, no f16: ok (the f32 path fits 16 KB)");
ok(g.checkLimits({ ...WEBGPU_DEFAULT, maxStorageBufferBindingSize: 2147483648 }, true).fails.length === 0, "defaults + 2 GB binding, f16: ok (the f32 path fits 16 KB)");
for (const [shared, f16, path] of [[16384, true, "f32-workgroup"], [17151, true, "f32-workgroup"], [17152, true, "f16"], [32768, true, "f16"], [32768, false, "f32"], [16384, false, "f32"]]) {
  const got = g.computePath({ ...WEBGPU_DEFAULT, maxComputeWorkgroupStorageSize: shared }, f16);
  ok(got === path, `computePath ${shared} B, f16 ${f16} → ${got} (want ${path})`);
}
ok(g.checkLimits({ ...M3, maxComputeWorkgroupStorageSize: 8192 }, true).fails[0].need === 16384, "below 16 KB: even the f32 path does not fit");
const mid = g.checkLimits({ ...M3, maxStorageBufferBindingSize: 180e6 }, true);
ok(mid.fails.length === 0 && /square image needs 189\.1 MB/.test(mid.notes[0]), `binding 180 MB: ok with the square-page note: ${mid.notes[0]}`);
let threw = false;
try {
  g.checkLimits({ maxBufferSize: 1 }, true);
} catch {
  threw = true;
}
ok(threw, "a limit missing from the adapter throws");

// ── bandwidth → est. tokens/s → class ──
const ref = g.REF;
const same = g.estimateTokS({ gbps: ref.gbps, dispatchUs: ref.dispatchUs }, true);
ok(Math.abs(same - ref.tokS) < 1e-9, `the reference itself → ${same.toFixed(2)} tok/s (want ${ref.tokS})`);
const half = g.estimateTokS({ gbps: ref.gbps / 2, dispatchUs: ref.dispatchUs }, true);
ok(half > ref.tokS / 2 && half < ref.tokS / 1.9, `half the bandwidth → ${half.toFixed(1)} tok/s`);
const uhd = g.estimateTokS({ gbps: 20, dispatchUs: 10 }, true);
ok(g.speedClass(uhd) === "slow", `20 GB/s · 10 µs → ${uhd.toFixed(1)} tok/s: ${g.speedClass(uhd)}`);
const tiny = g.estimateTokS({ gbps: 8, dispatchUs: 20 }, true);
ok(g.speedClass(tiny) === "very slow", `8 GB/s · 20 µs → ${tiny.toFixed(1)} tok/s: ${g.speedClass(tiny)}`);
const f32 = g.estimateTokS({ gbps: ref.gbps, dispatchUs: 20 }, false);
ok(f32 < g.estimateTokS({ gbps: ref.gbps, dispatchUs: 20 }, true), `no shader-f16 (f32 path, more dispatches) is slower: ${f32.toFixed(1)}`);
ok(g.speedClass(25) === "ok" && g.speedClass(24.9) === "slow" && g.speedClass(8) === "slow" && g.speedClass(7.9) === "very slow", "class borders 25 / 8");
ok(g.tokSText(84.4) === "~84" && g.tokSText(400) === "150+", "tokSText");

// ── verdicts ──
const env = (ua, extra = {}) => ({ ua, maxTouchPoints: 0, secure: true, flagged: null, ...extra });
const adapter = (info, limits = M3, f16 = true, fallback = false) => ({ info, limits, f16, fallback });
const FAST = { gbps: ref.gbps, dispatchUs: ref.dispatchUs };

let r = g.judge({ gpu: false, adapter: null, bench: null }, env(UA.firefox));
ok(r.verdict === "reject" && /Firefox/.test(r.hints[0].title) && /Chrome, Edge or Safari/.test(r.hints[0].text), `no WebGPU, Firefox: ${r.hints[0].text}`);
r = g.judge({ gpu: false, adapter: null, bench: null }, env(UA.macSafari18));
ok(r.verdict === "reject" && /version 26/.test(r.hints[0].text) && /macOS/.test(r.hints[0].text), `no WebGPU, Safari 18: ${r.hints[0].text}`);
r = g.judge({ gpu: false, adapter: null, bench: null }, env(UA.ipadChrome, { maxTouchPoints: 5 }));
ok(/iPadOS/.test(r.hints[0].text) && /Safari's engine/.test(r.hints[0].text), `no WebGPU, Chrome on iPad 17: ${r.hints[0].text}`);
r = g.judge({ gpu: false, adapter: null, bench: null }, env(UA.linuxChrome));
ok(/enable-unsafe-webgpu/.test(r.hints[0].text), `no WebGPU, Chrome on Linux: ${r.hints[0].text}`);
r = g.judge({ gpu: false, adapter: null, bench: null }, env(UA.winChrome110));
ok(/from version 113/.test(r.hints[0].text), `no WebGPU, Chrome 110: ${r.hints[0].text}`);
r = g.judge({ gpu: false, adapter: null, bench: null }, env(UA.winChrome, { secure: false }));
ok(/HTTPS/.test(r.hints[0].text), `insecure page: ${r.hints[0].text}`);
r = g.judge({ gpu: true, adapter: null, bench: null }, env(UA.winChrome));
ok(r.verdict === "reject" && /No usable GPU/.test(r.headline), `no adapter: ${r.headline}`);
r = g.judge({ gpu: true, adapter: adapter({ vendor: "google", architecture: "swiftshader" }, M3, true, true), bench: null }, env(UA.winChrome));
ok(r.verdict === "reject" && /software rendering/.test(r.headline), `fallback adapter: ${r.headline}`);
r = g.judge({ gpu: true, adapter: adapter({ vendor: "intel", architecture: "gen-9" }, WEBGPU_DEFAULT), bench: null }, env(UA.winChrome));
ok(r.verdict === "reject" && r.problems.length === 1 && /allows 128\.0 MB, OCR Lite needs 157\.6 MB/.test(r.problems[0]), `low limits: ${r.problems.join(" | ")}`);
r = g.judge({ gpu: true, adapter: adapter({ vendor: "apple", architecture: "metal-3" }), bench: null }, env(UA.macChrome));
ok(r.verdict === "pending", "limits pass → pending (the benchmark next)");
r = g.judge({ gpu: true, adapter: adapter({ vendor: "apple", architecture: "metal-3" }), bench: FAST }, env(UA.macChrome));
ok(r.verdict === "ok" && r.hints.length === 0 && /can run OCR Lite \(GPU: apple · metal-3, est\. ~84 tokens\/s\)/.test(r.headline), `M3: ${r.headline}`);
r = g.judge({ gpu: true, adapter: adapter({ vendor: "intel", architecture: "gen-9", description: "Intel(R) UHD Graphics 630" }), bench: { gbps: 20, dispatchUs: 10 } }, env(UA.winChrome));
ok(r.verdict === "slow" && r.cls === "slow" && /Graphics → add the browser/.test(r.hints[0].text) && /NVIDIA processor/.test(r.hints[1].text) && /mains power/.test(r.hints[2].text) && /integrated/.test(r.notes.join()), `Windows + UHD 630, slow: ${r.headline}`);
r = g.judge({ gpu: true, adapter: adapter({ vendor: "intel", architecture: "xe-lpg" }), bench: FAST }, env(UA.winChrome));
ok(r.verdict === "ok" && r.hints.length === 3, "Windows + fast integrated: ok, with the switch hints");
r = g.judge({ gpu: true, adapter: adapter({ vendor: "nvidia", architecture: "turing" }), bench: { gbps: 300, dispatchUs: 4 } }, env(UA.winChrome));
ok(r.verdict === "ok" && r.hints.length === 0 && /discrete/.test(r.detected[0]), `Windows + NVIDIA: ${r.headline}`);
r = g.judge({ gpu: true, adapter: adapter({ vendor: "nvidia", architecture: "turing" }), bench: { gbps: 6, dispatchUs: 40 } }, env(UA.winChrome));
ok(r.verdict === "slow" && r.cls === "very slow" && /very slow/.test(r.headline) && r.hints.length === 3, `very slow: ${r.headline}`);
r = g.judge({ gpu: true, adapter: adapter({ vendor: "intel", architecture: "gen-9" }), bench: { gbps: 20, dispatchUs: 10 } }, env(UA.linuxChrome));
ok(r.verdict === "slow" && /DRI_PRIME=1/.test(r.hints[0].text) && /prime-run/.test(r.hints[0].text), "Linux + Intel, slow: PRIME hint");
r = g.judge({ gpu: true, adapter: adapter({ vendor: "intel", architecture: "gen-9" }), bench: { gbps: 20, dispatchUs: 10 } }, env(UA.macChrome));
ok(r.verdict === "slow" && /Automatic graphics switching/.test(r.hints[0].text), "Intel Mac, slow: graphics switching hint");
r = g.judge({ gpu: true, adapter: adapter({ vendor: "apple", architecture: "metal-3" }, M3, false), bench: FAST }, env(UA.macSafari26));
ok(/f32 path/.test(r.notes.join()), "no shader-f16: the f32 note");
const WG16 = { ...M3, maxComputeWorkgroupStorageSize: 16384 };
r = g.judge({ gpu: true, adapter: adapter({ vendor: "nvidia", architecture: "turing" }, WG16), bench: null }, env(UA.winChrome));
ok(r.verdict === "pending" && r.problems.length === 0, `shader-f16 + 16 KB workgroup memory: not rejected (${r.verdict})`);
r = g.judge({ gpu: true, adapter: adapter({ vendor: "nvidia", architecture: "turing" }, WG16), bench: { gbps: 300, dispatchUs: 4 } }, env(UA.winChrome));
ok(r.verdict === "ok" && r.notes.includes("f32 path (this GPU's workgroup memory is too small for f16): slower, more memory."), `shader-f16 + 16 KB: ok with the f32-path note: ${r.notes.join(" | ")}`);
ok(r.estTokS === g.estimateTokS({ gbps: 300, dispatchUs: 4 }, false), "shader-f16 + 16 KB: estimated on the f32 path's dispatches");
r = g.judge({ gpu: true, adapter: adapter({ vendor: "apple", architecture: "metal-3" }), bench: FAST }, env(UA.iphone26, { maxTouchPoints: 5 }));
ok(r.verdict === "ok" && /iPhone/.test(r.notes[0]), "iPhone: ok + the memory note");
r = g.judge({ gpu: true, adapter: adapter({ vendor: "apple", architecture: "metal-3" }), bench: FAST }, env(UA.iphone26, { maxTouchPoints: 5, flagged: "The last run stopped during prefill…" }));
ok(r.verdict === "reject" && /ran out of memory/.test(r.headline), "iPhone flagged unsupported: rejected");
r = g.judge({ gpu: true, adapter: adapter({ vendor: "apple", architecture: "metal-3" }), bench: FAST }, env(UA.macSafari26, { flagged: "The last run stopped during prefill…" }));
ok(r.verdict === "ok" && /Earlier: The last run/.test(r.notes.join()), "a flagged Mac: ok + the earlier notice as a note");

console.log(failed === 0 ? "all passed" : `${failed} FAILED`);
process.exitCode = failed === 0 ? 0 : 1;
