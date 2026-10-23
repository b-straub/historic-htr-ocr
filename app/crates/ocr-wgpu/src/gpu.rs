// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// wgpu plumbing for the hand-written WGSL kernels: one device (the adapter's own limits requested — a full page needs
// storage buffers of ≈ 160 MB, over WebGPU's 128 MB default), one pipeline per kernel (binding 0 = the kernel's uniform
// params at a dynamic offset into ONE uniform buffer, 1… = storage buffers in the shader's order, read-only as declared),
// and a Batch that records dispatches into one compute pass (wgpu orders the dispatches; each reads what the previous
// one wrote). The decode step keeps its bind groups (BindCache): per token only the params change.
// The compute path (Compute) is chosen once at device creation from the adapter's capabilities: F32 (the reference:
// f32 everywhere, its reads the regression baseline), F16 (shader-f16: f16 workgroup tiles, f32 arithmetic — the
// browser), F16Coop (F16 + the matrix units for the vision matmuls / attention and the prefill matmuls — native
// Metal). The f16 paths are gated against llama.cpp's reads (llama.cpp itself computes in f16 / bf16).
// Kernels are compiled with naga's injected runtime checks in tests / debug builds only (SHADER_CHECKS, Gpu::create).
// The weights stay in their GGUF encoding on the GPU: bf16 packed two per u32 (unpacked in the shader, exact), q8_0 as
// its 34-byte blocks read as u32 words (dequantized in the shader, exact), f32 as is.
// Weight upload streams: file range → ONE reusable CPU chunk (the staging buffer: in the browser a slice of wasm memory
// the OPFS read fills directly) → queue.write_buffer at the tensor's offset (it copies the chunk at once), and a wait for
// the GPU each time a staging size has been written since the last — peak memory ≈ 2 × the staging size (the chunk +
// the driver's copies), whatever the model size (the browser must never hold the model in CPU memory).
// Everything that waits for the GPU is async (the browser build): WebGPU hands results back only through the
// event loop. Natively the same futures are completed by device.poll(wait) and driven with pollster::block_on.
// A buffer bound to a kernel must fit the device's maxStorageBufferBindingSize — checked per dispatch with a clear
// message (a browser may grant less than the 165 MB token_embd needs).
// GPU memory (an iPhone's Safari died right after the model load, no message): every buffer is a Buf,
// counted in the device's ledger by what it holds (MemKind) while any clone of it lives — the engine's resident bytes,
// its peak and each read phase's peak are MEASURED by that count (ReadMemory). Not in the count: the driver's rounding,
// wgpu's / the browser's internal staging, pipelines. No browser has an API for the FREE GPU memory (Safari has not even
// navigator.deviceMemory), so running out can only be flagged after the fact: a lost device or an out-of-memory error
// (Health — the engine is then dead, every later GPU wait fails with the reason instead of hanging) or, when the OS
// kills the whole tab, the browser host's crash marker (web/bridge-web.js).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Poll;

use anyhow::{Context, Result, anyhow, bail};
use futures_channel::oneshot;
use ocr_decode::sampling::{GpuBytes, PhaseMemory, ReadMemory};
use web_time::Instant;

use crate::gguf::{TensorInfo, TensorSource};

/// naga's runtime checks in the kernels (bounds, loop guards, integer division): on in tests and debug builds, off in
/// release builds (see Gpu::create); the load line reports it
pub const SHADER_CHECKS: bool = cfg!(any(test, debug_assertions));

/// a staging piece's start in the staging buffer: buffer copies need offsets of 4 bytes; 256 keeps them aligned
const COPY_ALIGN: u64 = 256;

/// the default staging size of the weight upload
pub const STAGING_BYTES: u64 = 32 << 20;

const MATMUL: &str = include_str!("shaders/matmul.wgsl");
const MATMUL128: &str = include_str!("shaders/matmul128.wgsl");

/// matmul128.wgsl with its weight loader and the 16 accumulator lines written out (`//@ACC <line>`: acc_R_C = row
/// R 0…7 × column group C 0…1; AROW = the row's A value, BCOL = the column group's B vec4, OUTROW / OUTCOL = where
/// the thread's outputs go)
fn matmul128(loader: &str, f16: bool) -> String {
    let src = MATMUL128.replace("//WEIGHT_LOADER4", loader).replace("TILE", if f16 { "vec4<f16>" } else { "vec4<f32>" });
    let src = if f16 { format!("enable f16;\n{src}") } else { src };
    let mut out = String::with_capacity(src.len() * 2);
    for line in src.lines() {
        match line.split_once("//@ACC ") {
            Some((indent, body)) if indent.trim().is_empty() => {
                for r in 0..8 {
                    for c in 0..2 {
                        let arow = format!("a{}.{}", r / 4, ["x", "y", "z", "w"][r % 4]);
                        let outrow = format!("row0 + {}u + lid.y * 4u + {}u", (r / 4) * 64, r % 4);
                        let l = body
                            .replace("acc_R_C", &format!("acc_{r}_{c}"))
                            .replace("AROW", &arow)
                            .replace("BCOL", &format!("b{c}"))
                            .replace("OUTROW", &outrow)
                            .replace("OUTCOL", &format!("{}u", c * 64));
                        out.push_str(indent);
                        out.push_str(&l);
                        out.push('\n');
                    }
                }
            }
            _ => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

/// the coop kernels' register blocks written out: a line `//@<MARK> <body>` becomes one line per (R, I, J) of the mark's
/// ranges; in the body `_R` / `_I` / `_J` become `_0`, `_1`, … (register names) and a standalone `R` / `I` / `J` becomes
/// `0u`, `1u`, …
///   COOP: I, J 0…3 · COOPA: I 0…3 · D8 / D16: I 0…7 / 0…15 · KB: J 0…3 · KBD8 / KBD16: I 0…7 / 0…15 × J 0…3 ·
///   R2D8 / R2KB / R2KBD8: the same with the row block R 0…1 · C42: I 0…3 × J 0…1 · J2: J 0…1
fn coop(src: &str) -> String {
    fn sub(body: &str, idx: [usize; 3]) -> String {
        let names = ['R', 'I', 'J'];
        let mut body = body.to_string();
        for (n, v) in names.iter().zip(idx) {
            body = body.replace(&format!("_{n}"), &format!("_{v}"));
        }
        let c: Vec<char> = body.chars().collect();
        let word = |k: usize| c.get(k).is_some_and(|ch| ch.is_alphanumeric() || *ch == '_');
        let mut out = String::with_capacity(body.len() + 8);
        for (k, &ch) in c.iter().enumerate() {
            match names.iter().position(|&n| n == ch) {
                Some(i) if (k == 0 || !word(k - 1)) && !word(k + 1) => out.push_str(&format!("{}u", idx[i])),
                _ => out.push(ch),
            }
        }
        out
    }
    // (mark, R range, I range, J range) — longer marks first where one is a prefix of another
    let marks: [(&str, usize, usize, usize); 12] = [
        ("//@C42 ", 1, 4, 2),
        ("//@J2 ", 1, 1, 2),
        ("//@COOPA ", 1, 4, 1),
        ("//@COOP ", 1, 4, 4),
        ("//@KBD8 ", 1, 8, 4),
        ("//@D8 ", 1, 8, 1),
        ("//@KBD16 ", 1, 16, 4),
        ("//@D16 ", 1, 16, 1),
        ("//@KB ", 1, 1, 4),
        ("//@R2KBD8 ", 2, 8, 4),
        ("//@R2D8 ", 2, 8, 1),
        ("//@R2KB ", 2, 1, 4),
    ];
    let mut out = String::with_capacity(src.len() * 2);
    'line: for line in src.lines() {
        for (mark, nr, ni, nj) in marks {
            if let Some((indent, body)) = line.split_once(mark).filter(|(i, _)| i.trim().is_empty()) {
                for r in 0..nr {
                    for i in 0..ni {
                        for j in 0..nj {
                            out.push_str(indent);
                            out.push_str(&sub(body, [r, i, j]));
                            out.push('\n');
                        }
                    }
                }
                continue 'line;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// rows per 32-lane group of the decode matvec kernels (dmv_common.wgsl): a workgroup computes 8 · DMV_R rows
pub const DMV_R: usize = 2;
/// the same for dmv_ffn (two matrices per group: 2 · DMV_FFN_R accumulators per lane)
pub const DMV_FFN_R: usize = 1;

/// a decode matvec kernel: [enable subgroups] + dmv_common + one reduction file + a dot function per weight binding +
/// the kernel `body` (r rows per group, ra accumulators per lane)
fn dmv_source(body: &str, weights: &[&str], r: usize, ra: usize, subgroups: bool) -> String {
    // (naga takes the subgroup builtins with the device feature alone — it rejects `enable subgroups;` for now)
    let reduce = if subgroups { include_str!("shaders/dmv_reduce_sg.wgsl") } else { include_str!("shaders/dmv_reduce_wg.wgsl") };
    let mut src = String::from(include_str!("shaders/dmv_common.wgsl"));
    src.push_str(reduce);
    for w in weights {
        src.push_str(&include_str!("shaders/dmv_dot.wgsl").replace("WNAME", w));
    }
    src.push_str(body);
    src.replace("RA_u", &format!("{ra}u")).replace("R_u", &format!("{r}u")).replace("TOT_u", &format!("{}u", 8 * ra)).replace("RED_u", &format!("{}u", 256 * ra))
}

/// dmv.wgsl without (dmv) or with (dmv_norm) the fused input norm
fn dmv_plain(norm: bool, subgroups: bool) -> String {
    let mut body = String::new();
    for line in include_str!("shaders/dmv.wgsl").lines() {
        match line.split_once("//@NORM ") {
            Some((indent, rest)) if indent.trim().is_empty() => {
                if norm {
                    body.push_str(indent);
                    body.push_str(rest);
                    body.push('\n');
                }
            }
            _ => {
                body.push_str(line);
                body.push('\n');
            }
        }
    }
    let body = if norm { body.replace("B_W", "3").replace("B_Y", "4") } else { body.replace("B_W", "2").replace("B_Y", "3") };
    dmv_source(&body, &["W"], DMV_R, DMV_R, subgroups)
}

/// the kernels: name → WGSL source (the f16 kernels only where the device computes in f16; the decode kernels'
/// reductions with subgroup sums where `subgroups`)
fn sources(compute: Compute, subgroups: bool) -> Vec<(&'static str, String)> {
    let mut v = sources_f32();
    if compute != Compute::F32 {
        v.push(("dmv", dmv_plain(false, subgroups)));
        v.push(("dmv_norm", dmv_plain(true, subgroups)));
        v.push(("dmv_qkv", dmv_source(include_str!("shaders/dmv_qkv.wgsl"), &["Wq", "Wk", "Wv"], DMV_R, DMV_R, subgroups)));
        v.push(("dmv_ffn", dmv_source(include_str!("shaders/dmv_ffn.wgsl"), &["Wg", "Wu"], DMV_FFN_R, 2 * DMV_FFN_R, subgroups)));
        v.push(("qkv_post", include_str!("shaders/qkv_post.wgsl").into()));
        v.push(("attn_decode", include_str!("shaders/attn_decode.wgsl").replace("CH_", &DECODE_CHUNK.to_string())));
        v.push(("attn_merge", include_str!("shaders/attn_merge.wgsl").replace("CH_", &DECODE_CHUNK.to_string())));
        for (name, loader) in [("matmul128h_bf16", include_str!("shaders/load4_bf16.wgsl")), ("matmul128h_f32", include_str!("shaders/load4_f32.wgsl")), ("matmul128h_q8_0", include_str!("shaders/load4_q8_0.wgsl"))] {
            v.push((name, matmul128(loader, true)));
        }
        v.push(("heads_f16", include_str!("shaders/heads_f16.wgsl").into()));
        v.push(("attention_tile", include_str!("shaders/attention_tile.wgsl").into()));
        v.push(("f16_from_bf16", include_str!("shaders/f16_from_bf16.wgsl").into()));
        v.push(("matmul128h_f16", matmul128(include_str!("shaders/load4_f16.wgsl"), true)));
        // the vision encoder's f16 activations: X read as f16 (exactly the values the f32-X kernel rounded to its tiles)
        v.push(("matmul128hx_f16", f16_x(&matmul128(include_str!("shaders/load4_f16.wgsl"), true))));
        v.push(("rms_norm_h", rms_norm_h()));
        v.push(("attention_causal_h", attention_causal_h()));
        v.push(("swiglu_h", swiglu_h()));
    }
    if compute == Compute::F16Coop {
        let mc = include_str!("shaders/matmul_coop.wgsl");
        v.push(("matmul_coop_bf16", coop(&mc.replace("//WEIGHT_LOADER4", include_str!("shaders/load4_bf16.wgsl")))));
        v.push(("matmul_coop_f32", coop(&mc.replace("//WEIGHT_LOADER4", include_str!("shaders/load4_f32.wgsl")))));
        v.push(("matmul_coop_q8_0", coop(&mc.replace("//WEIGHT_LOADER4", include_str!("shaders/load4_q8_0.wgsl")))));
        v.push(("matmul_coop_f16", coop(&mc.replace("//WEIGHT_LOADER4", include_str!("shaders/load4_f16.wgsl")))));
        v.push(("attention_coop", coop(include_str!("shaders/attention_coop.wgsl"))));
        v.push(("matmul_coopd", coop(include_str!("shaders/matmul_coopd.wgsl"))));
        v.push(("ffn_coopd", coop(include_str!("shaders/ffn_coopd.wgsl"))));
    }
    v
}

/// `src` with each (from, to) replaced — every `from` must occur (a renamed line in the WGSL fails loudly, at start-up)
fn patched(src: &str, pairs: &[(&str, &str)]) -> String {
    let mut s = src.to_string();
    for (from, to) in pairs {
        assert!(s.contains(from), "kernel source patch: {from:?} not found");
        s = s.replace(from, to);
    }
    s
}

/// matmul128 (f16 tiles) reading X as f16 (two per u32, four per vec2<u32>): exact, the values its tiles held before
fn f16_x(src: &str) -> String {
    patched(src, &[
        ("@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;", "@group(0) @binding(1) var<storage, read> X: array<vec2<u32>>;"),
        ("    let v = X[i];\n", "    let w = X[i];\n    let v = vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));\n"),
    ])
}

/// attention_causal.wgsl for the f16 paths: the K / V tiles kept as f16 in workgroup memory (exact: the cache is f16),
/// so tiles of 16 keys fit WebGPU's 16 KB default (f32 tiles: 8 keys) — half the barriers per key. Measured (M3,
/// a 980 × 1540 page, 1939 prompt tokens, ms per layer): 8-key f32 tiles 28.9, 16-key f16 tiles 23.7
fn attention_causal_h() -> String {
    let s = patched(include_str!("shaders/attention_causal.wgsl"), &[
        ("const TK: u32 = 8u;", "const TK: u32 = 16u;"),
        ("var<workgroup> Ks: array<vec4<f32>, 256>; // [TK][D4]", "var<workgroup> Ks: array<vec4<f16>, 512>; // [TK][D4]"),
        ("var<workgroup> Vs: array<vec4<f32>, 256>;", "var<workgroup> Vs: array<vec4<f16>, 512>;"),
        ("var<workgroup> P: array<f32, 1024>; // [2 halves][TK][64 queries]", "var<workgroup> P: array<f32, 2048>; // [2 halves][TK][64 queries]"),
        ("Ks[e] = h4(K[src]);", "Ks[e] = vec4<f16>(h4(K[src]));"),
        ("Vs[e] = h4(V[src]);", "Vs[e] = vec4<f16>(h4(V[src]));"),
        ("d = fma(q#, Ks[kb + #u], d);", "d = fma(q#, vec4<f32>(Ks[kb + #u]), d);"),
        ("acc# = fma(vec4<f32>(pj), Vs[vb + #u], acc#);", "acc# = fma(vec4<f32>(pj), vec4<f32>(Vs[vb + #u]), acc#);"),
    ]);
    unroll(&format!("enable f16;\n{s}"), 16)
}

/// rms_norm writing f16 (the vision encoder's matmul inputs on the f16 paths; the residual it reads stays f32)
fn rms_norm_h() -> String {
    let s = patched(include_str!("shaders/rms_norm.wgsl"), &[
        ("var<storage, read_write> Y: array<f32>;", "var<storage, read_write> Y: array<f16>;"),
        ("Y[base + i] = X[base + i] * scale * w[i];", "Y[base + i] = f16(X[base + i] * scale * w[i]);"),
    ]);
    format!("enable f16;\n{s}")
}

/// swiglu writing its result times out_scale (params word 1) as f16 into a third buffer H (the down projection's input
/// on the F16 path, ffn_f16)
fn swiglu_h() -> String {
    let s = patched(include_str!("shaders/swiglu.wgsl"), &[
        ("@group(0) @binding(2) var<storage, read_write> G: array<f32>;", "@group(0) @binding(2) var<storage, read> G: array<f32>;\n@group(0) @binding(3) var<storage, read_write> H: array<f16>;"),
        ("G[i] = g / (1.0 + exp(-g)) * U[i];", "H[i] = f16(g / (1.0 + exp(-g)) * U[i] * bitcast<f32>(p._p0));"),
    ]);
    format!("enable f16;\n{s}")
}

fn sources_f32() -> Vec<(&'static str, String)> {
    vec![
        ("matmul_bf16", MATMUL.replace("//WEIGHT_LOADER", include_str!("shaders/load_bf16.wgsl"))),
        ("matmul_f32", MATMUL.replace("//WEIGHT_LOADER", include_str!("shaders/load_f32.wgsl"))),
        ("matmul_q8_0", MATMUL.replace("//WEIGHT_LOADER", include_str!("shaders/load_q8_0.wgsl"))),
        ("matmul128_bf16", matmul128(include_str!("shaders/load4_bf16.wgsl"), false)),
        ("matmul128_f32", matmul128(include_str!("shaders/load4_f32.wgsl"), false)),
        ("matmul128_q8_0", matmul128(include_str!("shaders/load4_q8_0.wgsl"), false)),
        // f16 weights (bf16 converted at load on the f16 paths) in the generic kernels: the bisect / odd shapes
        ("matmul_f16", MATMUL.replace("//WEIGHT_LOADER", include_str!("shaders/load_f16.wgsl"))),
        ("matmul128_f16", matmul128(include_str!("shaders/load4_f16.wgsl"), false)),
        ("matvec_q8_0", include_str!("shaders/matvec_q8_0.wgsl").into()),
        ("embed_q8_0", include_str!("shaders/embed_q8_0.wgsl").into()),
        ("copy_words", include_str!("shaders/copy_words.wgsl").into()),
        ("rope_neox", include_str!("shaders/rope_neox.wgsl").into()),
        ("kv_store", include_str!("shaders/kv_store.wgsl").into()),
        ("attention_causal", unroll(include_str!("shaders/attention_causal.wgsl"), 16)),
        ("attn_scores", include_str!("shaders/attn_scores.wgsl").into()),
        ("attn_softmax", include_str!("shaders/attn_softmax.wgsl").into()),
        ("attn_values", include_str!("shaders/attn_values.wgsl").into()),
        ("rms_norm", include_str!("shaders/rms_norm.wgsl").into()),
        ("rope2d", include_str!("shaders/rope2d.wgsl").into()),
        ("attention", unroll(include_str!("shaders/attention.wgsl"), 16)),
        ("swiglu", include_str!("shaders/swiglu.wgsl").into()),
        ("gelu", include_str!("shaders/gelu.wgsl").into()),
        ("unfold2x2", include_str!("shaders/unfold2x2.wgsl").into()),
        ("patchify", include_str!("shaders/patchify.wgsl").into()),
        // the GPU token choice of the greedy presets (text.rs Choice): every compute path
        ("choose_pen", include_str!("shaders/choose_pen.wgsl").into()),
        ("choose_max", include_str!("shaders/choose_max.wgsl").into()),
        ("choose_pick", include_str!("shaders/choose_pick.wgsl").into()),
    ]
}

/// expand every `//@UNROLL <line>` into n lines with `#` = 0…n−1 (register-resident vectors, see attention.wgsl)
fn unroll(src: &str, n: usize) -> String {
    let mut out = String::with_capacity(src.len() * 2);
    for line in src.lines() {
        match line.split_once("//@UNROLL ") {
            Some((indent, body)) if indent.trim().is_empty() => {
                for i in 0..n {
                    out.push_str(indent);
                    out.push_str(&body.replace('#', &i.to_string()));
                    out.push('\n');
                }
            }
            _ => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

/// workgroups for a grid-stride elementwise kernel (256 threads each)
pub fn stride_groups(n: u64) -> u32 {
    n.div_ceil(256).clamp(1, 16384) as u32
}

/// the compute path, a capability of the adapter chosen once at device creation:
///   F32      f32 activations and arithmetic everywhere (the reference path: its reads are the regression baseline)
///   F16      shader-f16: f16 storage / inputs where it pays, f32 accumulation (browsers: Chrome, Safari)
///   F16Coop  F16 + the GPU's matrix units for the big products (wgpu's native cooperative matrices: Metal simdgroup
///            matrices, 8×8 f16 × f16 → f32)
/// The weights' STORAGE format is a separate thing (Engine::weights, "q8_0 text · bf16 vision" from the GGUFs): on the
/// f16 paths the bf16 vision weights are converted to f16 once at load (f16_from_bf16.wgsl, a load refuses a weight
/// beyond f16's range), the q8_0 text weights stay q8_0 and are dequantized in the kernels.
/// What the f16 paths keep in f32 (checked in round 3; vision.rs run_f16 lists it per tensor):
///   - the residual streams (vision x reaches 1.4e5 in a test page's last layer — beyond f16's 65504: an
///     f16 mmproj overflowed to NaN on CUDA; text x), every norm's sum of squares and output scale
///   - every accumulator: matmul / matrix-unit sums, attention scores, the softmax (running max, exp, sum), P·V
///   - q / k / v after their projections, RoPE, the projector tail's activations, the logits
/// What is f16: matmul / matrix-unit INPUTS (normed activations ≤ 10, attention outputs ≤ 20, the FFN's SwiGLU
/// product stored ÷ 64 — it reached 3.6e4, vision.rs FFN_DOWN_SCALE), the attention's K / V (≤ 21) and P (≤ 1), the
/// text KV cache (f16 like llama.cpp's default). Magnitudes measured on the test pages (three small crops,
/// four full pages).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compute {
    F32,
    F16,
    F16Coop,
}

impl Compute {
    fn features(self) -> wgpu::Features {
        match self {
            Compute::F32 => wgpu::Features::empty(),
            Compute::F16 => wgpu::Features::SHADER_F16,
            Compute::F16Coop => wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SUBGROUP | wgpu::Features::SUBGROUP_BARRIER,
        }
    }

    /// can an adapter with these features / limits run this path? (Err: why not; `coop_mixed`: it has an 8×8 f16 × f16
    /// → f32 cooperative matrix)
    fn check_caps(self, name: &str, features: wgpu::Features, limits: &wgpu::Limits, coop_mixed: bool) -> std::result::Result<(), String> {
        let missing = self.features() - features;
        if !missing.is_empty() {
            return Err(format!("{name} lacks {missing:?}"));
        }
        let shared = limits.max_compute_workgroup_storage_size;
        let need = self.workgroup_need()?;
        if shared < need {
            return Err(format!("{name}: {shared} bytes of workgroup memory, the {self:?} kernels need {need}"));
        }
        if self == Compute::F16Coop {
            if !coop_mixed {
                return Err(format!("{name} has no 8×8 f16 → f32 cooperative matrix"));
            }
            if shared < 24 * 1024 {
                return Err(format!("{name}: {shared} bytes of workgroup memory, attention_coop needs 24 KB"));
            }
        }
        Ok(())
    }

    /// the most workgroup memory one of this path's kernels declares (sources, workgroup_bytes) — the F16 path's
    /// attention_tile.wgsl is above WebGPU's guaranteed 16 KB, the F32 kernels are within it
    fn workgroup_need(self) -> std::result::Result<u32, String> {
        let mut need = 0;
        for (name, src) in sources(self, self == Compute::F16Coop) {
            need = need.max(workgroup_bytes(name, &src)?);
        }
        Ok(need)
    }

    /// the path for these capabilities (`want`: a forced path, an error where they lack it) and, for an F32 chosen
    /// although the adapter offers shader-f16, why: the status line's "f32 (…)"
    fn choose_caps(name: &str, features: wgpu::Features, limits: &wgpu::Limits, coop_mixed: bool, want: Option<Compute>) -> Result<(Compute, Option<String>)> {
        if let Some(c) = want {
            c.check_caps(name, features, limits, coop_mixed).map_err(|e| anyhow!("compute path {c:?}: {e}"))?;
            let why = match c {
                Compute::F32 if features.contains(wgpu::Features::SHADER_F16) => Some("forced".to_string()),
                Compute::F32 => Some("no shader-f16".to_string()),
                _ => None,
            };
            return Ok((c, why));
        }
        if let Some(c) = [Compute::F16Coop, Compute::F16].into_iter().find(|c| c.check_caps(name, features, limits, coop_mixed).is_ok()) {
            return Ok((c, None));
        }
        Compute::F32.check_caps(name, features, limits, coop_mixed).map_err(|e| anyhow!("compute path F32: {e}"))?;
        if !features.contains(wgpu::Features::SHADER_F16) {
            return Ok((Compute::F32, Some("no shader-f16".to_string())));
        }
        // shader-f16 without the workgroup memory of the f16 kernels (WebGPU guarantees 16 KB): the F32 path fits it
        let need = Compute::F16.workgroup_need().map_err(|e| anyhow!(e))?;
        let shared = limits.max_compute_workgroup_storage_size;
        if shared >= need {
            bail!("{name}: shader-f16 and {shared} bytes of workgroup memory, yet the F16 path was refused");
        }
        Ok((Compute::F32, Some(format!("workgroup memory {} KB", shared / 1024))))
    }

    fn choose(adapter: &wgpu::Adapter, info: &wgpu::AdapterInfo, want: Option<Compute>) -> Result<(Compute, Option<String>)> {
        // the cooperative matrix properties asked only where the feature is (the browser backend has neither)
        let features = adapter.features();
        let coop = features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) && coop_mixed(adapter);
        Self::choose_caps(&info.name, features, &adapter.limits(), coop, want)
    }
}

/// the adapter has the cooperative matrix matmul_coop.wgsl uses: 8×8 f16 × f16 → f32 (simdgroups of 32 consecutive
/// invocations — probed on the device: probe_subgroup_size)
fn coop_mixed(adapter: &wgpu::Adapter) -> bool {
    adapter.cooperative_matrix_properties().iter().any(|p| {
        p.m_size == 8 && p.n_size == 8 && p.k_size == 8 && p.ab_type == wgpu::CooperativeScalarType::F16 && p.cr_type == wgpu::CooperativeScalarType::F32
    })
}

/// the workgroup memory a kernel's WGSL declares, counted as WebGPU limits it (maxComputeWorkgroupStorageSize: each
/// `var<workgroup>` rounded up to 16 bytes): its `var<workgroup> X: array<T, N>;` lines, T a scalar / vector of f16,
/// f32, u32, i32 or an `alias` of one, N a literal — anything else fails loudly
fn workgroup_bytes(name: &str, src: &str) -> std::result::Result<u32, String> {
    let mut aliases: HashMap<&str, &str> = HashMap::new();
    for line in src.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("alias ") {
            let (a, t) = rest.split_once('=').ok_or_else(|| format!("kernel {name}: alias line {line:?}"))?;
            let t = t.split(';').next().ok_or_else(|| format!("kernel {name}: alias line {line:?}"))?;
            aliases.insert(a.trim(), t.trim());
        }
    }
    let scalar = |t: &str| -> Option<u32> {
        match t {
            "f16" => Some(2),
            "f32" | "u32" | "i32" => Some(4),
            _ => None,
        }
    };
    let size_of = |t: &str| -> Option<u32> {
        let t = aliases.get(t).copied().unwrap_or(t);
        if let Some(s) = scalar(t) {
            return Some(s);
        }
        let (n, inner) = if let Some(i) = t.strip_prefix("vec2<") {
            (2, i)
        } else if let Some(i) = t.strip_prefix("vec4<") {
            (4, i)
        } else {
            return None;
        };
        scalar(inner.strip_suffix('>')?).map(|s| n * s)
    };
    let mut total = 0;
    for line in src.lines() {
        let Some(rest) = line.trim_start().strip_prefix("var<workgroup>") else { continue };
        let bad = || format!("kernel {name}: workgroup line {line:?}");
        let decl = rest.split(';').next().ok_or_else(bad)?;
        let (_, ty) = decl.split_once(':').ok_or_else(bad)?;
        let ty = ty.trim();
        let bytes = if let Some(arr) = ty.strip_prefix("array<").and_then(|a| a.strip_suffix('>')) {
            let (elem, count) = arr.rsplit_once(',').ok_or_else(bad)?;
            let count: u32 = count.trim().trim_end_matches('u').parse().map_err(|_| bad())?;
            size_of(elem.trim()).ok_or_else(bad)? * count
        } else {
            size_of(ty).ok_or_else(bad)?
        };
        total += bytes.div_ceil(16) * 16;
    }
    Ok(total)
}

/// what a GPU buffer holds — the memory ledger's columns (GpuBytes)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemKind {
    Weights,
    KvCache,
    Work,
    Readback,
}

impl MemKind {
    fn slot(self, b: &mut GpuBytes) -> &mut u64 {
        match self {
            MemKind::Weights => &mut b.weights,
            MemKind::KvCache => &mut b.kv_cache,
            MemKind::Work => &mut b.work,
            MemKind::Readback => &mut b.readback,
        }
    }
}

/// a lock that survives a panic elsewhere: the ledger / health hold plain counters and flags, valid at any point
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// the memory ledger: the bytes of the live buffers by kind, the highest total since the device was created and, for
/// the running read, each phase's highest total with its parts (Gpu::phase names the phase; "-" = none)
struct Ledger {
    live: GpuBytes,
    peak: u64,
    phase: &'static str,
    phases: Vec<PhaseMemory>,
}

impl Ledger {
    /// a new total: the peaks follow it
    fn note(&mut self) {
        let live = self.live;
        self.peak = self.peak.max(live.total());
        let ph = self.phase;
        if ph == "-" {
            return;
        }
        match self.phases.iter_mut().find(|p| p.phase == ph) {
            Some(p) => {
                if live.total() > p.peak.total() {
                    p.peak = live;
                }
            }
            None => self.phases.push(PhaseMemory { phase: ph, peak: live }),
        }
    }
}

/// a GPU buffer of the engine: a range of a wgpu::Buffer, counted in the ledger of the Gpu that made it for as long as
/// any clone of it lives; the last clone destroys it (Counted::drop) — the GPU frees it once the work already
/// submitted with it is done. A Buf must outlive the submission of every batch that binds it.
/// A Buf is a range of a wgpu buffer — `offset`, `size` — bound, copied and read with that range (Buf::binding):
/// a buffer of its own (offset 0), or a slice of a memory pool's block.
/// A view (Buf::view) is a sub-range of a Buf: no memory of its own, it keeps its Buf alive.
#[derive(Clone)]
pub struct Buf {
    c: Arc<Counted>,
    /// the range in the wgpu buffer (a view's lies inside its Buf's)
    off: u64,
    len: u64,
}

struct Counted {
    buf: wgpu::Buffer,
    offset: u64,
    size: u64,
    kind: MemKind,
    ledger: Arc<Mutex<Ledger>>,
    /// a pool's slice: (the pool, its block, the bytes taken there) — given back on drop, never destroyed; None: a
    /// buffer of its own, destroyed on drop
    home: Option<(Arc<Mutex<Pool>>, usize, u64)>,
}

impl Drop for Counted {
    /// destroy, not only drop the handle: in the browser wgpu's drop of a buffer is a no-op (wgpu 30 webgpu backend)
    /// and the GPUBuffer lives until the page's garbage collector takes it — Safari on the Mac still billed the page
    /// 3.0 GB after a read. destroy() frees it once the submitted work using it is done (WebGPU
    /// spec; natively wgpu-core waits the same way), so the ledger's "freed" is the GPU's.
    fn drop(&mut self) {
        match &self.home {
            Some((pool, block, taken)) => lock(pool).give(*block, self.offset, *taken),
            None => self.buf.destroy(),
        }
        let mut l = lock(&self.ledger);
        *self.kind.slot(&mut l.live) -= self.size;
    }
}

impl Buf {
    /// the wgpu buffer the range lies in
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.c.buf
    }

    /// the range's start in buffer() (bytes)
    pub fn offset(&self) -> u64 {
        self.off
    }

    /// the range's length (bytes)
    pub fn size(&self) -> u64 {
        self.len
    }

    /// what it holds (the ledger's column; weights are bound read-only)
    pub fn kind(&self) -> MemKind {
        self.c.kind
    }

    /// the range as a storage binding
    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.c.buf, offset: self.off, size: wgpu::BufferSize::new(self.len) })
    }

    /// the range as a buffer slice (a copy's or a map's source)
    pub fn slice(&self) -> wgpu::BufferSlice<'_> {
        self.c.buf.slice(self.off..self.off + self.len)
    }

    /// bytes `start` … `start + len` of this range as a Buf of their own (no memory: the same buffer) — e.g. a block
    /// of rows for a kernel; `start` must keep the device's storage offset alignment where it is bound (Batch::dispatch
    /// checks)
    pub fn view(&self, start: u64, len: u64) -> Result<Buf> {
        if start + len > self.len || len == 0 {
            bail!("a view of bytes {start}…{} of a {}-byte buffer", start + len, self.len);
        }
        Ok(Buf { c: self.c.clone(), off: self.off + start, len })
    }
}

/// a buffer of its own, counted as `kind` (offset 0, its whole size)
fn counted(ledger: &Arc<Mutex<Ledger>>, buf: wgpu::Buffer, kind: MemKind) -> Buf {
    let size = buf.size();
    let mut l = lock(ledger);
    *kind.slot(&mut l.live) += size;
    l.note();
    drop(l);
    Buf { c: Arc::new(Counted { buf, offset: 0, size, kind, ledger: ledger.clone(), home: None }), off: 0, len: size }
}

/// the free ranges of one pool block — (start, length), sorted, neighbours merged: the pool's bookkeeping (CPU only)
#[derive(Debug, PartialEq)]
struct Free(Vec<(u64, u64)>);

impl Free {
    fn new(len: u64) -> Self {
        Free(vec![(0, len)])
    }

    /// the first free range holding `size` bytes from an `align`-ed start → that start (the range split around it)
    fn take(&mut self, size: u64, align: u64) -> Option<u64> {
        for i in 0..self.0.len() {
            let (s, l) = self.0[i];
            let a = s.div_ceil(align) * align;
            if a + size > s + l {
                continue;
            }
            let mut parts = Vec::with_capacity(2);
            if a > s {
                parts.push((s, a - s));
            }
            if a + size < s + l {
                parts.push((a + size, s + l - a - size));
            }
            self.0.splice(i..=i, parts);
            return Some(a);
        }
        None
    }

    /// a range back (merged with its free neighbours)
    fn give(&mut self, start: u64, size: u64) {
        let i = self.0.partition_point(|&(s, _)| s < start);
        if let Some(&(ps, pl)) = i.checked_sub(1).and_then(|j| self.0.get(j)) {
            assert!(ps + pl <= start, "pool: range {start}+{size} given back overlaps the free {ps}+{pl}");
        }
        if let Some(&(ns, _)) = self.0.get(i) {
            assert!(start + size <= ns, "pool: range {start}+{size} given back overlaps the free range at {ns}");
        }
        self.0.insert(i, (start, size));
        if i + 1 < self.0.len() && self.0[i].0 + self.0[i].1 == self.0[i + 1].0 {
            self.0[i].1 += self.0[i + 1].1;
            self.0.remove(i + 1);
        }
        if i > 0 && self.0[i - 1].0 + self.0[i - 1].1 == self.0[i].0 {
            self.0[i - 1].1 += self.0[i].1;
            self.0.remove(i);
        }
    }
}

/// what a pool block holds while any of its slices lives: weights (bound read-only) or work (bound writable) — never
/// both: WebGPU forbids one buffer read-only and writable in one dispatch
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    Weights,
    Work,
}

/// a pool block: its buffer, length, free ranges, the bytes taken and its role (None: all free — either role may take it)
struct Block {
    buf: wgpu::Buffer,
    len: u64,
    free: Free,
    used: u64,
    role: Option<Role>,
}

/// THE memory pool: GPU buffers ("blocks", STORAGE | COPY_SRC | COPY_DST) created once and handed out in slices, taken
/// back and handed out again — never destroyed while the pool lives (a block is destroyed only with the pool). iOS kept
/// counting destroyed GPU memory against the page (a second read on top of the first's freed buffers died at the
/// iPhone's 2048 MB); a pool's memory is allocated once and stays what it is. One pool for weights and work: a block
/// takes a role with its first slice and gives it up when its last slice comes back — the vision weights' blocks hold
/// the text weights and the KV cache after the image encode, so the pool holds about the read's peak in use (two pools,
/// one per kind, held the larger of each: 852 + 864 MiB for a 1356 MiB peak — the iPhone died in the prefill).
struct Pool {
    blocks: Vec<Block>,
    /// the small blocks' size; a larger request goes to a large block (Pool::fix: as large as the largest request)
    block: u64,
    align: u64,
    /// fixed (Pool::fix, at the engine's start): no block is ever added again — a request that does not fit is an error
    fixed: bool,
    /// the most small blocks holding slices at once (the measure of POOL_SMALL_BLOCKS)
    peak_small: usize,
    /// DIAGNOSTIC (OCR_WGPU_POISON=1, native): free memory filled with 0xFF bytes (NaN as f32 and f16) — at fix and on
    /// every give — so a kernel reading outside its slice reads NaN
    poison: Option<wgpu::Queue>,
}

impl Pool {
    fn new(limits: &wgpu::Limits) -> Self {
        let block = POOL_BLOCK.min(limits.max_buffer_size).min(limits.max_storage_buffer_binding_size as u64);
        Pool { blocks: Vec::new(), block, align: (limits.min_storage_buffer_offset_alignment as u64).max(4), fixed: false, peak_small: 0, poison: None }
    }

    /// the pool at its full size, once: `small` small blocks and one large block of at least `large` bytes — from now on
    /// it never grows
    fn fix(&mut self, device: &wgpu::Device, small: usize, large: u64) -> Result<()> {
        if self.fixed {
            bail!("the memory pool is fixed already");
        }
        while self.blocks.iter().filter(|b| b.len == self.block).count() < small {
            self.add(device, self.block);
        }
        let large = large.div_ceil(self.align) * self.align;
        if !self.blocks.iter().any(|b| b.len > self.block && b.len >= large) {
            self.add(device, large);
        }
        self.fixed = true;
        if let Some(q) = &self.poison {
            for b in &self.blocks {
                q.write_buffer(&b.buf, 0, &vec![0xFFu8; b.len as usize]);
            }
        }
        Ok(())
    }

    fn add(&mut self, device: &wgpu::Device, len: u64) -> usize {
        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("pool"), size: len, usage: POOL_USAGE, mapped_at_creation: false });
        self.blocks.push(Block { buf, len, free: Free::new(len), used: 0, role: None });
        self.blocks.len() - 1
    }

    /// `size` bytes (a multiple of 4) for `role` → (the block's buffer, the slice's offset, the block's index): a block of
    /// that role, else a free one (it takes the role), else a new one
    /// small requests (≤ block) only in small blocks, larger ones only in a large block — a large block is never cut up by
    /// small slices. Fixed: a request that does not fit is an error, never a new block
    fn take(&mut self, device: &wgpu::Device, role: Role, size: u64) -> Result<(wgpu::Buffer, u64, usize)> {
        let align = self.align;
        let small = size <= self.block;
        let block = self.block;
        let fits = |b: &Block| (b.len == block) == small;
        let order = self.blocks.iter().enumerate().filter(|(_, b)| fits(b) && b.role == Some(role)).map(|(i, _)| i)
            .chain(self.blocks.iter().enumerate().filter(|(_, b)| fits(b) && b.role.is_none()).map(|(i, _)| i))
            .collect::<Vec<_>>();
        let mut found = None;
        for i in order {
            let b = &mut self.blocks[i];
            if let Some(off) = b.free.take(size, align) {
                found = Some((i, off));
                break;
            }
        }
        let (i, off) = match found {
            Some(f) => f,
            None if self.fixed => {
                let held: u64 = self.blocks.iter().map(|b| b.len).sum();
                bail!("the memory pool ({:.0} MiB, fixed) has no room for {:.1} MiB more", held as f64 / 1048576.0, size as f64 / 1048576.0);
            }
            None => {
                let i = self.add(device, if small { block } else { size });
                let off = self.blocks[i].free.take(size, align).context("a new block holds its first request")?;
                (i, off)
            }
        };
        let b = &mut self.blocks[i];
        b.used += size;
        b.role = Some(role);
        let buf = b.buf.clone();
        let n = self.blocks.iter().filter(|b| b.len == block && b.role.is_some()).count();
        self.peak_small = self.peak_small.max(n);
        Ok((buf, off, i))
    }

    fn give(&mut self, block: usize, off: u64, size: u64) {
        let b = &mut self.blocks[block];
        if let Some(q) = &self.poison {
            q.write_buffer(&b.buf, off, &vec![0xFFu8; size as usize]);
        }
        b.free.give(off, size);
        b.used -= size;
        if b.used == 0 {
            b.role = None;
        }
    }

    /// the bytes of all blocks (what the GPU holds for the pool)
    fn bytes(&self) -> u64 {
        self.blocks.iter().map(|b| b.len).sum()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for b in &self.blocks {
            b.buf.destroy();
        }
    }
}

/// the fixed pool's small blocks (Engine::open → Gpu::fix_pool): the most a read of the largest page (1540 × 1540, 3025
/// image tokens) holds at once on the F16 path — 23, measured natively (OCR_WGPU_COMPUTE=f16; its peak in use 1533 MiB in
/// the prefill; 44v then 20, a large page after it no more) — + 1 in reserve (a text longer than its image tokens grows
/// the KV cache)
pub const POOL_SMALL_BLOCKS: usize = 24;

/// the pool's blocks: what every pooled Buf is a slice of
const POOL_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC).union(wgpu::BufferUsages::COPY_DST);
/// a pool block's size (at most the device's maxBufferSize / maxStorageBufferBindingSize); a larger request — the text
/// model's token_embd 157.6 MiB, the vision FFN's f16 output ~80 MiB — gets a block of its own size (free again, it
/// serves smaller slices of either role)
const POOL_BLOCK: u64 = 64 << 20;

/// the vision FFN's row block on the F16 path (Batch::ffn_f16): its gate / up buffers hold this many rows. A multiple of
/// 128 (whole matmul tiles); the row views start at r0 · k · 2 bytes — aligned for any K of the model (1024)
pub const FFN_ROWS: usize = 1024;


/// the most rows of a large product (the vision encoder, the prefill) one command buffer holds on the F16 path (the
/// browsers): larger products are recorded in row blocks, each submitted on its own (Batch::flush) — iOS restarts the
/// GPU when one command buffer runs for long (a 1540 × 1540 page: 12,100 rows; one command buffer per layer still
/// died there, a fresh small page read)
pub const SUBMIT_ROWS: usize = 2048;
/// the most query × key pairs (all heads) of an attention one command buffer holds (Batch::flush; ~28 ms on the M3 for
/// the vision attention, attention_tile.wgsl's 198 ms per layer for 7700²)
pub const SUBMIT_PAIRS: usize = 8 << 20;
/// the most command buffers a recording keeps queued before it waits for the GPU (Batch::throttle): 1024-row blocks
/// queued without a wait (~1800 command buffers for a small page) lost the iPhone's device on every page; one per layer
/// (24 queued) read
pub const MAX_IN_FLIGHT: usize = 8;

/// an attention over `n_q` queries (blocks of 64) and up to `keys` keys in chunks of SUBMIT_PAIRS → (the first query
/// block, the blocks) per chunk
pub fn query_chunks(n_q: usize, keys: usize) -> Vec<(u32, u32)> {
    let blocks = n_q.div_ceil(64);
    let per = (SUBMIT_PAIRS / (64 * keys.max(1))).max(1);
    (0..blocks).step_by(per).map(|b0| (b0 as u32, per.min(blocks - b0) as u32)).collect()
}

/// how the host hears of a fatal GPU event (set_fatal_hook): `kind` = "lost" (the device was lost — on a phone /
/// tablet most likely the OS reclaiming memory) | "destroyed" (device.destroy() — the tests) | "outOfMemory" |
/// "validation" | "internal" (an uncaptured GPU error); `message` = the text every later GPU wait fails with.
/// A plain fn (no captures): wgpu's callbacks must be Send, a JS function is not — the browser host reaches JS
/// through its global scope.
pub type FatalHook = fn(kind: &'static str, message: &str);

static FATAL_HOOK: Mutex<Option<FatalHook>> = Mutex::new(None);

/// the host's hook for fatal GPU events (the browser build: the worker posts them to the page at once, a read running
/// or not)
pub fn set_fatal_hook(f: FatalHook) {
    *lock(&FATAL_HOOK) = Some(f);
}

/// the device's health: set once by the device-lost callback or an uncaptured GPU error
#[derive(Default)]
struct Health {
    /// why the device is unusable (the first fatal event) — every later GPU wait fails with it
    dead: Option<String>,
    /// `dead` is a suspicion (a failed readback, Gpu::map_result): the lost callback's reason, when it comes, replaces it
    provisional: bool,
    /// the Gpu is being dropped: the lost callback of its own teardown is no event
    closing: bool,
    /// the GPU waits in progress (Gpu::race): woken when the device dies — a lost device may never answer them
    waiters: Vec<oneshot::Sender<()>>,
}

fn fatal(h: &Mutex<Health>, kind: &'static str, message: String) {
    let hook = {
        let mut g = lock(h);
        if g.closing || (g.dead.is_some() && !g.provisional) {
            return;
        }
        g.dead = Some(message.clone());
        g.provisional = false;
        for w in g.waiters.drain(..) {
            // a waiter whose wait already ended has dropped its receiver: nothing to wake
            let _ = w.send(());
        }
        *lock(&FATAL_HOOK)
    };
    if let Some(f) = hook {
        f(kind, &message);
    }
}

/// a failed readback: the device is taken for dead until the lost callback names the reason (in the browser a lost
/// device's mapAsync rejects BEFORE its device.lost promise resolves) — the host hears the reason, not this
fn suspect(h: &Mutex<Health>, message: String) {
    let mut g = lock(h);
    if g.closing || g.dead.is_some() {
        return;
    }
    g.dead = Some(message);
    g.provisional = true;
    for w in g.waiters.drain(..) {
        let _ = w.send(());
    }
}

/// the text of a lost device
fn lost_message(reason: wgpu::DeviceLostReason, message: &str) -> (&'static str, String) {
    let msg = if message.is_empty() { String::new() } else { format!(": {message}") };
    match reason {
        wgpu::DeviceLostReason::Unknown => ("lost", format!("GPU device lost (unknown){msg} — most likely out of memory on this device")),
        wgpu::DeviceLostReason::Destroyed => ("destroyed", format!("GPU device lost (destroyed){msg} — device.destroy() was called")),
    }
}

/// the text of an uncaptured GPU error
fn error_message(e: &wgpu::Error) -> (&'static str, String) {
    match e {
        wgpu::Error::OutOfMemory { .. } => ("outOfMemory", format!("GPU out of memory: {e} — this device cannot hold OCR Lite's GPU buffers")),
        wgpu::Error::Validation { .. } => ("validation", format!("GPU validation error: {e}")),
        wgpu::Error::Internal { .. } => ("internal", format!("GPU internal error: {e}")),
    }
}

/// marks the Gpu as closing when dropped — the FIRST field of Gpu, so it runs before the device goes
struct Closing(Arc<Mutex<Health>>);

impl Drop for Closing {
    fn drop(&mut self) {
        lock(&self.0).closing = true;
    }
}

pub struct Gpu {
    /// dropped first (fields drop in order): the device's own teardown is no fatal event
    _closing: Closing,
    health: Arc<Mutex<Health>>,
    mem: Arc<Mutex<Ledger>>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    /// how this device computes (chosen at creation from the adapter's capabilities, reported in the load line)
    pub compute: Compute,
    /// the decode kernels reduce with subgroup sums (F16Coop: the subgroup feature, subgroups of 32 — probed); else in
    /// workgroup memory (F16 — wgpu's WebGPU backend does not expose the browser's subgroups yet). Reported in the load line.
    pub subgroups: bool,
    /// why an F32 device computes in f32 (Compute::choose_caps): "no shader-f16", "forced", "workgroup memory 16 KB"
    f32_why: Option<String>,
    kernels: HashMap<&'static str, Kernel>,
    /// the kernels' uniform params: one slot per distinct parameter set of a batch, bound with a dynamic offset (no
    /// buffer per dispatch); written once per batch right before its submit (queue-ordered: an earlier batch still in
    /// flight keeps the values it was submitted with)
    params: Buf,
    param_align: u64,
    /// the weight upload's staging buffer (MAP_WRITE | COPY_SRC, STAGING_BYTES), created once and refilled for every
    /// piece: queue.writeBuffer made a shared-memory copy per call in the page (WebKit) — 1.4 GB of them per read
    staging: Buf,
    /// the memory pool (Pool): the model's weights and the work (activations, scratch, the KV cache, uploads) — every
    /// storage Buf is a slice of it
    pool: Arc<Mutex<Pool>>,
    /// per-kernel GPU time (timestamp queries), natively when OCR_WGPU_PROFILE is set — a measuring tool, off otherwise
    prof: Option<RefCell<Profiler>>,
}

/// the timestamp-query profiler: every dispatch of a batch gets begin / end timestamps (its own compute pass), the
/// batch's submit waits for them and adds the durations to `stats` under "phase/kernel [groups]"
pub struct Profiler {
    qs: wgpu::QuerySet,
    resolve: Buf,
    read: Buf,
    period_ns: f64,
    phase: &'static str,
    /// key → (dispatches, total ns)
    stats: BTreeMap<String, (u64, f64)>,
}

const PROF_QUERIES: u32 = 4096;

/// keys per workgroup of the split decode attention (attn_decode.wgsl; 64 and 128 measured equal on a page)
pub const DECODE_CHUNK: usize = 128;

/// the uniform ring: slots of `param_align` bytes (256 on most GPUs) — a batch's DISTINCT parameter sets
const PARAM_SLOTS: u64 = 1024;
/// the bound size of one parameter slot (the largest Params struct is 8 words)
const PARAM_BYTES: u64 = 32;

/// a compiled kernel: binding 0 = its uniform params (dynamic offset), 1… = storage buffers (read-only or not, as the
/// shader declares them)
struct Kernel {
    pipe: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    n_bufs: usize,
    /// binding i (1…) read-only: a weight binding (weights_read_only) — dispatch binds weights there and nothing else
    read_only: Vec<bool>,
}

impl Gpu {
    /// the compute format for the status / settings line: "f16 + matrix units" (F16Coop), "f16" (F16), "f32 (no
    /// shader-f16)" / "f32 (forced)" / "f32 (workgroup memory 16 KB)" — what the products are computed in, not how the
    /// weights are stored (Engine::weights)
    pub fn compute_label(&self) -> String {
        match (self.compute, &self.f32_why) {
            (Compute::F16Coop, _) => "f16 + matrix units".to_string(),
            (Compute::F16, _) => "f16".to_string(),
            (Compute::F32, Some(why)) => format!("f32 ({why})"),
            (Compute::F32, None) => "f32".to_string(),
        }
    }
}

/// the names of the storage bindings that receive weights (the weight pool's slices) — the only read-only ones
const WEIGHT_BINDINGS: &[&str] = &["W", "Wq", "Wk", "Wv", "Wg", "Wu", "NW", "w", "QN", "KN"];

/// a kernel's WGSL with every `var<storage, read>` binding read_write except those named in WEIGHT_BINDINGS
fn weights_read_only(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + 1024);
    for line in src.split_inclusive('\n') {
        match line.find("var<storage, read> ") {
            Some(i) => {
                let rest = &line[i + "var<storage, read> ".len()..];
                let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                if WEIGHT_BINDINGS.contains(&name.as_str()) {
                    out.push_str(line);
                } else {
                    out.push_str(&line.replacen("var<storage, read> ", "var<storage, read_write> ", 1));
                }
            }
            None => out.push_str(line),
        }
    }
    out
}

/// the storage bindings of a kernel's WGSL: binding i (1…) → read-only? (`var<storage, read>` vs `read_write`)
fn storage_bindings(name: &str, src: &str) -> Result<Vec<bool>> {
    let mut out: Vec<Option<bool>> = Vec::new();
    for line in src.lines() {
        let Some(rest) = line.trim_start().strip_prefix("@group(0) @binding(") else { continue };
        let (idx, decl) = rest.split_once(')').with_context(|| format!("kernel {name}: binding line {line:?}"))?;
        let i: usize = idx.parse().with_context(|| format!("kernel {name}: binding {idx:?}"))?;
        if i == 0 {
            if !decl.contains("var<uniform>") {
                bail!("kernel {name}: binding 0 must be the uniform params");
            }
            continue;
        }
        let ro = if decl.contains("var<storage, read>") {
            true
        } else if decl.contains("var<storage, read_write>") {
            false
        } else {
            bail!("kernel {name}: binding {i} is no storage buffer: {line:?}");
        };
        if out.len() < i {
            out.resize(i, None);
        }
        out[i - 1] = Some(ro);
    }
    out.into_iter().enumerate().map(|(i, b)| b.with_context(|| format!("kernel {name}: binding {} missing", i + 1))).collect()
}

impl Gpu {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    /// the device with the best compute path the adapter offers (OCR_WGPU_COMPUTE = f32 / f16 / coop natively forces one)
    pub async fn new_async() -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        let want = match std::env::var("OCR_WGPU_COMPUTE").as_deref() {
            Ok("f32") => Some(Compute::F32),
            Ok("f16") => Some(Compute::F16),
            Ok("coop") => Some(Compute::F16Coop),
            Ok(other) => bail!("OCR_WGPU_COMPUTE={other}: f32, f16 or coop"),
            Err(_) => None,
        };
        #[cfg(target_arch = "wasm32")]
        let want = None;
        Self::with_compute(want).await
    }

    /// the device with the given compute path (None: the best the adapter offers); a path the adapter lacks is an error
    pub async fn with_compute(want: Option<Compute>) -> Result<Self> {
        let gpu = Self::create(want).await?;
        if gpu.compute == Compute::F16Coop {
            let width = gpu.probe_subgroup_size().await?;
            if width != 32 {
                if want.is_some() {
                    bail!("compute path F16Coop: {} runs subgroups of {width}, the coop kernels need 32", gpu.info.name);
                }
                drop(gpu);
                return Self::create(Some(Compute::F16)).await;
            }
        }
        Ok(gpu)
    }

    /// the subgroup width compute dispatches actually run with (a one-thread-group probe kernel)
    async fn probe_subgroup_size(&self) -> Result<u32> {
        let src = "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
                   @compute @workgroup_size(128) fn main(@builtin(local_invocation_index) li: u32, @builtin(subgroup_size) n: u32) { if (li == 0u) { o[0] = n; } }";
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("subgroup probe"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("subgroup probe"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = scope.pop().await {
            bail!("subgroup probe: {e}");
        }
        let out = self.scratch("subgroup probe", 1);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("subgroup probe"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: out.binding() }],
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("subgroup probe") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("subgroup probe"), timestamp_writes: None });
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit([enc.finish()]);
        Ok(self.read_f32(&out, 1).await?[0].to_bits())
    }

    async fn create(want: Option<Compute>) -> Result<Self> {
        #[cfg(target_arch = "wasm32")]
        let (instance, no_adapter) = {
            // WebGPU only — no WebGL fallback (the kernels are compute shaders)
            let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
            d.backends = wgpu::Backends::BROWSER_WEBGPU;
            (wgpu::Instance::new(d), "this browser offers no WebGPU adapter (OCR Lite needs WebGPU: Chrome / Edge 113+, Safari 26+)")
        };
        #[cfg(not(target_arch = "wasm32"))]
        let (instance, no_adapter) = (wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()), "no wgpu adapter");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
            .await
            .context(no_adapter)?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        #[cfg(not(target_arch = "wasm32"))]
        let profile = std::env::var_os("OCR_WGPU_PROFILE").is_some();
        #[cfg(target_arch = "wasm32")]
        let profile = false;
        if profile && !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            bail!("OCR_WGPU_PROFILE is set but {} has no timestamp queries", info.name);
        }
        let mut required_features = wgpu::Features::empty();
        if profile {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let (compute, f32_why) = Compute::choose(&adapter, &info, want)?;
        required_features |= compute.features();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ocr-wgpu"),
                required_features,
                required_limits: limits.clone(),
                // SAFETY (experimental features: wgpu_cooperative_matrix, the F16Coop path only): the matmul_coop kernel
                // is checked against the f32 kernels by gpu.rs's tests and the vision / read gates
                experimental_features: if compute == Compute::F16Coop { unsafe { wgpu::ExperimentalFeatures::enabled() } } else { wgpu::ExperimentalFeatures::disabled() },
                ..Default::default()
            })
            .await
            .context("wgpu request_device")?;
        let health = Arc::new(Mutex::new(Health::default()));
        let closing = Closing(health.clone());
        {
            let h = health.clone();
            device.set_device_lost_callback(move |reason, message| {
                let (kind, text) = lost_message(reason, &message);
                fatal(&h, kind, text);
            });
        }
        {
            // replaces wgpu's default (a panic natively): the error is the engine's death, reported where it waits next
            let h = health.clone();
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                let (kind, text) = error_message(&e);
                fatal(&h, kind, text);
            }));
        }
        let mem = Arc::new(Mutex::new(Ledger { live: GpuBytes::default(), peak: 0, phase: "-", phases: Vec::new() }));
        let mut kernels = HashMap::new();
        let subgroups = compute == Compute::F16Coop;
        for (name, src) in sources(compute, subgroups) {
            // the memory pools hold many tensors per buffer: a dispatch may read one slice and write another of the same
            // buffer — WebGPU forbids a buffer read-only and writable in one dispatch, but allows writable
            // non-overlapping ranges. So every storage binding is read_write, except the weights' (WEIGHT_BINDINGS):
            // they are only ever read, live in their own pool, and read-only bindings of one buffer never wait on each
            // other (bound writable, every dispatch "wrote" the weights' blocks: a barrier before each, natively 76 →
            // 33 tok/s). Dispatch checks each Buf against its binding. The kernels' maths is unchanged.
            let src = weights_read_only(&src);
            let storage = storage_bindings(name, &src)?;
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let mut entries = vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: None },
                count: None,
            }];
            for (i, &read_only) in storage.iter().enumerate() {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i as u32 + 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                });
            }
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(name), entries: &entries });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(name), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let desc = wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(src.into()) };
            // naga's injected runtime checks — a bounds check per buffer access, a guard counter in every loop (it stops
            // Metal from unrolling) and checked integer division — cost 26 % of the vision encoder natively with
            // bit-identical output (small_a: 915 → 674 ms), the main part of Chrome being faster than native wgpu.
            // So: ON in tests and debug builds (an out-of-range index in our WGSL stays contained there),
            // OFF in release builds only. In the browser Tint applies its own robustness either way.
            let module = if SHADER_CHECKS {
                device.create_shader_module(desc)
            } else {
                // SAFETY: every loop of the kernels ends (bounds from the params), no divisor is zero (n_head / n_kv are
                // checked at load), every buffer index is guarded or clamped in the kernel — exercised with the checks
                // on by cargo test and debug builds
                unsafe { device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) }
            };
            let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            if let Some(e) = scope.pop().await {
                bail!("kernel {name}: {e}");
            }
            kernels.insert(name, Kernel { pipe, layout, n_bufs: storage.len(), read_only: storage.clone() });
        }
        let param_align = (limits.min_uniform_buffer_offset_alignment as u64).max(PARAM_BYTES);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: PARAM_SLOTS * param_align,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = counted(&mem, params, MemKind::Work);
        let prof = profile.then(|| {
            let buf = |label, usage| counted(&mem, device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: PROF_QUERIES as u64 * 8, usage, mapped_at_creation: false }), MemKind::Readback);
            RefCell::new(Profiler {
                qs: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("prof"), ty: wgpu::QueryType::Timestamp, count: PROF_QUERIES }),
                resolve: buf("prof resolve", wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
                read: buf("prof read", wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
                period_ns: queue.get_timestamp_period() as f64,
                phase: "-",
                stats: BTreeMap::new(),
            })
        });
        #[allow(unused_mut)]
        let mut pool = Pool::new(&limits);
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var("OCR_WGPU_POISON").as_deref() == Ok("1") {
            pool.poison = Some(queue.clone());
        }
        let pool = Arc::new(Mutex::new(pool));
        let staging = counted(
            &mem,
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("upload staging"), size: STAGING_BYTES, usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
            MemKind::Readback,
        );
        let gpu = Self { _closing: closing, health, mem, device, queue, info, limits, compute, subgroups, f32_why, kernels, params, param_align, staging, pool, prof };
        gpu.warm_up().await?;
        Ok(gpu)
    }

    /// one tiny dispatch per pipeline at device creation (part of the model load), so nothing of a pipeline's first use
    /// (a driver's deferred compile / residency) lands in the first read (without it the first browser read was
    /// ~10 s slower than the second). All params 0 (every kernel's loops and writes are bounded by them: no work),
    /// one group, each binding its own zeroed 1 MB buffer (no read-only / writable aliasing).
    async fn warm_up(&self) -> Result<()> {
        // its own 1 MB slice per binding: work slices at the writable bindings, weight slices (zeros) at the read-only
        // weight bindings (dispatch checks the kinds)
        let n_bufs = self.kernels.values().map(|k| k.n_bufs).max().unwrap_or(0);
        let work: Vec<Buf> = (0..n_bufs).map(|_| self.scratch("warm-up", 1 << 18)).collect();
        let weights: Vec<Buf> = (0..n_bufs).map(|_| self.upload_weight("warm-up", &vec![0u8; 1 << 20])).collect();
        let mut names: Vec<&'static str> = self.kernels.keys().copied().collect();
        names.sort_unstable();
        self.phase("warm-up");
        let mut b = self.batch();
        for name in names {
            let k = &self.kernels[name];
            let refs: Vec<&Buf> = (0..k.n_bufs).map(|i| if k.read_only[i] { &weights[i] } else { &work[i] }).collect();
            b.dispatch(name, &[0; 8], &refs, (1, 1, 1))?;
        }
        b.submit();
        self.wait().await?;
        self.phase("-");
        Ok(())
    }

    /// the phase of the following work (vision / prefill / decode; "-" = none): the memory ledger's per-phase peaks and
    /// the profiler's label for the following batches
    pub fn phase(&self, name: &'static str) {
        {
            let mut l = lock(&self.mem);
            l.phase = name;
            l.note();
        }
        if let Some(p) = &self.prof {
            p.borrow_mut().phase = name;
        }
    }

    /// the bytes of the live buffers now, by kind
    pub fn mem_live(&self) -> GpuBytes {
        lock(&self.mem).live
    }

    /// the most bytes alive at once since the device was created or the last mem_begin_load
    pub fn mem_peak(&self) -> u64 {
        lock(&self.mem).peak
    }

    /// a weight load starts: the peak restarts from the bytes alive now (ModelMemory.load_peak = that load's)
    pub fn mem_begin_load(&self) {
        let mut l = lock(&self.mem);
        l.peak = l.live.total();
    }

    /// a read starts: its phase peaks start empty
    pub fn mem_begin_read(&self) {
        lock(&self.mem).phases.clear();
    }

    /// the read's memory since mem_begin_read: `resident` = the bytes alive at its start
    pub fn mem_read(&self, resident: u64) -> Result<ReadMemory> {
        let l = lock(&self.mem);
        let Some(top) = l.phases.iter().max_by_key(|p| p.peak.total()) else {
            bail!("the memory ledger saw no read phase");
        };
        Ok(ReadMemory { resident_bytes: resident, peak_bytes: top.peak.total(), peak_phase: top.phase, phases: l.phases.clone() })
    }

    /// Err(why) once the device is lost or a GPU error went uncaught: the engine is dead, every further use fails
    pub fn alive(&self) -> Result<()> {
        match &lock(&self.health).dead {
            Some(why) => Err(anyhow!("{why}")),
            None => Ok(()),
        }
    }

    /// a buffer of `bytes` (a weight bound whole to a kernel) against this device's limits — the load's pre-check, before
    /// anything is uploaded; Err names the limit and both numbers
    pub fn check_fits(&self, what: &str, bytes: u64) -> Result<()> {
        let l = &self.limits;
        if bytes > l.max_buffer_size {
            bail!("{what} needs a {:.1} MB GPU buffer, this GPU allows {:.1} MB (maxBufferSize) — OCR Lite cannot run on it", mb(bytes), mb(l.max_buffer_size));
        }
        if bytes > l.max_storage_buffer_binding_size {
            bail!(
                "{what} needs a {:.1} MB storage binding, this GPU binds at most {:.1} MB (maxStorageBufferBindingSize) — OCR Lite cannot run on it",
                mb(bytes),
                mb(l.max_storage_buffer_binding_size)
            );
        }
        Ok(())
    }

    /// the profiler's table (per phase, kernels by total time) and reset; None without the profiler
    pub fn profile_report(&self) -> Option<String> {
        let p = self.prof.as_ref()?;
        let stats = std::mem::take(&mut p.borrow_mut().stats);
        let mut phases: BTreeMap<&str, Vec<(&str, u64, f64)>> = BTreeMap::new();
        for (k, &(n, ns)) in &stats {
            let (ph, kernel) = k.split_once('/').unwrap_or(("-", k));
            phases.entry(ph).or_default().push((kernel, n, ns));
        }
        let mut out = String::new();
        for (ph, mut rows) in phases {
            rows.sort_by(|a, b| b.2.total_cmp(&a.2));
            let total: f64 = rows.iter().map(|r| r.2).sum();
            let calls: u64 = rows.iter().map(|r| r.1).sum();
            out.push_str(&format!("{ph}: GPU {:.1} ms in {calls} dispatches\n", total / 1e6));
            for (k, n, ns) in rows.iter().take(14) {
                out.push_str(&format!("  {:>9.2} ms {:>5.1}% {:>7} × {:>8.1} µs  {k}\n", ns / 1e6, 100.0 * ns / total, n, ns / 1e3 / *n as f64));
            }
        }
        Some(out)
    }

    /// completes when the GPU has finished all work submitted so far (natively: poll until idle)
    pub async fn wait(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        self.settle(rx).await
    }

    /// await a wgpu callback's channel; natively the callback only runs inside device.poll
    async fn settle<T>(&self, rx: oneshot::Receiver<T>) -> Result<T> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(e) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
            self.alive()?;
            return Err(e).context("wgpu poll");
        }
        self.race(rx).await
    }

    /// a wgpu callback's channel or the device's death, whichever comes first (a lost device may never call back);
    /// a dead device's reason wins over the callback's own result
    async fn race<T>(&self, mut rx: oneshot::Receiver<T>) -> Result<T> {
        let mut lost = {
            let mut h = lock(&self.health);
            if let Some(why) = &h.dead {
                bail!("{why}");
            }
            h.waiters.retain(|w| !w.is_canceled());
            let (tx, lost) = oneshot::channel();
            h.waiters.push(tx);
            lost
        };
        let r = std::future::poll_fn(|cx| {
            if let Poll::Ready(v) = Pin::new(&mut rx).poll(cx) {
                return Poll::Ready(Some(v));
            }
            if Pin::new(&mut lost).poll(cx).is_ready() {
                return Poll::Ready(None);
            }
            Poll::Pending
        })
        .await;
        self.alive()?;
        match r {
            Some(Ok(v)) => Ok(v),
            Some(Err(_)) => bail!("wgpu callback dropped"),
            None => bail!("the GPU wait was woken without a recorded death"),
        }
    }

    /// a map_async result: a failed map marks the device dead (suspect) — the error is the current reason
    fn map_result(&self, r: std::result::Result<(), wgpu::BufferAsyncError>) -> Result<()> {
        let Err(e) = r else { return Ok(()) };
        suspect(&self.health, format!("GPU readback failed ({e}) — most likely the GPU device was lost"));
        match &lock(&self.health).dead {
            Some(why) => bail!("{why}"),
            None => bail!("GPU readback failed ({e})"),
        }
    }

    fn count(&self, buf: wgpu::Buffer, kind: MemKind) -> Buf {
        counted(&self.mem, buf, kind)
    }

    /// a buffer of `size` bytes, counted as `kind`. Zeroed on the GPU here (all but weights, which the upload writes
    /// whole, their pad zeroed on the CPU), not trusted to the API: WebGPU promises zero-initialised buffers, but on iOS a fresh buffer
    /// came back with an earlier read's leftovers (the same page read once right, then "000…" or random characters —
    /// the vision kernels read the padding rows of their 64-row tiles, NaN there spreads through the attention). A
    /// clear is a GPU memset, submitted before any batch that uses the buffer (the queue keeps the order).
    /// Storage buffers (STORAGE | COPY_SRC | COPY_DST) are slices of a pool (Pool: weights, or work for the rest); other
    /// usages (readback, the uniform params) get a buffer of their own.
    fn alloc(&self, label: &str, size: u64, usage: wgpu::BufferUsages, kind: MemKind) -> Buf {
        if usage != POOL_USAGE {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
            if kind != MemKind::Weights && usage.contains(wgpu::BufferUsages::COPY_DST) {
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zero a new buffer") });
                enc.clear_buffer(&buf, 0, None);
                self.queue.submit([enc.finish()]);
            }
            return self.count(buf, kind);
        }
        let role = if kind == MemKind::Weights { Role::Weights } else { Role::Work };
        let taken = size.max(4).div_ceil(4) * 4;
        let pool = &self.pool;
        // the fixed pool is sized for the largest read (Engine checks the image against MAX_IMAGE_TOKENS before a read):
        // no room is a sizing bug — a hard stop with the numbers, never a new block
        let taken_slot = lock(pool).take(&self.device, role, taken);
        let (buf, offset, block) = match taken_slot {
            Ok(t) => t,
            Err(e) => panic!("{label}: {e:#}"),
        };
        if kind != MemKind::Weights {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zero a new slice") });
            enc.clear_buffer(&buf, offset, Some(taken));
            self.queue.submit([enc.finish()]);
        }
        let mut l = lock(&self.mem);
        *kind.slot(&mut l.live) += size;
        l.note();
        drop(l);
        Buf { c: Arc::new(Counted { buf, offset, size, kind, ledger: self.mem.clone(), home: Some((pool.clone(), block, taken)) }), off: offset, len: size }
    }

    /// `bytes` in a new work slice (padded to 4 bytes with zeros)
    pub fn upload(&self, label: &str, bytes: &[u8]) -> Buf {
        self.upload_as(label, bytes, MemKind::Work)
    }

    /// `bytes` as a weight (the weight pool; bound read-only) — the tests' weights (a model's come by upload_tensor)
    pub fn upload_weight(&self, label: &str, bytes: &[u8]) -> Buf {
        self.upload_as(label, bytes, MemKind::Weights)
    }

    pub fn upload_weight_f32(&self, label: &str, v: &[f32]) -> Buf {
        self.upload_weight(label, bytemuck::cast_slice(v))
    }

    fn upload_as(&self, label: &str, bytes: &[u8], kind: MemKind) -> Buf {
        let dst = self.alloc(label, bytes.len() as u64, POOL_USAGE, kind);
        if bytes.len() % 4 == 0 {
            self.queue.write_buffer(dst.buffer(), dst.offset(), bytes);
        } else {
            let mut padded = bytes.to_vec();
            padded.resize(bytes.len().div_ceil(4) * 4, 0);
            self.queue.write_buffer(dst.buffer(), dst.offset(), &padded);
        }
        dst
    }

    /// the bytes the memory pool holds on the GPU (its blocks): what the device keeps for the engine, slices in use or not
    pub fn pool_bytes(&self) -> u64 {
        lock(&self.pool).bytes()
    }

    /// the pool at its full size, once (Engine::open): POOL_SMALL_BLOCKS blocks of 64 MiB + one large block of `large`
    /// bytes; from now on it never grows
    pub fn fix_pool(&self, small: usize, large: u64) -> Result<()> {
        lock(&self.pool).fix(&self.device, small, large)
    }

    /// (the most small pool blocks in use at once so far, the small blocks there are) — POOL_SMALL_BLOCKS' measure
    pub fn pool_small_blocks(&self) -> (usize, usize) {
        let p = lock(&self.pool);
        (p.peak_small, p.blocks.iter().filter(|b| b.len == p.block).count())
    }

    pub fn upload_f32(&self, label: &str, v: &[f32]) -> Buf {
        self.upload(label, bytemuck::cast_slice(v))
    }

    /// the reusable staging chunk of the weight upload (size rounded up to 4 bytes)
    pub fn uploader(&self, bytes: u64) -> Uploader {
        let size = (bytes.max(4).div_ceil(4) * 4).min(self.staging.size());
        Uploader { size, chunk: vec![0u8; size as usize], mapped: false, fill: 0, copies: Vec::new(), converts: Vec::new(), read_ms: 0.0, wait_ms: 0.0 }
    }

    /// stream a tensor's file range into a new GPU buffer through the staging chunk (the buffer is padded to 4 bytes,
    /// the pad zeroed)
    pub async fn upload_tensor(&self, up: &mut Uploader, src: &dyn TensorSource, t: &TensorInfo) -> Result<Buf> {
        let padded = tensor_buffer_bytes(t);
        if padded > self.limits.max_buffer_size {
            bail!("tensor {} needs a {padded}-byte GPU buffer, this GPU allows {} (maxBufferSize)", t.name, self.limits.max_buffer_size);
        }
        let len = t.n_bytes();
        let dst = self.alloc(&t.name, padded, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, MemKind::Weights);
        let mut pos = 0u64;
        while pos < padded {
            // the file range straight into the mapped staging buffer — several tensors per fill (~45 GPU round trips for
            // 1.4 GB, not one per tensor); copied into the pool slices when it is full or the load ends
            let piece = (padded - pos).min(up.size);
            if up.fill + piece > up.size {
                self.upload_submit(up);
            }
            if !up.mapped {
                let t1 = Instant::now();
                let (tx, rx) = oneshot::channel();
                self.staging.buffer().slice(0..up.size).map_async(wgpu::MapMode::Write, move |r| {
                    let _ = tx.send(r);
                });
                self.map_result(self.settle(rx).await?)?;
                up.mapped = true;
                up.fill = 0;
                up.wait_ms += t1.elapsed().as_secs_f64() * 1e3;
            }
            let real = (len.saturating_sub(pos)).min(piece) as usize;
            let at = up.fill;
            let buf = &mut up.chunk[..piece as usize];
            let t0 = Instant::now();
            src.read_at(t.file_offset + pos, &mut buf[..real]).with_context(|| format!("tensor {}", t.name))?;
            up.read_ms += t0.elapsed().as_secs_f64() * 1e3;
            buf[real..].fill(0);
            self.staging.buffer().slice(at..at + piece).get_mapped_range_mut().context("map the upload staging buffer")?.copy_from_slice(buf);
            up.copies.push((at, dst.clone(), pos, piece));
            up.fill = (at + piece).div_ceil(COPY_ALIGN) * COPY_ALIGN;
            pos += piece;
        }
        // the copies wait in the staging buffer until it is full (one GPU round trip per 32 MiB, not per tensor): the
        // tensor is on the GPU after upload_flush; a conversion of it goes through weight_to_f16_uploaded
        Ok(dst)
    }

    /// the staging buffer's contents to their pool slices: unmapped, the copies submitted (the next map waits for them)
    fn upload_submit(&self, up: &mut Uploader) {
        if !up.mapped {
            return;
        }
        self.staging.buffer().unmap();
        up.mapped = false;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("weight upload") });
        for (at, dst, pos, n) in up.copies.drain(..) {
            enc.copy_buffer_to_buffer(self.staging.buffer(), at, dst.buffer(), dst.offset() + pos, n);
        }
        self.queue.submit([enc.finish()]);
        for (w, over, scale) in up.converts.drain(..) {
            let words = w.size() / 4;
            let mut b = self.batch();
            // the arguments were checked when the conversion was queued (weight_to_f16_uploaded)
            if let Err(e) = b.dispatch("f16_from_bf16", &[words as u32, scale.to_bits()], &[&w, &over], (stride_groups(words), 1, 1)) {
                panic!("f16_from_bf16: {e:#}");
            }
            b.submit();
        }
    }

    /// the load's last copies submitted and taken by the GPU (the end of a load)
    pub async fn upload_flush(&self, up: &mut Uploader) -> Result<()> {
        self.upload_submit(up);
        self.wait().await?;
        Ok(())
    }

    /// a bf16 weight converted to f16 in place, times `scale` (a power of two: exact) — the f16 paths,
    /// f16_from_bf16.wgsl; words holding values beyond f16's range are counted into `over` (one u32, checked by the
    /// caller once all weights are converted)
    pub fn weight_to_f16(&self, w: &mut Weight, over: &Buf, scale: f32) -> Result<()> {
        if scale.to_bits() & 0x007f_ffff != 0 || scale <= 0.0 {
            bail!("weight_to_f16: scale {scale} is not a power of two");
        }
        if w.ty != WeightType::Bf16 {
            bail!("weight_to_f16: a {:?} weight", w.ty);
        }
        let words = w.buf.size() / 4;
        let mut b = self.batch();
        b.dispatch("f16_from_bf16", &[words as u32, scale.to_bits()], &[&w.buf, over], (stride_groups(words), 1, 1))?;
        b.submit();
        w.ty = WeightType::F16;
        Ok(())
    }

    /// weight_to_f16 for a weight just streamed through `up` (upload_tensor): the conversion is submitted right after the
    /// weight's last copies (upload_submit); the weight is f16 from here on for what is recorded after upload_flush
    pub fn weight_to_f16_uploaded(&self, up: &mut Uploader, w: &mut Weight, over: &Buf, scale: f32) -> Result<()> {
        if scale.to_bits() & 0x007f_ffff != 0 || scale <= 0.0 {
            bail!("weight_to_f16: scale {scale} is not a power of two");
        }
        if w.ty != WeightType::Bf16 {
            bail!("weight_to_f16: a {:?} weight", w.ty);
        }
        if !self.kernels.contains_key("f16_from_bf16") {
            bail!("weight_to_f16: no kernel f16_from_bf16 on the {:?} path", self.compute);
        }
        up.converts.push((w.buf.clone(), over.clone(), scale));
        w.ty = WeightType::F16;
        Ok(())
    }

    /// an uninitialised f32 buffer of n elements
    pub fn scratch(&self, label: &str, n: usize) -> Buf {
        self.scratch_as(label, n, MemKind::Work)
    }

    /// an uninitialised f32 buffer of n elements, counted as `kind` (the KV cache)
    pub fn scratch_as(&self, label: &str, n: usize, kind: MemKind) -> Buf {
        self.alloc(label, (n.max(1) * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, kind)
    }

    /// the first n f32 of a buffer back to the CPU (waits for all submitted work)
    pub async fn read_f32(&self, buf: &Buf, n: usize) -> Result<Vec<f32>> {
        let size = (n * 4) as u64;
        let staging = self.alloc("readback", size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, MemKind::Readback);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_buffer_to_buffer(buf.buffer(), buf.offset(), staging.buffer(), staging.offset(), size);
        self.queue.submit([enc.finish()]);
        let (tx, rx) = oneshot::channel();
        staging.slice().map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.map_result(self.settle(rx).await?)?;
        let view = staging.slice().get_mapped_range().context("map readback")?;
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        staging.buffer().unmap();
        Ok(out)
    }

    /// a buffer the CPU reads GPU results from (a batch copies into it, `read_mapped` maps it)
    pub fn readback(&self, label: &str, n: usize) -> Buf {
        self.alloc(label, (n.max(1) * 4) as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, MemKind::Readback)
    }

    /// the first n u32 of a readback buffer once submission `idx` (the one that copied them there) has finished —
    /// natively it polls for that submission only, so the work queued after it keeps the GPU busy meanwhile (the
    /// run-ahead decode); the map is requested here, so the buffer must not be in use by a later submission
    pub async fn read_mapped_u32(&self, buf: &Buf, n: usize, idx: wgpu::SubmissionIndex) -> Result<Vec<u32>> {
        let slice = buf.buffer().slice(buf.offset()..buf.offset() + (n * 4) as u64);
        let (tx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(e) = self.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }) {
            self.alive()?;
            return Err(e).context("wgpu poll");
        }
        #[cfg(target_arch = "wasm32")]
        let _ = idx;
        self.map_result(self.race(rx).await?)?;
        let view = slice.get_mapped_range().context("map readback")?;
        let out = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        buf.buffer().unmap();
        Ok(out)
    }

    /// the first n f32 of a readback buffer, once the GPU has written them (waits for all submitted work)
    pub async fn read_mapped(&self, buf: &Buf, n: usize) -> Result<Vec<f32>> {
        let slice = buf.buffer().slice(buf.offset()..buf.offset() + (n * 4) as u64);
        let (tx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.map_result(self.settle(rx).await?)?;
        let view = slice.get_mapped_range().context("map readback")?;
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        buf.buffer().unmap();
        Ok(out)
    }

    pub fn batch(&self) -> Batch<'_> {
        self.batch_with(None)
    }

    /// a batch whose bind groups are kept in `cache` (the decode step: the same dispatches every token)
    pub fn batch_cached<'a>(&'a self, cache: &'a BindCache) -> Batch<'a> {
        self.batch_with(Some(cache))
    }

    fn batch_with<'a>(&'a self, cache: Option<&'a BindCache>) -> Batch<'a> {
        Batch {
            gpu: self,
            enc: self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch") }),
            pass: None,
            words: Vec::new(),
            slots: HashMap::new(),
            cache,
            prof: Vec::new(),
            written: 0,
            in_flight: 0,
        }
    }
}

/// a weight load's progress through the staging buffer (Gpu::staging): mapped (for writing) or not, how far it is
/// filled, and the copies its contents wait for — (staging offset, destination, destination offset, bytes)
pub struct Uploader {
    size: u64,
    /// the file range read here first (a mapped view is write-only), then copied into the staging buffer
    chunk: Vec<u8>,
    mapped: bool,
    fill: u64,
    copies: Vec<(u64, Buf, u64, u64)>,
    /// bf16 → f16 conversions (weight, over-range counter, scale) waiting for their weight's copies: submitted right after
    /// them (upload_submit — the queue keeps the order)
    converts: Vec<(Buf, Buf, f32)>,
    /// the time spent reading the model files (TensorSource::read_at) — the load's I/O part
    pub read_ms: f64,
    /// the time spent waiting for the staging buffer (its map) and the GPU's copies
    pub wait_ms: f64,
}

/// bind groups kept across batches: (kernel, bound ranges — buffer, offset, size) → bind group. It holds the buffers alive
/// — owned by what owns those buffers (a decode session) and dropped with it.
#[derive(Default)]
pub struct BindCache(RefCell<HashMap<(&'static str, Vec<(wgpu::Buffer, u64, u64)>), wgpu::BindGroup>>);

impl BindCache {
    /// drops the bind groups that bind any of `bufs` (buffers being replaced — the KV cache grown): kept, they would
    /// hold the old buffers alive past their Buf (and the ledger)
    pub fn forget(&self, bufs: &[&Buf]) {
        self.0.borrow_mut().retain(|(_, bound), _| !bound.iter().any(|(b, o, _)| bufs.iter().any(|x| x.buffer() == b && x.offset() == *o)));
    }
}

/// dispatches recorded into ONE compute pass (a buffer copy ends the pass, the next dispatch opens a new one); WebGPU
/// orders the dispatches of a pass like separate passes (each dispatch is its own usage scope)
pub struct Batch<'a> {
    gpu: &'a Gpu,
    enc: wgpu::CommandEncoder,
    pass: Option<wgpu::ComputePass<'static>>,
    /// the batch's parameter slots (param_align / 4 words each), written to gpu.params at submit
    words: Vec<u32>,
    /// a parameter set → its byte offset in gpu.params
    slots: HashMap<Vec<u32>, u32>,
    cache: Option<&'a BindCache>,
    /// the profiler's key per timed dispatch (query pair i)
    prof: Vec<String>,
    /// the words of `words` already written to gpu.params (flush writes only the new ones: a slot a queued command
    /// buffer reads is never rewritten by its own recording)
    written: usize,
    /// command buffers submitted by flush since the last throttle wait
    in_flight: usize,
}

impl Batch<'_> {
    /// one dispatch: params (u32 / f32 words, at most 8) at binding 0, `bufs` at bindings 1…
    pub fn dispatch(&mut self, kernel: &str, params: &[u32], bufs: &[&Buf], groups: (u32, u32, u32)) -> Result<()> {
        let gpu = self.gpu;
        let (&name, k) = gpu.kernels.get_key_value(kernel).with_context(|| format!("no kernel {kernel}"))?;
        if bufs.len() != k.n_bufs {
            bail!("kernel {kernel}: {} buffers given, the shader binds {}", bufs.len(), k.n_bufs);
        }
        let align = gpu.limits.min_storage_buffer_offset_alignment as u64;
        for (i, b) in bufs.iter().enumerate() {
            if b.offset() % align != 0 {
                bail!("kernel {kernel}: binding {} starts at byte {} — the device binds storage at multiples of {align}", i + 1, b.offset());
            }
        }
        // weights only at the read-only (weight) bindings, nothing else there — the conversion at load writes weights
        for (i, b) in bufs.iter().enumerate() {
            let weight = b.kind() == MemKind::Weights;
            if weight != k.read_only[i] && kernel != "f16_from_bf16" {
                bail!("kernel {kernel}: binding {} ({}) gets {} — weights belong at the read-only weight bindings (WEIGHT_BINDINGS), nothing else", i + 1, if k.read_only[i] { "read-only" } else { "writable" }, if weight { "a weight" } else { "no weight" });
            }
        }
        if params.len() as u64 * 4 > PARAM_BYTES {
            bail!("kernel {kernel}: {} param words, a slot holds {}", params.len(), PARAM_BYTES / 4);
        }
        let offset = match self.slots.get(params) {
            Some(&o) => o,
            None => {
                let slot_words = (gpu.param_align / 4) as usize;
                let o = self.words.len() as u64 * 4;
                if o + gpu.param_align > PARAM_SLOTS * gpu.param_align {
                    bail!("a batch with more than {PARAM_SLOTS} distinct parameter sets");
                }
                self.words.extend_from_slice(params);
                self.words.resize(self.words.len() + slot_words - params.len(), 0);
                self.slots.insert(params.to_vec(), o as u32);
                o as u32
            }
        };
        let make = || -> Result<wgpu::BindGroup> {
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: gpu.params.buffer(), offset: gpu.params.offset(), size: wgpu::BufferSize::new(PARAM_BYTES) }),
            }];
            let max = gpu.limits.max_storage_buffer_binding_size;
            for (i, b) in bufs.iter().enumerate() {
                if b.size() > max {
                    bail!(
                        "kernel {kernel}: a {} MB buffer is bound, this GPU binds at most {} MB (maxStorageBufferBindingSize) — the image is too large for it",
                        b.size() >> 20,
                        max >> 20
                    );
                }
                entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.binding() });
            }
            Ok(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(kernel), layout: &k.layout, entries: &entries }))
        };
        let bg = match self.cache {
            Some(c) => {
                let key = (name, bufs.iter().map(|b| (b.buffer().clone(), b.offset(), b.size())).collect::<Vec<_>>());
                let mut map = c.0.borrow_mut();
                match map.get(&key) {
                    Some(bg) => bg.clone(),
                    None => {
                        let bg = make()?;
                        map.insert(key, bg.clone());
                        bg
                    }
                }
            }
            None => make()?,
        };
        if let Some(p) = &gpu.prof {
            // profiling: every dispatch in its own pass with begin / end timestamps
            self.pass = None;
            let i = self.prof.len() as u32;
            let p = p.borrow();
            let timestamp_writes = (i < PROF_QUERIES / 2).then(|| wgpu::ComputePassTimestampWrites {
                query_set: &p.qs,
                beginning_of_pass_write_index: Some(2 * i),
                end_of_pass_write_index: Some(2 * i + 1),
            });
            if timestamp_writes.is_some() {
                self.prof.push(format!("{}/{kernel} {}x{}x{}", p.phase, groups.0, groups.1, groups.2));
            }
            let mut pass = self.enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(kernel), timestamp_writes });
            pass.set_pipeline(&k.pipe);
            pass.set_bind_group(0, &bg, &[offset]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            return Ok(());
        }
        let pass = match &mut self.pass {
            Some(p) => p,
            None => self.pass.insert(self.enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("batch"), timestamp_writes: None }).forget_lifetime()),
        };
        pass.set_pipeline(&k.pipe);
        pass.set_bind_group(0, &bg, &[offset]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        Ok(())
    }

    /// Y[M,N] (+)= X[M,K] · W[N,K]ᵀ with W bf16 (packed) or f32
    pub fn matmul(&mut self, x: &Buf, w: &Weight, y: &Buf, m: usize, accumulate: bool) -> Result<()> {
        self.matmul_p(x, w, y, m, accumulate, Precision::F32)
    }

    pub fn matmul_p(&mut self, x: &Buf, w: &Weight, y: &Buf, m: usize, accumulate: bool, prec: Precision) -> Result<()> {
        let xround = match (prec, w.ty) {
            (Precision::F32, _) => 0u32,
            (Precision::LlamaMetal, WeightType::Bf16) => 1,
            (Precision::LlamaMetal, WeightType::F32 | WeightType::Q8_0) => 2,
            (Precision::LlamaMetal, WeightType::F16) => bail!("the llama.cpp precision bisect emulates bf16 weights: run it on the f32 path (OCR_WGPU_COMPUTE=f32)"),
        };
        let (n, k) = (w.rows as u32, w.cols as u32);
        let params = [m as u32, n, k, accumulate as u32, xround];
        if k % 4 == 0 && self.gpu.compute == Compute::F16Coop && prec == Precision::F32 {
            let kernel = match w.ty {
                WeightType::Bf16 => "matmul_coop_bf16",
                WeightType::F16 => "matmul_coop_f16",
                WeightType::F32 => "matmul_coop_f32",
                WeightType::Q8_0 => "matmul_coop_q8_0",
            };
            return self.dispatch(kernel, &params, &[x, &w.buf, y], (n.div_ceil(64), (m as u32).div_ceil(64), 1));
        }
        if k % 4 == 0 && self.gpu.compute == Compute::F16 && prec == Precision::F32 {
            let kernel = match w.ty {
                WeightType::Bf16 => "matmul128h_bf16",
                WeightType::F16 => "matmul128h_f16",
                WeightType::F32 => "matmul128h_f32",
                WeightType::Q8_0 => "matmul128h_q8_0",
            };
            let (xrow, yrow) = (k as u64 * 4, n as u64 * 4);
            return self.in_row_blocks(m, |b, r0, rows| {
                let (xv, yv) = (x.view(r0 as u64 * xrow, rows as u64 * xrow)?, y.view(r0 as u64 * yrow, rows as u64 * yrow)?);
                b.dispatch(kernel, &[rows as u32, n, k, accumulate as u32, xround], &[&xv, &w.buf, &yv], (n.div_ceil(128), (rows as u32).div_ceil(128), 1))
            });
        }
        if k % 4 == 0 {
            let kernel = match w.ty {
                WeightType::Bf16 => "matmul128_bf16",
                WeightType::F16 => "matmul128_f16",
                WeightType::F32 => "matmul128_f32",
                WeightType::Q8_0 => "matmul128_q8_0",
            };
            return self.dispatch(kernel, &params, &[x, &w.buf, y], (n.div_ceil(128), (m as u32).div_ceil(128), 1));
        }
        let kernel = match w.ty {
            WeightType::Bf16 => "matmul_bf16",
            WeightType::F16 => "matmul_f16",
            WeightType::F32 => "matmul_f32",
            WeightType::Q8_0 => "matmul_q8_0",
        };
        self.dispatch(kernel, &params, &[x, &w.buf, y], (n.div_ceil(64), (m as u32).div_ceil(64), 1))
    }

    /// Y[M,N] (+)= X[M,K] · W[N,K]ᵀ with X f16 (an f16-writing producer) and W f16 (converted at load): the vision
    /// encoder on the f16 paths. F16Coop: matmul_coopd (direct matrix-unit loads; X / Y hold whole 64-row tiles);
    /// F16: matmul128hx_f16 (f16 tiles, f32 fma)
    pub fn matmul_f16x(&mut self, x: &Buf, w: &Weight, y: &Buf, m: usize, accumulate: bool) -> Result<()> {
        let (n, k) = (w.rows, w.cols);
        if w.ty != WeightType::F16 || k % 8 != 0 || n % 64 != 0 {
            bail!("matmul_f16x: {:?} [{n} × {k}] — f16 weights, K % 8 == 0, N % 64 == 0", w.ty);
        }
        let params = [m as u32, n as u32, k as u32, accumulate as u32];
        match self.gpu.compute {
            Compute::F16Coop => {
                let mp = m.div_ceil(64) * 64;
                if x.size() < (mp * k * 2) as u64 || y.size() < (mp * n * 4) as u64 {
                    bail!("matmul_f16x: X / Y must hold {mp} rows (whole 64-row tiles)");
                }
                self.dispatch("matmul_coopd", &params, &[x, &w.buf, y], ((n / 64) as u32, (mp / 64) as u32, 1))
            }
            Compute::F16 => {
                let (xrow, yrow) = (k as u64 * 2, n as u64 * 4);
                self.in_row_blocks(m, |b, r0, rows| {
                    let (xv, yv) = (x.view(r0 as u64 * xrow, rows as u64 * xrow)?, y.view(r0 as u64 * yrow, rows as u64 * yrow)?);
                    b.dispatch("matmul128hx_f16", &[rows as u32, n as u32, k as u32, accumulate as u32, 0], &[&xv, &w.buf, &yv], ((n as u32).div_ceil(128), (rows as u32).div_ceil(128), 1))
                })
            }
            Compute::F32 => bail!("matmul_f16x on the f32 path"),
        }
    }

    /// G[M,F] = silu(X·Wgᵀ) · (X·Wuᵀ) · out_scale as f16 — the vision FFN's first half on the f16 paths. F16Coop: one fused kernel
    /// (ffn_coopd, no f32 gate / up buffers: `gate` / `up` unused); F16: the two products into `gate` / `up` (f32) and
    /// swiglu_h
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_f16(&mut self, x: &Buf, wg: &Weight, wu: &Weight, g16: &Buf, m: usize, out_scale: f32, gate: &Buf, up: &Buf) -> Result<()> {
        let (f, k) = (wg.rows, wg.cols);
        if wu.rows != f || wu.cols != k {
            bail!("ffn_f16: gate [{f} × {k}] vs up [{} × {}]", wu.rows, wu.cols);
        }
        match self.gpu.compute {
            Compute::F16Coop => {
                let mp = m.div_ceil(64) * 64;
                if wg.ty != WeightType::F16 || wu.ty != WeightType::F16 || k % 8 != 0 || f % 32 != 0 {
                    bail!("ffn_f16: {:?} [{f} × {k}] — f16 weights, K % 8 == 0, F % 32 == 0", wg.ty);
                }
                if x.size() < (mp * k * 2) as u64 || g16.size() < (mp * f * 2) as u64 {
                    bail!("ffn_f16: X / G must hold {mp} rows (whole 64-row tiles)");
                }
                self.dispatch("ffn_coopd", &[m as u32, f as u32, k as u32, out_scale.to_bits()], &[x, &wg.buf, &wu.buf, g16], ((f / 32) as u32, (mp / 64) as u32, 1))
            }
            Compute::F16 => {
                // FFN_ROWS rows at a time: `gate` / `up` hold one block (f32) — for all rows at once they were ~300 MiB of
                // the iPhone's 2048 (a full page's image encode 1453 MiB in use). The same products, in row blocks.
                let block = FFN_ROWS.min(m);
                if gate.size() < (block * f * 4) as u64 || up.size() < (block * f * 4) as u64 {
                    bail!("ffn_f16: gate / up must hold {block} rows of {f}");
                }
                for r0 in (0..m).step_by(FFN_ROWS) {
                    let rows = FFN_ROWS.min(m - r0);
                    let xv = x.view((r0 * k * 2) as u64, (rows * k * 2) as u64)?;
                    let gv = g16.view((r0 * f * 2) as u64, (rows * f * 2) as u64)?;
                    self.matmul_f16x(&xv, wg, gate, rows, false)?;
                    self.matmul_f16x(&xv, wu, up, rows, false)?;
                    self.dispatch("swiglu_h", &[(rows * f) as u32, out_scale.to_bits()], &[up, gate, &gv], (stride_groups((rows * f) as u64), 1, 1))?;
                    if (r0 + rows) % SUBMIT_ROWS == 0 {
                        self.flush();
                    }
                }
                Ok(())
            }
            Compute::F32 => bail!("ffn_f16 on the f32 path"),
        }
    }

    /// rms_norm writing f16 (the f16 paths' vision matmul inputs)
    pub fn rms_norm_h(&mut self, x: &Buf, w: &Buf, y: &Buf, rows: usize, dim: usize, eps: f32) -> Result<()> {
        self.dispatch("rms_norm_h", &[rows as u32, dim as u32, eps.to_bits()], &[x, w, y], (rows as u32, 1, 1))
    }

    /// y[N] (+)= W[N,K] · x[K] — one token through a q8_0 matrix (the decode step)
    pub fn matvec(&mut self, x: &Buf, w: &Weight, y: &Buf, accumulate: bool) -> Result<()> {
        if w.ty != WeightType::Q8_0 || w.cols % 32 != 0 {
            bail!("matvec: {:?} [{} × {}] — the kernel is q8_0 with whole blocks only", w.ty, w.rows, w.cols);
        }
        self.dispatch("matvec_q8_0", &[w.rows as u32, w.cols as u32, accumulate as u32], &[x, &w.buf, y], ((w.rows as u32).div_ceil(8), 1, 1))
    }

    /// copy n f32 from src[src0…] to dst[dst0…] (recorded in order with the dispatches). Two slices of one pool block:
    /// a kernel (copy_words) — WebGPU's buffer copy refuses one buffer as source and destination
    pub fn copy(&mut self, src: &Buf, src0: usize, dst: &Buf, dst0: usize, n: usize) -> Result<()> {
        if src.buffer() == dst.buffer() {
            return self.dispatch("copy_words", &[n as u32, src0 as u32, dst0 as u32], &[src, dst], (stride_groups(n as u64), 1, 1));
        }
        self.pass = None;
        self.enc.copy_buffer_to_buffer(src.buffer(), src.offset() + (src0 * 4) as u64, dst.buffer(), dst.offset() + (dst0 * 4) as u64, (n * 4) as u64);
        Ok(())
    }

    /// the device's compute path (which kernels a recording picks)
    pub fn compute(&self) -> Compute {
        self.gpu.compute
    }

    pub fn rms_norm(&mut self, x: &Buf, w: &Buf, y: &Buf, rows: usize, dim: usize, eps: f32) -> Result<()> {
        self.dispatch("rms_norm", &[rows as u32, dim as u32, eps.to_bits()], &[x, w, y], (rows as u32, 1, 1))
    }

    /// rows 0…m in blocks of SUBMIT_ROWS, each recorded by `f(batch, first row, rows)` and submitted on its own
    /// (flush); m ≤ SUBMIT_ROWS: one call, nothing submitted
    fn in_row_blocks(&mut self, m: usize, mut f: impl FnMut(&mut Self, usize, usize) -> Result<()>) -> Result<()> {
        if m <= SUBMIT_ROWS {
            return f(self, 0, m);
        }
        for r0 in (0..m).step_by(SUBMIT_ROWS) {
            f(self, r0, SUBMIT_ROWS.min(m - r0))?;
            self.flush();
        }
        Ok(())
    }

    /// what is recorded so far to the queue as a command buffer of its own; the batch records on into a new one. A long
    /// recording (the vision encoder, the prefill over a large page: seconds of GPU time on a phone) is cut into one
    /// command buffer per layer: iOS takes a command buffer running for seconds as a GPU hang and restarts the GPU
    /// (iPhone 16 Pro: "IOGPUScheduler … GPURestartSignaled type=2", the other work "discarded (victim of GPU
    /// error/recovery)") — the read's output garbage, the next read's first work lost too, no error reaches the page
    /// The batch's parameter slots go on: each command buffer's parameters keep their own slots (written once, before
    /// that command buffer is submitted). Callers wait with throttle between flushes.
    pub fn flush(&mut self) {
        self.pass = None;
        self.write_params();
        let enc = std::mem::replace(&mut self.enc, self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch") }));
        self.gpu.queue.submit([enc.finish()]);
        self.in_flight += 1;
    }

    /// wait for the GPU once MAX_IN_FLIGHT flushed command buffers are queued (a recording of hundreds of them, queued
    /// at once, lost the iPhone's device)
    pub async fn throttle(&mut self) -> Result<()> {
        if self.in_flight >= MAX_IN_FLIGHT {
            self.gpu.wait().await?;
            self.in_flight = 0;
        }
        Ok(())
    }

    /// the parameter words not yet in gpu.params written there (at their own offsets)
    fn write_params(&mut self) {
        if self.words.len() > self.written {
            let at = self.gpu.params.offset() + self.written as u64 * 4;
            self.gpu.queue.write_buffer(self.gpu.params.buffer(), at, bytemuck::cast_slice(&self.words[self.written..]));
            self.written = self.words.len();
        }
    }

    /// the batch to the queue → its submission index (what a map of its results waits for, Gpu::read_mapped_u32)
    pub fn submit(mut self) -> wgpu::SubmissionIndex {
        self.pass = None;
        self.write_params();
        let Some(p) = &self.gpu.prof else {
            return self.gpu.queue.submit([self.enc.finish()]);
        };
        let n = self.prof.len() as u32;
        if n == 0 {
            return self.gpu.queue.submit([self.enc.finish()]);
        }
        let mut pr = p.borrow_mut();
        self.enc.resolve_query_set(&pr.qs, 0..2 * n, pr.resolve.buffer(), 0);
        self.enc.copy_buffer_to_buffer(pr.resolve.buffer(), 0, pr.read.buffer(), 0, 2 * n as u64 * 8);
        let idx = self.gpu.queue.submit([self.enc.finish()]);
        // natively only (the profiler is never on in the browser): wait for the timestamps right here
        let slice = pr.read.buffer().slice(..2 * n as u64 * 8);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let ticks: Vec<u64> = match slice.get_mapped_range() { Ok(v) => bytemuck::cast_slice::<u8, u64>(&v).to_vec(), Err(e) => panic!("profiler readback: {e}") };
        pr.read.buffer().unmap();
        let period = pr.period_ns;
        for (i, key) in self.prof.drain(..).enumerate() {
            let ns = ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * period;
            let e = pr.stats.entry(key).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ns;
        }
        idx
    }
}

/// how activations meet the weights: F32 = this engine (f32 activations everywhere); LlamaMetal = round them the way
/// llama.cpp's Metal kernels do (bf16 into bf16 matmuls, f16 into f32-weight matmuls and flash attention) — the
/// precision bisect: if the wgpu output then matches llama.cpp, the remaining difference is llama.cpp's rounding
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Precision {
    F32,
    LlamaMetal,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightType {
    Bf16,
    /// bf16 converted to f16 at load (the f16 compute paths, Gpu::weights_to_f16), two per u32
    F16,
    F32,
    Q8_0,
}

/// the GPU buffer a tensor gets: its bytes padded to whole u32 words (Gpu::upload_tensor)
pub fn tensor_buffer_bytes(t: &TensorInfo) -> u64 {
    t.n_bytes().max(4).div_ceil(4) * 4
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
}

/// a matrix on the GPU in ggml order: `rows` rows (output features) of `cols` contiguous values (input features)
pub struct Weight {
    pub buf: Buf,
    pub ty: WeightType,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(test)]
mod tests {
    /// the pool's bookkeeping: aligned first fit, splits, merges back to one free range, overlap refused
    #[test]
    fn free_ranges_take_give_merge() {
        let mut f = super::Free::new(1024);
        assert_eq!(f.take(100, 32), Some(0));
        assert_eq!(f.take(100, 32), Some(128)); // 100 → aligned to 128
        assert_eq!(f.take(1000, 32), None);
        assert_eq!(f.0, vec![(100, 28), (228, 796)]);
        f.give(0, 100);
        assert_eq!(f.0, vec![(0, 128), (228, 796)]);
        f.give(128, 100);
        assert_eq!(f.0, vec![(0, 1024)]);
        // a range taken from the middle, given back: whole again
        assert_eq!(f.take(64, 256), Some(0));
        assert_eq!(f.take(64, 256), Some(256));
        f.give(0, 64);
        f.give(256, 64);
        assert_eq!(f.0, vec![(0, 1024)]);
        let r = std::panic::catch_unwind(move || {
            let mut g = super::Free::new(64);
            g.give(0, 32);
        });
        assert!(r.is_err(), "a range given back twice is refused");
    }

    use super::*;

    /// the path choice from an adapter's features / limits (no GPU): shader-f16 with less workgroup memory than the f16
    /// kernels declare → the F32 path (labelled why), not a failure; the needs counted from the kernels' WGSL
    #[test]
    fn compute_choice_from_limits() {
        let need_f32 = Compute::F32.workgroup_need().unwrap();
        let need_f16 = Compute::F16.workgroup_need().unwrap();
        let webgpu = wgpu::Limits::defaults().max_compute_workgroup_storage_size;
        assert_eq!(webgpu, 16384);
        assert!(need_f32 <= webgpu, "the F32 path must fit WebGPU's guaranteed workgroup memory: {need_f32}");
        assert!(need_f16 > webgpu, "attention_tile.wgsl's declarations: {need_f16}");
        let f16 = wgpu::Features::SHADER_F16;
        let coop = Compute::F16Coop.features();
        let lim = |shared: u32| wgpu::Limits { max_compute_workgroup_storage_size: shared, ..wgpu::Limits::defaults() };
        // (features, workgroup memory, 8×8 mixed coop matrix) → path, why f32
        let table: [(wgpu::Features, u32, bool, Compute, Option<String>); 7] = [
            (wgpu::Features::empty(), 16384, false, Compute::F32, Some("no shader-f16".into())),
            (wgpu::Features::empty(), 32768, false, Compute::F32, Some("no shader-f16".into())),
            (f16, 16384, false, Compute::F32, Some("workgroup memory 16 KB".into())),
            (f16, need_f16 - 1, false, Compute::F32, Some(format!("workgroup memory {} KB", (need_f16 - 1) / 1024))),
            (f16, need_f16, false, Compute::F16, None),
            (coop, 32768, true, Compute::F16Coop, None),
            (coop, 32768, false, Compute::F16, None),
        ];
        for (features, shared, mixed, path, why) in table {
            let got = Compute::choose_caps("test", features, &lim(shared), mixed, None).unwrap();
            assert_eq!(got, (path, why), "{features:?} {shared} B coop {mixed}");
        }
        // forced: a path the limits lack is an error; F32 says why
        assert!(Compute::choose_caps("test", f16, &lim(16384), false, Some(Compute::F16)).is_err());
        assert_eq!(Compute::choose_caps("test", f16, &lim(16384), false, Some(Compute::F32)).unwrap(), (Compute::F32, Some("forced".into())));
        assert!(Compute::choose_caps("test", f16, &lim(need_f32 - 16), false, Some(Compute::F32)).is_err());
    }

    /// the device check (web/gpu-check.js NEEDS) judges with the same workgroup memory numbers the engine chooses by
    #[test]
    fn compute_choice_matches_gpu_check_js() {
        let js = include_str!("../../../web/gpu-check.js");
        let need_f16 = Compute::F16.workgroup_need().unwrap();
        let need_f32 = Compute::F32.workgroup_need().unwrap();
        assert!(js.contains(&format!("workgroupStorageF16: {need_f16},")), "web/gpu-check.js NEEDS.workgroupStorageF16 must be {need_f16}");
        assert!(js.contains(&format!("workgroupStorageF32: {need_f32},")), "web/gpu-check.js NEEDS.workgroupStorageF32 must be {need_f32}");
    }

    /// the workgroup bytes of a declaration list: aliases resolved, each variable rounded up to 16 bytes
    #[test]
    fn workgroup_bytes_counts_declarations() {
        let src = "alias Tile = vec4<f16>;\nvar<workgroup> A: array<Tile, 4>;\nvar<workgroup> b: array<f32, 3u>; // c\nvar<workgroup> n: u32;\n";
        assert_eq!(workgroup_bytes("t", src).unwrap(), 32 + 16 + 16);
        assert!(workgroup_bytes("t", "var<workgroup> A: array<mat4x4<f32>, 4>;").is_err());
        let tile = sources(Compute::F16, false).into_iter().find(|(n, _)| *n == "attention_tile").unwrap().1;
        // Qs [64][16] vec4<f16> + Ks, Vs [16][16] vec4<f16> + Ss [16][64] f32 + Red 128 f32 + Cr 64 f32
        assert_eq!(workgroup_bytes("attention_tile", &tile).unwrap(), 64 * 16 * 8 + 2 * 16 * 16 * 8 + 16 * 64 * 4 + 128 * 4 + 64 * 4);
    }

    /// the F32 path (the exact kernels these tests check to f32 tolerances)
    fn f32_gpu() -> Gpu {
        pollster::block_on(Gpu::with_compute(Some(Compute::F32))).unwrap()
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// the tiled matmul (bf16 + f32 weights, with and without accumulate, sizes off the tile grid) against a CPU loop
    #[test]
    fn matmul_matches_cpu() {
        let gpu = f32_gpu();
        let (m, n, k) = (70usize, 37usize, 50usize);
        let mut s = 7u64;
        let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
        let wf: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
        let wb: Vec<half::bf16> = wf.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let base: Vec<f32> = (0..m * n).map(|_| lcg(&mut s)).collect();
        for (ty, wvals, bytes) in [
            (WeightType::F32, wf.clone(), bytemuck::cast_slice::<f32, u8>(&wf).to_vec()),
            (WeightType::Bf16, wb.iter().map(|v| v.to_f32()).collect(), wb.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ] {
            for accumulate in [false, true] {
                let w = Weight { buf: gpu.upload_weight("w", &bytes), ty, rows: n, cols: k };
                let xb = gpu.upload_f32("x", &x);
                let yb = gpu.upload_f32("y", &base);
                let mut b = gpu.batch();
                b.matmul(&xb, &w, &yb, m, accumulate).unwrap();
                b.submit();
                let y = pollster::block_on(gpu.read_f32(&yb, m * n)).unwrap();
                for r in 0..m {
                    for c in 0..n {
                        let mut want: f32 = (0..k).map(|i| x[r * k + i] * wvals[c * k + i]).sum();
                        if accumulate {
                            want += base[r * n + c];
                        }
                        let got = y[r * n + c];
                        assert!((got - want).abs() < 1e-4, "{ty:?} acc={accumulate} [{r},{c}] {got} vs {want}");
                    }
                }
            }
        }
    }

    /// the 128×128 tile gives the 64×64 tile's bits (bf16 + f32 weights, with and without accumulate, M / N off the
    /// tile grid, K not a multiple of the 16-wide k step)
    #[test]
    fn matmul128_is_bit_identical_to_matmul64() {
        let gpu = f32_gpu();
        let (m, n, k) = (300usize, 170usize, 588usize);
        let mut s = 17u64;
        let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
        let wf: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
        let wb: Vec<u8> = wf.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect();
        let base: Vec<f32> = (0..m * n).map(|_| lcg(&mut s)).collect();
        for (ty, bytes) in [("f32", bytemuck::cast_slice::<f32, u8>(&wf).to_vec()), ("bf16", wb)] {
            for acc in [0u32, 1] {
                let w = gpu.upload_weight("w", &bytes);
                let xb = gpu.upload_f32("x", &x);
                let (y64, y128) = (gpu.upload_f32("y64", &base), gpu.upload_f32("y128", &base));
                let params = [m as u32, n as u32, k as u32, acc, 0];
                let mut b = gpu.batch();
                b.dispatch(&format!("matmul_{ty}"), &params, &[&xb, &w, &y64], ((n as u32).div_ceil(64), (m as u32).div_ceil(64), 1)).unwrap();
                b.dispatch(&format!("matmul128_{ty}"), &params, &[&xb, &w, &y128], ((n as u32).div_ceil(128), (m as u32).div_ceil(128), 1)).unwrap();
                b.submit();
                let a = pollster::block_on(gpu.read_f32(&y64, m * n)).unwrap();
                let c = pollster::block_on(gpu.read_f32(&y128, m * n)).unwrap();
                for i in 0..m * n {
                    assert_eq!(a[i].to_bits(), c[i].to_bits(), "{ty} acc={acc} [{}, {}] {} vs {}", i / n, i % n, a[i], c[i]);
                }
            }
        }
    }

    /// a tensor streamed through a staging buffer far smaller than it (odd length, odd offset) arrives intact
    #[test]
    fn streamed_upload_is_exact() {
        let gpu = f32_gpu();
        let file: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
        let t = TensorInfo { name: "t".into(), shape: vec![501], ty: crate::gguf::TensorType::F16, offset: 0, file_offset: 97 };
        // 501 f16 = 1002 bytes would run past the file: use 449 values from 97 (898 bytes, ends at 995)
        let t = TensorInfo { shape: vec![449], ..t };
        let mut up = gpu.uploader(40);
        let buf = pollster::block_on(gpu.upload_tensor(&mut up, &file, &t)).unwrap();
        pollster::block_on(gpu.upload_flush(&mut up)).unwrap();
        let words = pollster::block_on(gpu.read_f32(&buf, 900 / 4)).unwrap();
        let bytes: Vec<u8> = bytemuck::cast_slice(&words).to_vec();
        assert_eq!(&bytes[..898], &file[97..995]);
        assert!(bytes[898..].iter().all(|&b| b == 0));
    }

    static HOOK_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    fn count_hook(_kind: &'static str, _message: &str) {
        HOOK_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// a lost device (device.destroy(), as the browser selftest does) marks the engine dead: a wait fails with the
    /// reason instead of hanging, so does every later one, the host's hook hears it once; an uncaptured GPU error (an
    /// oversized buffer) does the same; dropping a healthy Gpu is no event
    #[test]
    fn lost_device_and_gpu_errors_surface() {
        set_fatal_hook(count_hook);
        let before = HOOK_CALLS.load(std::sync::atomic::Ordering::SeqCst);
        drop(f32_gpu());
        let gpu = f32_gpu();
        let b = gpu.scratch("x", 16);
        gpu.alive().unwrap();
        gpu.device.destroy();
        // natively wgpu calls the lost callback in the next poll (the browser: the device.lost promise)
        let e = pollster::block_on(gpu.wait()).unwrap_err().to_string();
        assert!(e.starts_with("GPU device lost (destroyed)"), "{e}");
        let e2 = pollster::block_on(gpu.read_f32(&b, 16)).unwrap_err().to_string();
        assert_eq!(e, e2);
        assert_eq!(gpu.alive().unwrap_err().to_string(), e);
        drop(b);
        drop(gpu);

        let gpu = f32_gpu();
        let too_big = gpu.limits.max_buffer_size + 4;
        let _b = gpu.alloc("too big", too_big, wgpu::BufferUsages::STORAGE, MemKind::Work);
        let e = gpu.alive().unwrap_err().to_string();
        assert!(e.starts_with("GPU validation error"), "{e}");
        assert!(pollster::block_on(gpu.wait()).unwrap_err().to_string() == e);
        drop(_b);
        drop(gpu);
        assert_eq!(HOOK_CALLS.load(std::sync::atomic::Ordering::SeqCst) - before, 2);
    }

    /// the ledger counts a buffer while any clone lives, by kind; the read's phase peaks follow the total
    #[test]
    fn memory_ledger_counts_live_buffers() {
        let gpu = f32_gpu();
        let base = gpu.mem_live();
        gpu.mem_begin_read();
        gpu.phase("vision");
        let a = gpu.scratch("a", 1000);
        let a2 = a.clone();
        let kv = gpu.scratch_as("kv", 500, MemKind::KvCache);
        assert_eq!(gpu.mem_live().work - base.work, 4000);
        assert_eq!(gpu.mem_live().kv_cache - base.kv_cache, 2000);
        drop(a);
        assert_eq!(gpu.mem_live().work - base.work, 4000, "a clone keeps the buffer");
        drop(a2);
        assert_eq!(gpu.mem_live().work, base.work);
        gpu.phase("prefill");
        drop(kv);
        let r = gpu.mem_read(base.total()).unwrap();
        assert_eq!(r.peak_phase, "vision");
        assert_eq!(r.peak_bytes, base.total() + 6000);
        assert_eq!(r.phases.iter().find(|p| p.phase == "prefill").unwrap().peak.total(), base.total() + 2000);
        gpu.phase("-");
    }

    #[test]
    fn rms_norm_matches_cpu() {
        let gpu = f32_gpu();
        let (rows, dim, eps) = (5usize, 1000usize, 1e-5f32);
        let mut s = 3u64;
        let x: Vec<f32> = (0..rows * dim).map(|_| lcg(&mut s) * 3.0).collect();
        let w: Vec<f32> = (0..dim).map(|_| lcg(&mut s)).collect();
        let (xb, wb, yb) = (gpu.upload_f32("x", &x), gpu.upload_weight_f32("w", &w), gpu.scratch("y", rows * dim));
        let mut b = gpu.batch();
        b.rms_norm(&xb, &wb, &yb, rows, dim, eps).unwrap();
        b.submit();
        let y = pollster::block_on(gpu.read_f32(&yb, rows * dim)).unwrap();
        for r in 0..rows {
            let row = &x[r * dim..(r + 1) * dim];
            let scale = 1.0 / ((row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / dim as f64) as f32 + eps).sqrt();
            for i in 0..dim {
                let want = row[i] * scale * w[i];
                assert!((y[r * dim + i] - want).abs() < 1e-5, "[{r},{i}] {} vs {want}", y[r * dim + i]);
            }
        }
    }
}
