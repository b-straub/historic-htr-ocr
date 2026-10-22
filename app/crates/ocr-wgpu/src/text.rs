// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The LightOnOCR text decoder on wgpu — llama.cpp's Qwen3 graph (src/models/qwen3.cpp + llama-graph build_norm /
// build_qkv / build_attn / build_ffn), op for op:
//   x = token_embd[tok] (q8_0 rows) or the image embeddings → 28 × [RMSNorm attn_norm → q / k / v (no bias) →
//   per-head RMSNorm q_norm / k_norm (over the 128 dims of each head) → RoPE NEOX θ = 1e6 on q, k at the token's
//   position → K, V into the f16 cache → causal softmax(q·kᵀ/√128)·v, GQA (16 q heads on 8 kv heads) → attn_output,
//   + residual → RMSNorm ffn_norm → silu(gate)·up → down, + residual] → RMSNorm output_norm → logits = the tied
//   token_embd (no output.weight in the GGUF) · x, for the last position only.
// Weights stay q8_0 on the GPU (34-byte blocks read as u32 words, dequantized in the kernels); norms f32.
// Prefill = one batch over all prompt rows (tiled matmul + the causal flash kernel); decode = one token: on the F32
// reference q8_0 matvec + three small attention kernels (18 dispatches per layer), on the f16 paths the fused step
// (layers_fused: norms, SwiGLU, RoPE and the KV store folded into the projections, split attention — 7 per layer).
// KV cache: f16, two per u32 — llama.cpp's default cache type — grown with the read (Session::reserve: the prompt +
// kv_grow rows, then kv_grow more whenever a step would write past the end, at most prompt + max new tokens), so it
// holds what the read uses, not its cap; freed with the session at the read's end. A read sizes kv_grow by its image
// (session_for: the image tokens — a page's text is rarely longer): each growth allocates the whole larger cache and
// frees the old one, and iOS keeps counting freed GPU memory against the page (the iPhone died at the end of a read
// whose cache had grown three times by 512 rows) — so a read's cache is allocated once, a growth is the exception.
// Positions run 0, 1, … over every row, image rows included (mtmd POS_TYPE_NORMAL: an image chunk's n_pos = its
// token count; the attention is causal over the image too — mtmd_decode_use_non_causal is false for LIGHTONOCR).

use std::collections::VecDeque;

use anyhow::{Context, Result, bail};
use ocr_decode::presets::Preset;

use crate::gguf::{Gguf, TensorSource, TensorType};
use crate::gpu::{BindCache, Buf, Compute, DMV_FFN_R, DMV_R, Gpu, MemKind, Uploader, Weight, WeightType, stride_groups};
use crate::tokenizer::Token;

/// marks a prompt row the embedding gather skips (an image row, copied from the vision encoder)
const SKIP: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct TextConfig {
    pub n_layer: usize,
    pub dim: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub d_head: usize,
    pub n_ff: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub n_vocab: usize,
    pub n_ctx_train: usize,
}

impl TextConfig {
    pub fn from_gguf<S: TensorSource>(g: &Gguf<S>) -> Result<Self> {
        let arch = g.str("general.architecture")?;
        if arch != "qwen3" {
            bail!("architecture {arch:?}: this engine runs qwen3 only");
        }
        let u = |k: &str| -> Result<usize> { Ok(g.u32(&format!("qwen3.{k}"))? as usize) };
        let d_head = u("attention.key_length")?;
        if d_head != 128 || u("attention.value_length")? != 128 {
            bail!("head dim {d_head}: the attention kernels are written for 128");
        }
        let (n_head, n_kv) = (u("attention.head_count")?, u("attention.head_count_kv")?);
        if n_kv == 0 || n_head % n_kv != 0 {
            bail!("{n_head} q heads on {n_kv} kv heads");
        }
        if g.has_key("qwen3.rope.scaling.type") || g.has_key("qwen3.rope.dimension_count") {
            bail!("rope scaling / partial rope in the GGUF: not part of this engine's graph");
        }
        let tok = g.info("token_embd.weight")?;
        Ok(Self {
            n_layer: u("block_count")?,
            dim: u("embedding_length")?,
            n_head,
            n_kv,
            d_head,
            n_ff: u("feed_forward_length")?,
            eps: g.f32("qwen3.attention.layer_norm_rms_epsilon")?,
            rope_base: g.f32("qwen3.rope.freq_base")?,
            n_vocab: *tok.shape.last().context("token_embd without shape")? as usize,
            n_ctx_train: u("context_length")?,
        })
    }
}

struct Layer {
    attn_norm: Buf,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: Buf,
    k_norm: Buf,
    ffn_norm: Buf,
    gate: Weight,
    up: Weight,
    down: Weight,
}

pub struct TextModel {
    pub cfg: TextConfig,
    tok_embd: Weight,
    layers: Vec<Layer>,
    output_norm: Buf,
}

/// one prompt piece: text tokens, or rows of embeddings already on the GPU ([n][dim] f32)
pub enum Input<'a> {
    Tokens(&'a [Token]),
    Embeddings { buf: &'a Buf, n: usize },
}

async fn weight<S: TensorSource>(gpu: &Gpu, up: &mut Uploader, g: &Gguf<S>, name: &str, rows: usize, cols: usize) -> Result<Weight> {
    let t = g.info(name)?;
    if t.shape != [cols as u64, rows as u64] {
        bail!("{name}: shape {:?}, expected [{cols}, {rows}]", t.shape);
    }
    if t.ty != TensorType::Q8_0 {
        bail!("{name}: {:?} — the text decoder's kernels read q8_0 weights", t.ty);
    }
    Ok(Weight { buf: gpu.upload_tensor(up, &g.source, t).await?, ty: WeightType::Q8_0, rows, cols })
}

async fn vector<S: TensorSource>(gpu: &Gpu, up: &mut Uploader, g: &Gguf<S>, name: &str, n: usize) -> Result<Buf> {
    let t = g.info(name)?;
    if t.ty != TensorType::F32 || t.shape != [n as u64] {
        bail!("{name}: {:?} {:?}, expected F32 [{n}]", t.ty, t.shape);
    }
    gpu.upload_tensor(up, &g.source, t).await
}

/// cos / sin of RoPE NEOX per position as llama.cpp Metal's kernel_rope_neox computes them (f32):
/// θ = (float) pos · pow(base, −2i/d) — [pos][d/2]
pub fn rope_table(base: f32, d_head: usize, n_pos: usize) -> Vec<[f32; 2]> {
    let half = d_head / 2;
    let mut out = Vec::with_capacity(n_pos * half);
    for pos in 0..n_pos {
        for i in 0..half {
            let theta = pos as f32 * base.powf(-((2 * i) as f32) / d_head as f32);
            out.push([theta.cos(), theta.sin()]);
        }
    }
    out
}

/// argmax partials of the GPU token choice (choose_max's workgroups → choose_pick)
const CHOOSE_GROUPS: usize = 128;
/// −inf's bits: the argmax kernels' start value comes from the params (no WGSL constant can be −inf)
const NEG_INF: u32 = 0xff80_0000;
/// the most choices one run-ahead submission records (its distinct parameter sets must fit the uniform ring)
pub const MAX_CHUNK: usize = 64;

/// why a preset's next token cannot be chosen on the GPU (Ok = it can): only greedy presets — a sampled read keeps
/// ocr-decode's CPU path (top-p sorts the distribution; its random draw is the CPU's SplitMix64); the repetition
/// penalty must be a positive normal f32 (the kernel's exact division, choose_pen.wgsl)
pub fn gpu_choice_capable(p: &Preset) -> std::result::Result<(), &'static str> {
    if !p.greedy() {
        return Err("sampled (temperature > 0): the token is drawn on the CPU");
    }
    let rp = p.repetition_penalty as f32;
    if p.repetition_penalty != 1.0 && !(rp.is_normal() && rp > 0.0) {
        return Err("the repetition penalty is no positive normal f32");
    }
    Ok(())
}

/// one read's GPU token choice (Choice::new): the history (the window's start, then every chosen token — what
/// the CPU reads back), the output counts + fp · count (frequency penalty), the argmax partials, the readback ring
pub struct Choice {
    hist: Buf,
    seed_len: usize,
    window: usize,
    rp: Option<f32>,
    fp: Option<f32>,
    n_vocab: usize,
    counts: Buf,
    fpen: Buf,
    part: Buf,
    slots: Vec<Buf>,
    chunk: usize,
}

impl Choice {
    /// the GPU token choice of a read with preset `p` (gpu_choice_capable) over `n_vocab` logits: `history` = the
    /// repetition window's start as ocr-decode's generate gets it (the prompt tail); `chunk` choices per submission,
    /// `depth` submissions in flight (TextModel::decode_gpu)
    pub fn new(gpu: &Gpu, p: &Preset, history: &[Token], n_vocab: usize, chunk: usize, depth: usize) -> Result<Self> {
        if let Err(why) = gpu_choice_capable(p) {
            bail!("preset '{}': {why}", p.name);
        }
        if !(1..=MAX_CHUNK).contains(&chunk) || depth < 1 {
            bail!("run-ahead {chunk} choices × {depth} submissions: 1…{MAX_CHUNK} × ≥ 1");
        }
        // the window's start (all ocr-decode's choose ever looks at), then room for every chosen token
        let seed = &history[history.len().saturating_sub(p.repetition_window)..];
        let mut hist: Vec<u32> = Vec::with_capacity(seed.len() + p.max_new_tokens);
        for &t in seed {
            if t < 0 || t as usize >= n_vocab {
                bail!("history token {t} is outside the vocabulary");
            }
            hist.push(t as u32);
        }
        hist.resize(seed.len() + p.max_new_tokens, 0);
        let fp = (p.frequency_penalty > 0.0).then_some(p.frequency_penalty as f32);
        // the output counts + fp · count (choose_pick / choose_max), zeroed; without fp two 1-word stand-ins
        let words = if fp.is_some() { n_vocab } else { 1 };
        Ok(Self {
            hist: gpu.upload("history", bytemuck::cast_slice(&hist)),
            seed_len: seed.len(),
            window: p.repetition_window,
            rp: (p.repetition_penalty != 1.0).then_some(p.repetition_penalty as f32),
            fp,
            n_vocab,
            counts: gpu.upload("output counts", &vec![0u8; words * 4]),
            fpen: gpu.upload("frequency penalties", &vec![0u8; words * 4]),
            part: gpu.scratch("argmax partials", 2 * CHOOSE_GROUPS),
            slots: (0..depth).map(|_| gpu.readback("token readback", chunk)).collect(),
            chunk,
        })
    }

    /// choice `i` of the read (0 = from the prompt's logits) into `b`: `logits` (penalised in place) → argmax → `tok`
    /// (the next step's embedding input) + the history
    pub fn record(&self, b: &mut crate::gpu::Batch, logits: &Buf, tok: &Buf, i: usize) -> Result<()> {
        let len = (self.seed_len + i) as u32;
        if let Some(rp) = self.rp {
            b.dispatch("choose_pen", &[len, self.window as u32, rp.to_bits(), 0], &[&self.hist, logits], (1, 1, 1))?;
        }
        let use_fp = self.fp.is_some() as u32;
        b.dispatch("choose_max", &[self.n_vocab as u32, use_fp, NEG_INF, 0], &[logits, &self.counts, &self.fpen, &self.part], (CHOOSE_GROUPS as u32, 1, 1))?;
        let fp = self.fp.unwrap_or(0.0).to_bits();
        b.dispatch("choose_pick", &[CHOOSE_GROUPS as u32, len, use_fp, fp, NEG_INF], &[&self.part, tok, &self.hist, &self.counts, &self.fpen], (1, 1, 1))
    }
}

/// rows (positions) the KV cache gets beyond the need when it grows (Session::reserve): 512 = 56 MiB on this model —
/// a page's read grows a few times, a 7024-token run to the cap 14 times
pub const KV_GROW: usize = 512;
/// layers whose K / V caches one growth submission replaces before it waits for the GPU (the old buffers gone): the
/// growth's extra memory is the new cache of these layers, not a whole second cache
const KV_GROW_LAYERS: usize = 4;

/// one read's state: the KV cache (grown with the read), the RoPE table, the decode step's scratch buffers
pub struct Session {
    /// the positions the read may reach (prompt + max new tokens): the RoPE table, the attention partials, the cap
    pub n_ctx: usize,
    pub n_past: usize,
    /// the KV cache rows allocated (Session::reserve), ≤ n_ctx
    kv_rows: usize,
    /// the rows a growth adds beyond the need (KV_GROW; the tests make it small)
    kv_grow: usize,
    /// one row's u32 words per layer (n_kv · d_head f16)
    kv_words: usize,
    n_layer: usize,
    k: Vec<Buf>,
    v: Vec<Buf>,
    rope: Buf,
    // decode scratch (one row)
    x: Buf,
    h: Buf,
    qr: Buf,
    q: Buf,
    kr: Buf,
    kn: Buf,
    vr: Buf,
    a: Buf,
    g: Buf,
    u: Buf,
    scores: Buf,
    logits: Buf,
    tok: Buf,
    /// the logits' CPU side (copied there in the step's own batch)
    readback: Buf,
    /// the decode step's bind groups (the same buffers every token)
    binds: BindCache,
    /// the fused decode step's [q | k | v] row (F16 paths: dmv_qkv → qkv_post → the attention)
    qkv: Buf,
    /// the split decode attention's per-chunk partials (F16 paths): [n_head][n_chunk][130]
    part: Buf,
    n_chunk: usize,
}

impl Session {
    /// the KV cache rows allocated now (positions 0 … kv_rows − 1)
    pub fn kv_rows(&self) -> usize {
        self.kv_rows
    }

    /// KV cache rows for positions 0 … rows − 1: when it holds fewer, it grows to rows + kv_grow (at most n_ctx).
    /// The first call allocates it; a growth copies the rows recorded so far (n_past — the GPU's queue orders the copy
    /// after the submissions in flight that write them) into larger buffers, KV_GROW_LAYERS layers per submission,
    /// and waits for each before the next allocates: the old buffers of a group are gone (their bind groups
    /// forgotten) by then, so the growth's peak is the old cache + one group's new buffers — what the ledger counts.
    async fn reserve(&mut self, gpu: &Gpu, rows: usize) -> Result<()> {
        if rows <= self.kv_rows {
            return Ok(());
        }
        if rows > self.n_ctx {
            bail!("{rows} KV cache rows asked, the session holds at most {}", self.n_ctx);
        }
        let new_rows = (rows + self.kv_grow).min(self.n_ctx);
        let (words, copied) = (new_rows * self.kv_words, self.n_past * self.kv_words);
        for l0 in (0..self.n_layer).step_by(KV_GROW_LAYERS) {
            let mut b = gpu.batch();
            let mut old: Vec<Buf> = Vec::new();
            for li in l0..(l0 + KV_GROW_LAYERS).min(self.n_layer) {
                for (label, cache) in [("k cache", &mut self.k), ("v cache", &mut self.v)] {
                    let new = gpu.scratch_as(label, words, MemKind::KvCache);
                    if li == cache.len() {
                        cache.push(new);
                        continue;
                    }
                    if copied > 0 {
                        b.copy(&cache[li], 0, &new, 0, copied);
                    }
                    old.push(std::mem::replace(&mut cache[li], new));
                }
            }
            if old.is_empty() {
                continue;
            }
            self.binds.forget(&old.iter().collect::<Vec<&Buf>>());
            b.submit();
            drop(old);
            gpu.wait().await?;
        }
        self.kv_rows = new_rows;
        Ok(())
    }
}

impl TextModel {
    /// the weights streamed file → GPU through `up` (gpu.rs: one staging buffer, never the model in CPU memory)
    pub async fn load<S: TensorSource>(gpu: &Gpu, g: &Gguf<S>, up: &mut Uploader) -> Result<Self> {
        let cfg = TextConfig::from_gguf(g)?;
        let (d, f, qd, kvd) = (cfg.dim, cfg.n_ff, cfg.n_head * cfg.d_head, cfg.n_kv * cfg.d_head);
        if d % 32 != 0 || f % 32 != 0 || qd % 32 != 0 {
            bail!("dims {d} / {f} / {qd}: q8_0 rows must be whole blocks");
        }
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{i}.{s}.weight");
            for absent in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
                let b = format!("blk.{i}.{absent}.bias");
                if g.has(&b) {
                    bail!("{b} is in the GGUF but not part of this engine's graph");
                }
            }
            if g.has(&n("attn_qkv")) {
                bail!("blk.{i}: a fused attn_qkv — this engine reads separate q / k / v");
            }
            layers.push(Layer {
                attn_norm: vector(gpu, up, g, &n("attn_norm"), d).await?,
                q: weight(gpu, up, g, &n("attn_q"), qd, d).await?,
                k: weight(gpu, up, g, &n("attn_k"), kvd, d).await?,
                v: weight(gpu, up, g, &n("attn_v"), kvd, d).await?,
                o: weight(gpu, up, g, &n("attn_output"), d, qd).await?,
                q_norm: vector(gpu, up, g, &n("attn_q_norm"), cfg.d_head).await?,
                k_norm: vector(gpu, up, g, &n("attn_k_norm"), cfg.d_head).await?,
                ffn_norm: vector(gpu, up, g, &n("ffn_norm"), d).await?,
                gate: weight(gpu, up, g, &n("ffn_gate"), f, d).await?,
                up: weight(gpu, up, g, &n("ffn_up"), f, d).await?,
                down: weight(gpu, up, g, &n("ffn_down"), d, f).await?,
            });
        }
        if g.has("output.weight") {
            bail!("output.weight is in the GGUF: this engine ties the output to token_embd");
        }
        let m = Self {
            tok_embd: weight(gpu, up, g, "token_embd.weight", cfg.n_vocab, d).await?,
            output_norm: vector(gpu, up, g, "output_norm.weight", d).await?,
            layers,
            cfg,
        };
        gpu.wait().await?;
        Ok(m)
    }

    /// a session for n_ctx positions (prompt + the tokens to generate); its KV cache is allocated by the prefill
    pub fn session(&self, gpu: &Gpu, n_ctx: usize) -> Result<Session> {
        self.session_grown(gpu, n_ctx, KV_GROW)
    }

    /// `session` for a read of `expect` new tokens (its image tokens): the cache holds the prompt + `expect` rows from
    /// the prefill on and grows by as much again only when the text runs longer (at least KV_GROW rows)
    pub fn session_for(&self, gpu: &Gpu, n_ctx: usize, expect: usize) -> Result<Session> {
        self.session_grown(gpu, n_ctx, expect.max(KV_GROW))
    }

    /// `session` with the KV cache growing by `kv_grow` rows
    fn session_grown(&self, gpu: &Gpu, n_ctx: usize, kv_grow: usize) -> Result<Session> {
        let c = &self.cfg;
        if kv_grow == 0 {
            bail!("a KV cache growth of 0 rows");
        }
        if gpu.compute != Compute::F32 && c.n_head != 2 * c.n_kv {
            bail!("{} q heads on {} kv heads: the split decode attention (attn_decode.wgsl) is written for groups of 2", c.n_head, c.n_kv);
        }
        if gpu.compute != Compute::F32 && ((c.n_head * c.d_head) % (8 * DMV_R) != 0 || (c.n_kv * c.d_head) % (8 * DMV_R) != 0) {
            bail!("q / k widths {} / {}: dmv_qkv needs multiples of {}", c.n_head * c.d_head, c.n_kv * c.d_head, 8 * DMV_R);
        }
        if n_ctx > c.n_ctx_train {
            bail!("{n_ctx} positions asked, the model was trained on {}", c.n_ctx_train);
        }
        let (qd, kvd) = (c.n_head * c.d_head, c.n_kv * c.d_head);
        let cs: Vec<f32> = rope_table(c.rope_base, c.d_head, n_ctx).into_iter().flatten().collect();
        Ok(Session {
            n_ctx,
            n_past: 0,
            kv_rows: 0,
            kv_grow,
            kv_words: kvd / 2,
            n_layer: c.n_layer,
            k: Vec::with_capacity(c.n_layer),
            v: Vec::with_capacity(c.n_layer),
            rope: gpu.upload_f32("rope", &cs),
            x: gpu.scratch("x", c.dim),
            h: gpu.scratch("h", c.dim),
            qr: gpu.scratch("q raw", qd),
            q: gpu.scratch("q", qd),
            kr: gpu.scratch("k raw", kvd),
            kn: gpu.scratch("k", kvd),
            vr: gpu.scratch("v", kvd),
            a: gpu.scratch("attn", qd),
            g: gpu.scratch("gate", c.n_ff),
            u: gpu.scratch("up", c.n_ff),
            scores: gpu.scratch("scores", c.n_head * n_ctx),
            logits: gpu.scratch("logits", c.n_vocab),
            tok: gpu.scratch("token", 1),
            readback: gpu.readback("logits readback", c.n_vocab),
            binds: BindCache::default(),
            qkv: gpu.scratch("qkv", qd + 2 * kvd),
            part: gpu.scratch("attention partials", c.n_head * n_ctx.div_ceil(crate::gpu::DECODE_CHUNK) * 130),
            n_chunk: n_ctx.div_ceil(crate::gpu::DECODE_CHUNK),
        })
    }

    /// the prompt in one batch (positions n_past …) → the last row's logits
    pub async fn prefill(&self, gpu: &Gpu, s: &mut Session, inputs: &[Input<'_>]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let mut ids: Vec<u32> = Vec::new();
        for inp in inputs {
            match inp {
                Input::Tokens(t) => {
                    for &id in *t {
                        if id < 0 || id as usize >= c.n_vocab {
                            bail!("token {id} is outside the vocabulary");
                        }
                        ids.push(id as u32);
                    }
                }
                Input::Embeddings { n, .. } => ids.extend(std::iter::repeat_n(SKIP, *n)),
            }
        }
        let l = ids.len();
        if l == 0 {
            bail!("empty prompt");
        }
        if s.n_past + l > s.n_ctx {
            bail!("{} + {l} positions overflow the session's {}", s.n_past, s.n_ctx);
        }
        s.reserve(gpu, s.n_past + l).await?;
        let (d, f, qd, kvd) = (c.dim, c.n_ff, c.n_head * c.d_head, c.n_kv * c.d_head);
        let pos0 = s.n_past as u32;
        let tok = gpu.upload("tokens", bytemuck::cast_slice(&ids));
        let x = gpu.scratch("x", l * d);
        let h = gpu.scratch("h", l * d);
        let (qr, q) = (gpu.scratch("q raw", l * qd), gpu.scratch("q", l * qd));
        let (kr, kn, vr) = (gpu.scratch("k raw", l * kvd), gpu.scratch("k", l * kvd), gpu.scratch("v", l * kvd));
        let (g, u) = (gpu.scratch("gate", l * f), gpu.scratch("up", l * f));

        let mut b = gpu.batch();
        let mut row = 0usize;
        for inp in inputs {
            match inp {
                Input::Tokens(t) => row += t.len(),
                Input::Embeddings { buf, n } => {
                    b.copy(buf, 0, &x, row * d, n * d);
                    row += n;
                }
            }
        }
        b.dispatch("embed_q8_0", &[l as u32, d as u32], &[&tok, &self.tok_embd.buf, &x], (stride_groups((l * d) as u64), 1, 1))?;
        let attn = [l as u32, c.n_head as u32, c.n_kv as u32, pos0, (1.0 / (c.d_head as f32).sqrt()).to_bits()];
        for (li, ly) in self.layers.iter().enumerate() {
            b.rms_norm(&x, &ly.attn_norm, &h, l, d, c.eps)?;
            b.matmul(&h, &ly.q, &qr, l, false)?;
            b.matmul(&h, &ly.k, &kr, l, false)?;
            b.matmul(&h, &ly.v, &vr, l, false)?;
            b.rms_norm(&qr, &ly.q_norm, &q, l * c.n_head, c.d_head, c.eps)?;
            b.rms_norm(&kr, &ly.k_norm, &kn, l * c.n_kv, c.d_head, c.eps)?;
            self.rope(&mut b, s, &q, l, c.n_head, pos0)?;
            self.rope(&mut b, s, &kn, l, c.n_kv, pos0)?;
            b.dispatch("kv_store", &[l as u32, kvd as u32, pos0], &[&kn, &s.k[li]], (stride_groups((l * kvd / 2) as u64), 1, 1))?;
            b.dispatch("kv_store", &[l as u32, kvd as u32, pos0], &[&vr, &s.v[li]], (stride_groups((l * kvd / 2) as u64), 1, 1))?;
            // the attention output reuses the raw-q buffer
            // (a matrix-unit version of this kernel — 16 queries per workgroup, 30 KB of workgroup memory — measured
            // slower than this one on the M3: 31 vs 24 ms per layer on a full test page)
            let causal = if gpu.compute == Compute::F32 { "attention_causal" } else { "attention_causal_h" };
            b.dispatch(causal, &attn, &[&q, &s.k[li], &s.v[li], &qr], ((l as u32).div_ceil(64), c.n_head as u32, 1))?;
            b.matmul(&qr, &ly.o, &x, l, true)?;
            b.rms_norm(&x, &ly.ffn_norm, &h, l, d, c.eps)?;
            b.matmul(&h, &ly.gate, &g, l, false)?;
            b.matmul(&h, &ly.up, &u, l, false)?;
            b.dispatch("swiglu", &[(l * f) as u32], &[&u, &g], (stride_groups((l * f) as u64), 1, 1))?;
            b.matmul(&g, &ly.down, &x, l, true)?;
        }
        b.copy(&x, (l - 1) * d, &s.x, 0, d);
        self.head(&mut b, s)?;
        b.submit();
        s.n_past += l;
        gpu.read_f32(&s.logits, c.n_vocab).await
    }

    /// one token at position n_past → its logits (the CPU token choice: the logits read back every step)
    pub async fn step(&self, gpu: &Gpu, s: &mut Session, token: Token) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if token < 0 || token as usize >= c.n_vocab {
            bail!("token {token} is outside the vocabulary");
        }
        s.reserve(gpu, s.n_past + 1).await?;
        gpu.queue.write_buffer(s.tok.buffer(), s.tok.offset(), bytemuck::bytes_of(&(token as u32)));
        let mut b = gpu.batch_cached(&s.binds);
        self.record_step(&mut b, s, s.n_past)?;
        b.copy(&s.logits, 0, &s.readback, 0, c.n_vocab);
        b.submit();
        s.n_past += 1;
        gpu.read_mapped(&s.readback, c.n_vocab).await
    }

    /// one decode step into `b`: the token in s.tok at position `pos` → s.logits (the caller advances s.n_past)
    fn record_step(&self, b: &mut crate::gpu::Batch, s: &Session, pos: usize) -> Result<()> {
        let c = &self.cfg;
        if pos + 1 > s.kv_rows {
            bail!("position {pos}: the KV cache holds {} rows (Session::reserve first)", s.kv_rows);
        }
        let (d, f, kvd) = (c.dim, c.n_ff, c.n_kv * c.d_head);
        let pos0 = pos as u32;
        b.dispatch("embed_q8_0", &[1, d as u32], &[&s.tok, &self.tok_embd.buf, &s.x], (stride_groups(d as u64), 1, 1))?;
        // the f16 paths: the fused step; the F32 reference below keeps its kernels (its reads are the baseline)
        if b.compute() != Compute::F32 {
            self.layers_fused(b, s, pos)?;
            let groups = (c.n_vocab.div_ceil(8 * DMV_R) as u32, 1, 1);
            return b.dispatch("dmv_norm", &[c.n_vocab as u32, d as u32, 0, c.eps.to_bits()], &[&s.x, &self.output_norm, &self.tok_embd.buf, &s.logits], groups);
        }
        let n_keys = pos + 1;
        let attn = [n_keys as u32, c.n_head as u32, c.n_kv as u32, s.n_ctx as u32, (1.0 / (c.d_head as f32).sqrt()).to_bits()];
        for (li, ly) in self.layers.iter().enumerate() {
            b.rms_norm(&s.x, &ly.attn_norm, &s.h, 1, d, c.eps)?;
            b.matvec(&s.h, &ly.q, &s.qr, false)?;
            b.matvec(&s.h, &ly.k, &s.kr, false)?;
            b.matvec(&s.h, &ly.v, &s.vr, false)?;
            b.rms_norm(&s.qr, &ly.q_norm, &s.q, c.n_head, c.d_head, c.eps)?;
            b.rms_norm(&s.kr, &ly.k_norm, &s.kn, c.n_kv, c.d_head, c.eps)?;
            self.rope(b, s, &s.q, 1, c.n_head, pos0)?;
            self.rope(b, s, &s.kn, 1, c.n_kv, pos0)?;
            b.dispatch("kv_store", &[1, kvd as u32, pos0], &[&s.kn, &s.k[li]], (stride_groups((kvd / 2) as u64), 1, 1))?;
            b.dispatch("kv_store", &[1, kvd as u32, pos0], &[&s.vr, &s.v[li]], (stride_groups((kvd / 2) as u64), 1, 1))?;
            b.dispatch("attn_scores", &attn, &[&s.q, &s.k[li], &s.scores], (stride_groups((c.n_head * n_keys) as u64), 1, 1))?;
            b.dispatch("attn_softmax", &attn, &[&s.scores], (c.n_head as u32, 1, 1))?;
            b.dispatch("attn_values", &attn, &[&s.scores, &s.v[li], &s.a], (c.n_head as u32, 1, 1))?;
            b.matvec(&s.a, &ly.o, &s.x, true)?;
            b.rms_norm(&s.x, &ly.ffn_norm, &s.h, 1, d, c.eps)?;
            b.matvec(&s.h, &ly.gate, &s.g, false)?;
            b.matvec(&s.h, &ly.up, &s.u, false)?;
            b.dispatch("swiglu", &[f as u32], &[&s.u, &s.g], (stride_groups(f as u64), 1, 1))?;
            b.matvec(&s.g, &ly.down, &s.x, true)?;
        }
        self.head(b, s)
    }

    /// The read's decode with the token chosen on the GPU, run ahead of the CPU: up to `max` choices, the first from
    /// the prompt's logits (s.logits after prefill), each further one after a decode step that embeds the previous
    /// choice straight from s.tok. Recorded `chunk` choices per submission with `depth` submissions in flight, so the
    /// GPU never waits for the CPU; the CPU reads each submission's chosen ids back (4 bytes a token, not the 600 KB
    /// of logits) and hands them to `on_token` in order — false (or an error) ends the read there: the steps already
    /// queued past it finish on the GPU unseen (their KV cache rows lie beyond the read and are never attended).
    /// The KV cache grows (Session::reserve) before a submission is recorded whose steps would write past its rows.
    pub async fn decode_gpu(&self, gpu: &Gpu, s: &mut Session, ch: &Choice, max: usize, mut on_token: impl FnMut(Token) -> Result<bool>) -> Result<()> {
        if ch.seed_len + max > ch.hist.size() as usize / 4 {
            bail!("{max} choices: the history holds {}", ch.hist.size() / 4 - ch.seed_len as u64);
        }
        let mut free: Vec<usize> = (0..ch.slots.len()).rev().collect();
        let mut queue: VecDeque<(usize, usize, wgpu::SubmissionIndex)> = VecDeque::new();
        let mut next = 0usize;
        loop {
            while next < max {
                let Some(slot) = free.pop() else { break };
                let n = ch.chunk.min(max - next);
                // the submission's steps (choice 0 has none) need their KV rows before the recording binds the cache
                let steps = n - (next == 0) as usize;
                s.reserve(gpu, s.n_past + steps).await?;
                let mut b = gpu.batch_cached(&s.binds);
                let mut pos = s.n_past;
                for i in next..next + n {
                    if i > 0 {
                        self.record_step(&mut b, s, pos)?;
                        pos += 1;
                    }
                    ch.record(&mut b, &s.logits, &s.tok, i)?;
                }
                b.copy(&ch.hist, ch.seed_len + next, &ch.slots[slot], 0, n);
                let idx = b.submit();
                s.n_past = pos;
                queue.push_back((slot, n, idx));
                next += n;
            }
            let Some((slot, n, idx)) = queue.pop_front() else { return Ok(()) };
            let ids = gpu.read_mapped_u32(&ch.slots[slot], n, idx).await?;
            free.push(slot);
            for id in ids {
                if !on_token(id as Token)? {
                    return Ok(());
                }
            }
        }
    }

    /// the decode step's layers on the F16 paths: 7 dispatches per layer (the F32 reference keeps its 18) —
    /// dmv_qkv (attn_norm fused) → qkv_post (q / k norms, RoPE, KV store) → split attention (2) → o (+ residual) →
    /// dmv_ffn (ffn_norm, gate, up, SwiGLU fused) → down (+ residual)
    fn layers_fused(&self, b: &mut crate::gpu::Batch, s: &Session, pos: usize) -> Result<()> {
        let c = &self.cfg;
        let (d, f, qd, kvd) = (c.dim, c.n_ff, c.n_head * c.d_head, c.n_kv * c.d_head);
        let (pos0, n_keys) = (pos as u32, pos + 1);
        let eps = c.eps.to_bits();
        let mv = |n: usize| (n.div_ceil(8 * DMV_R) as u32, 1, 1);
        let split = [n_keys as u32, c.n_head as u32, c.n_kv as u32, s.n_chunk as u32, (1.0 / (c.d_head as f32).sqrt()).to_bits()];
        for (li, ly) in self.layers.iter().enumerate() {
            b.dispatch("dmv_qkv", &[qd as u32, kvd as u32, d as u32, eps], &[&s.x, &ly.attn_norm, &ly.q.buf, &ly.k.buf, &ly.v.buf, &s.qkv], mv(qd + 2 * kvd))?;
            b.dispatch("qkv_post", &[c.n_head as u32, c.n_kv as u32, pos0, eps], &[&s.rope, &s.qkv, &ly.q_norm, &ly.k_norm, &s.k[li], &s.v[li]], ((c.n_head + 2 * c.n_kv) as u32, 1, 1))?;
            b.dispatch("attn_decode", &split, &[&s.qkv, &s.k[li], &s.v[li], &s.part], (n_keys.div_ceil(crate::gpu::DECODE_CHUNK) as u32, c.n_kv as u32, 1))?;
            b.dispatch("attn_merge", &split, &[&s.part, &s.a], (c.n_head as u32, 1, 1))?;
            b.dispatch("dmv", &[d as u32, qd as u32, 1, 0], &[&s.a, &ly.o.buf, &s.x], mv(d))?;
            b.dispatch("dmv_ffn", &[f as u32, d as u32, eps, 0], &[&s.x, &ly.ffn_norm, &ly.gate.buf, &ly.up.buf, &s.g], (f.div_ceil(8 * DMV_FFN_R) as u32, 1, 1))?;
            b.dispatch("dmv", &[d as u32, f as u32, 1, 0], &[&s.g, &ly.down.buf, &s.x], mv(d))?;
        }
        Ok(())
    }

    fn rope(&self, b: &mut crate::gpu::Batch, s: &Session, t: &Buf, n_tok: usize, n_head: usize, pos0: u32) -> Result<()> {
        let d = self.cfg.d_head;
        b.dispatch("rope_neox", &[n_tok as u32, n_head as u32, d as u32, pos0], &[&s.rope, t], (stride_groups((n_tok * n_head * d / 2) as u64), 1, 1))
    }

    /// output_norm + the tied output projection of the hidden row in s.x → s.logits
    fn head(&self, b: &mut crate::gpu::Batch, s: &Session) -> Result<()> {
        b.rms_norm(&s.x, &self.output_norm, &s.h, 1, self.cfg.dim, self.cfg.eps)?;
        b.matvec(&s.h, &self.tok_embd, &s.logits, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the F32 path (the exact kernels these tests check to f32 tolerances)
    fn f32_gpu() -> Gpu {
        pollster::block_on(Gpu::with_compute(Some(Compute::F32))).unwrap()
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// ggml quantize_row_q8_0_ref: d = amax / 127 (stored f16), q = round(x / d) → (the GGUF bytes, the dequantized values)
    fn q8_0(x: &[f32]) -> (Vec<u8>, Vec<f32>) {
        let (mut bytes, mut deq) = (Vec::new(), Vec::new());
        for blk in x.chunks_exact(32) {
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let dh = half::f16::from_f32(d);
            bytes.extend(dh.to_le_bytes());
            for &v in blk {
                let q = (v * id).round() as i8;
                bytes.push(q as u8);
                deq.push(q as f32 * dh.to_f32());
            }
        }
        (bytes, deq)
    }

    fn padded(mut b: Vec<u8>) -> Vec<u8> {
        while b.len() % 4 != 0 {
            b.push(0);
        }
        b
    }

    fn f16_round(v: f32) -> f32 {
        half::f16::from_f32(v).to_f32()
    }

    /// matvec (decode) and the tiled matmul (prefill) on q8_0 weights against the dequantized CPU product; 37 rows of
    /// 3 blocks: odd rows start their blocks on odd 2-byte boundaries
    #[test]
    fn q8_0_matvec_and_matmul_match_cpu_dequant() {
        let gpu = f32_gpu();
        let (m, n, k) = (70usize, 37usize, 96usize);
        let mut s = 11u64;
        let wf: Vec<f32> = (0..n * k).map(|_| lcg(&mut s) * 0.3).collect();
        let (bytes, wd) = q8_0(&wf);
        let w = Weight { buf: gpu.upload("w", &padded(bytes)), ty: WeightType::Q8_0, rows: n, cols: k };
        let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
        let base: Vec<f32> = (0..m * n).map(|_| lcg(&mut s)).collect();
        let want = |r: usize, c: usize| -> f32 { (0..k).map(|i| x[r * k + i] * wd[c * k + i]).sum() };
        for accumulate in [false, true] {
            let xb = gpu.upload_f32("x", &x[..k]);
            let yb = gpu.upload_f32("y", &base[..n]);
            let mut b = gpu.batch();
            b.matvec(&xb, &w, &yb, accumulate).unwrap();
            b.submit();
            let y = pollster::block_on(gpu.read_f32(&yb, n)).unwrap();
            for c in 0..n {
                let e = want(0, c) + if accumulate { base[c] } else { 0.0 };
                assert!((y[c] - e).abs() < 1e-4, "matvec acc={accumulate} [{c}] {} vs {e}", y[c]);
            }
            let xb = gpu.upload_f32("x", &x);
            let yb = gpu.upload_f32("y", &base);
            let mut b = gpu.batch();
            b.matmul(&xb, &w, &yb, m, accumulate).unwrap();
            b.submit();
            let y = pollster::block_on(gpu.read_f32(&yb, m * n)).unwrap();
            for r in 0..m {
                for c in 0..n {
                    let e = want(r, c) + if accumulate { base[r * n + c] } else { 0.0 };
                    assert!((y[r * n + c] - e).abs() < 1e-4, "matmul acc={accumulate} [{r},{c}] {} vs {e}", y[r * n + c]);
                }
            }
        }
    }

    /// the q8_0 embedding gather fills text rows and leaves SKIP rows alone
    #[test]
    fn embed_gather_dequantizes_rows() {
        let gpu = f32_gpu();
        let (vocab, dim) = (9usize, 64usize);
        let mut s = 5u64;
        let table: Vec<f32> = (0..vocab * dim).map(|_| lcg(&mut s)).collect();
        let (bytes, deq) = q8_0(&table);
        let w = gpu.upload("embd", &padded(bytes));
        let ids = [3u32, SKIP, 8, 0, 3];
        let tok = gpu.upload("tok", bytemuck::cast_slice(&ids));
        let x = gpu.upload_f32("x", &vec![7.0; ids.len() * dim]);
        let mut b = gpu.batch();
        b.dispatch("embed_q8_0", &[ids.len() as u32, dim as u32], &[&tok, &w, &x], (stride_groups((ids.len() * dim) as u64), 1, 1)).unwrap();
        b.submit();
        let got = pollster::block_on(gpu.read_f32(&x, ids.len() * dim)).unwrap();
        for (t, &id) in ids.iter().enumerate() {
            for i in 0..dim {
                let want = if id == SKIP { 7.0 } else { deq[id as usize * dim + i] };
                assert_eq!(got[t * dim + i], want, "row {t} [{i}]");
            }
        }
    }

    /// RoPE NEOX against a CPU rotation in f64 (pairs i, i + d/2, θ = pos · base^(−2i/d)) from position pos0
    #[test]
    fn rope_neox_matches_cpu() {
        let gpu = f32_gpu();
        let (n_tok, n_head, d, pos0, base) = (5usize, 3usize, 128usize, 1000usize, 1e6f32);
        let mut s = 9u64;
        let x: Vec<f32> = (0..n_tok * n_head * d).map(|_| lcg(&mut s)).collect();
        let cs: Vec<f32> = rope_table(base, d, pos0 + n_tok).into_iter().flatten().collect();
        let (csb, xb) = (gpu.upload_f32("cs", &cs), gpu.upload_f32("x", &x));
        let mut b = gpu.batch();
        b.dispatch("rope_neox", &[n_tok as u32, n_head as u32, d as u32, pos0 as u32], &[&csb, &xb], (stride_groups((n_tok * n_head * d / 2) as u64), 1, 1)).unwrap();
        b.submit();
        let y = pollster::block_on(gpu.read_f32(&xb, x.len())).unwrap();
        for t in 0..n_tok {
            for h in 0..n_head {
                for i in 0..d / 2 {
                    let theta = (pos0 + t) as f64 * (base as f64).powf(-((2 * i) as f64) / d as f64);
                    let o = (t * n_head + h) * d + i;
                    let (x0, x1) = (x[o] as f64, x[o + d / 2] as f64);
                    let (w0, w1) = (x0 * theta.cos() - x1 * theta.sin(), x0 * theta.sin() + x1 * theta.cos());
                    // f32 θ at position ~1000 is good to ~1e-4 rad
                    assert!((y[o] as f64 - w0).abs() < 5e-4 && (y[o + d / 2] as f64 - w1).abs() < 5e-4, "t {t} h {h} i {i}");
                }
            }
        }
    }

    /// causal GQA attention: the prefill kernel (batch at pos0 over the f16 cache) and the three decode kernels
    /// against a CPU softmax on the f16-rounded K / V
    #[test]
    fn attention_prefill_and_decode_match_cpu() {
        let gpu = f32_gpu();
        let (n_head, n_kv, d, pos0, n_q) = (4usize, 2usize, 128usize, 9usize, 70usize);
        let n_pos = pos0 + n_q;
        let kvd = n_kv * d;
        let scale = 1.0 / (d as f32).sqrt();
        let mut s = 21u64;
        let kf: Vec<f32> = (0..n_pos * kvd).map(|_| lcg(&mut s) * 2.0).collect();
        let vf: Vec<f32> = (0..n_pos * kvd).map(|_| lcg(&mut s)).collect();
        let q: Vec<f32> = (0..n_q * n_head * d).map(|_| lcg(&mut s) * 2.0).collect();
        let (kc, vc) = (gpu.scratch("k", n_pos * kvd / 2), gpu.scratch("v", n_pos * kvd / 2));
        let (kb, vb) = (gpu.upload_f32("kf", &kf), gpu.upload_f32("vf", &vf));
        let mut b = gpu.batch();
        b.dispatch("kv_store", &[n_pos as u32, kvd as u32, 0], &[&kb, &kc], (stride_groups((n_pos * kvd / 2) as u64), 1, 1)).unwrap();
        b.dispatch("kv_store", &[n_pos as u32, kvd as u32, 0], &[&vb, &vc], (stride_groups((n_pos * kvd / 2) as u64), 1, 1)).unwrap();
        b.submit();
        let reference = |qrow: &[f32], qpos: usize, h: usize| -> Vec<f64> {
            let kh = h / (n_head / n_kv);
            let sc: Vec<f64> = (0..=qpos)
                .map(|j| (0..d).map(|i| qrow[i] as f64 * f16_round(kf[j * kvd + kh * d + i]) as f64).sum::<f64>() * scale as f64)
                .collect();
            let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = sc.iter().map(|v| (v - mx).exp()).collect();
            let z: f64 = e.iter().sum();
            (0..d).map(|i| (0..=qpos).map(|j| e[j] / z * f16_round(vf[j * kvd + kh * d + i]) as f64).sum()).collect()
        };

        let (qb, ob) = (gpu.upload_f32("q", &q), gpu.scratch("o", n_q * n_head * d));
        let mut b = gpu.batch();
        let p = [n_q as u32, n_head as u32, n_kv as u32, pos0 as u32, scale.to_bits()];
        b.dispatch("attention_causal", &p, &[&qb, &kc, &vc, &ob], ((n_q as u32).div_ceil(64), n_head as u32, 1)).unwrap();
        b.submit();
        let o = pollster::block_on(gpu.read_f32(&ob, n_q * n_head * d)).unwrap();
        for t in [0usize, 1, 37, 63, 64, n_q - 1] {
            for h in 0..n_head {
                let base = (t * n_head + h) * d;
                let want = reference(&q[base..base + d], pos0 + t, h);
                for i in 0..d {
                    assert!((o[base + i] as f64 - want[i]).abs() < 1e-4, "prefill t {t} h {h} [{i}] {} vs {}", o[base + i], want[i]);
                }
            }
        }

        // decode: the last batch row's query over all n_pos keys
        let ql = &q[(n_q - 1) * n_head * d..];
        let qd = gpu.upload_f32("q1", ql);
        let n_ctx = n_pos + 3;
        let (sc, od) = (gpu.scratch("scores", n_head * n_ctx), gpu.scratch("o1", n_head * d));
        let p = [n_pos as u32, n_head as u32, n_kv as u32, n_ctx as u32, scale.to_bits()];
        let mut b = gpu.batch();
        b.dispatch("attn_scores", &p, &[&qd, &kc, &sc], (stride_groups((n_head * n_pos) as u64), 1, 1)).unwrap();
        b.dispatch("attn_softmax", &p, &[&sc], (n_head as u32, 1, 1)).unwrap();
        b.dispatch("attn_values", &p, &[&sc, &vc, &od], (n_head as u32, 1, 1)).unwrap();
        b.submit();
        let o = pollster::block_on(gpu.read_f32(&od, n_head * d)).unwrap();
        for h in 0..n_head {
            let want = reference(&ql[h * d..(h + 1) * d], n_pos - 1, h);
            for i in 0..d {
                assert!((o[h * d + i] as f64 - want[i]).abs() < 1e-4, "decode h {h} [{i}] {} vs {}", o[h * d + i], want[i]);
            }
        }
    }

    /// the fused decode kernels of the f16 paths (dmv, dmv_norm, dmv_qkv, dmv_ffn, qkv_post) against CPU references, on
    /// every f16 path this adapter offers (F16: workgroup-memory reductions, F16Coop: subgroup sums). K = 96: three
    /// q8_0 blocks per row, so odd rows start their blocks half-way into a word; row counts off the 8R grid.
    #[test]
    fn fused_decode_kernels_match_cpu() {
        for compute in [Compute::F16, Compute::F16Coop] {
            let Ok(gpu) = pollster::block_on(Gpu::with_compute(Some(compute))) else {
                eprintln!("{compute:?}: not offered by this adapter, skipped");
                continue;
            };
            let mut s = 41u64;
            let k = 96usize;
            let eps = 1e-6f32;
            let x: Vec<f32> = (0..k).map(|_| lcg(&mut s) * 2.0).collect();
            let nw: Vec<f32> = (0..k).map(|_| lcg(&mut s)).collect();
            let scale = 1.0 / ((x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / k as f64) as f32 + eps).sqrt();
            let xn: Vec<f32> = (0..k).map(|i| x[i] * scale * nw[i]).collect();
            let mat = |s: &mut u64, n: usize| {
                let wf: Vec<f32> = (0..n * k).map(|_| lcg(s) * 0.3).collect();
                let (bytes, deq) = q8_0(&wf);
                (gpu.upload("w", &padded(bytes)), deq)
            };
            let dotrow = |w: &[f32], r: usize, v: &[f32]| -> f32 { (0..k).map(|i| w[r * k + i] * v[i]).sum() };
            let close = |got: f32, want: f32, what: &str| assert!((got - want).abs() < 1e-3 * (1.0 + want.abs()), "{compute:?} {what}: {got} vs {want}");
            let (xb, nwb) = (gpu.upload_f32("x", &x), gpu.upload_f32("nw", &nw));

            // dmv (+ acc) and dmv_norm: 37 rows
            let n = 37usize;
            let (w, wd) = mat(&mut s, n);
            let base: Vec<f32> = (0..n).map(|_| lcg(&mut s)).collect();
            let (y, yn) = (gpu.upload_f32("y", &base), gpu.scratch("yn", n));
            let groups = (n.div_ceil(8 * DMV_R) as u32, 1, 1);
            let mut b = gpu.batch();
            b.dispatch("dmv", &[n as u32, k as u32, 1, 0], &[&xb, &w, &y], groups).unwrap();
            b.dispatch("dmv_norm", &[n as u32, k as u32, 0, eps.to_bits()], &[&xb, &nwb, &w, &yn], groups).unwrap();
            b.submit();
            let (got, gotn) = (pollster::block_on(gpu.read_f32(&y, n)).unwrap(), pollster::block_on(gpu.read_f32(&yn, n)).unwrap());
            for r in 0..n {
                close(got[r], base[r] + dotrow(&wd, r, &x), &format!("dmv row {r}"));
                close(gotn[r], dotrow(&wd, r, &xn), &format!("dmv_norm row {r}"));
            }

            // dmv_qkv: q 32 rows, k / v 16 rows each, into one buffer
            let (nq, nkv) = (2 * 8 * DMV_R, 8 * DMV_R);
            let ((wq, wqd), (wk, wkd), (wv, wvd)) = (mat(&mut s, nq), mat(&mut s, nkv), mat(&mut s, nkv));
            let qkv = gpu.scratch("qkv", nq + 2 * nkv);
            let mut b = gpu.batch();
            b.dispatch("dmv_qkv", &[nq as u32, nkv as u32, k as u32, eps.to_bits()], &[&xb, &nwb, &wq, &wk, &wv, &qkv], ((nq + 2 * nkv).div_ceil(8 * DMV_R) as u32, 1, 1)).unwrap();
            b.submit();
            let got = pollster::block_on(gpu.read_f32(&qkv, nq + 2 * nkv)).unwrap();
            for r in 0..nq {
                close(got[r], dotrow(&wqd, r, &xn), &format!("dmv_qkv q row {r}"));
            }
            for r in 0..nkv {
                close(got[nq + r], dotrow(&wkd, r, &xn), &format!("dmv_qkv k row {r}"));
                close(got[nq + nkv + r], dotrow(&wvd, r, &xn), &format!("dmv_qkv v row {r}"));
            }

            // dmv_ffn: silu(gate) · up, 37 rows
            let ((wg, wgd), (wu, wud)) = (mat(&mut s, n), mat(&mut s, n));
            let gb = gpu.scratch("g", n);
            let mut b = gpu.batch();
            b.dispatch("dmv_ffn", &[n as u32, k as u32, eps.to_bits(), 0], &[&xb, &nwb, &wg, &wu, &gb], (n.div_ceil(8 * DMV_FFN_R) as u32, 1, 1)).unwrap();
            b.submit();
            let got = pollster::block_on(gpu.read_f32(&gb, n)).unwrap();
            for r in 0..n {
                let (g, u) = (dotrow(&wgd, r, &xn), dotrow(&wud, r, &xn));
                close(got[r], g / (1.0 + (-g).exp()) * u, &format!("dmv_ffn row {r}"));
            }

            // qkv_post: 4 q heads, 2 kv heads of 128, position 5 of a 7-row cache
            let (n_head, n_kv, d, pos, n_ctx) = (4usize, 2usize, 128usize, 5usize, 7usize);
            let raw: Vec<f32> = (0..(n_head + 2 * n_kv) * d).map(|_| lcg(&mut s) * 3.0).collect();
            let (qn, kn): (Vec<f32>, Vec<f32>) = ((0..d).map(|_| lcg(&mut s)).collect(), (0..d).map(|_| lcg(&mut s)).collect());
            let cs: Vec<f32> = rope_table(1e6, d, n_ctx).into_iter().flatten().collect();
            let (rawb, csb, qnb, knb) = (gpu.upload_f32("qkv", &raw), gpu.upload_f32("cs", &cs), gpu.upload_f32("qn", &qn), gpu.upload_f32("kn", &kn));
            let (kc, vc) = (gpu.scratch("k", n_ctx * n_kv * d / 2), gpu.scratch("v", n_ctx * n_kv * d / 2));
            let mut b = gpu.batch();
            b.dispatch("qkv_post", &[n_head as u32, n_kv as u32, pos as u32, eps.to_bits()], &[&csb, &rawb, &qnb, &knb, &kc, &vc], ((n_head + 2 * n_kv) as u32, 1, 1)).unwrap();
            b.submit();
            let q = pollster::block_on(gpu.read_f32(&rawb, n_head * d)).unwrap();
            let kw = pollster::block_on(gpu.read_f32(&kc, n_ctx * n_kv * d / 2)).unwrap();
            let vw = pollster::block_on(gpu.read_f32(&vc, n_ctx * n_kv * d / 2)).unwrap();
            let unpack = |w: &[f32], i: usize| -> f32 {
                let bits = w[i / 2].to_bits();
                half::f16::from_bits(if i % 2 == 0 { bits as u16 } else { (bits >> 16) as u16 }).to_f32()
            };
            let theta = |i: usize| -> (f32, f32) { (cs[(pos * d / 2 + i) * 2], cs[(pos * d / 2 + i) * 2 + 1]) };
            let normed_roped = |h: usize, w: &[f32]| -> Vec<f32> {
                let row = &raw[h * d..(h + 1) * d];
                let sc = 1.0 / ((row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / d as f64) as f32 + eps).sqrt();
                let y: Vec<f32> = (0..d).map(|i| row[i] * sc * w[i]).collect();
                let mut out = y.clone();
                for i in 0..d / 2 {
                    let (c, sn) = theta(i);
                    out[i] = y[i] * c - y[i + d / 2] * sn;
                    out[i + d / 2] = y[i] * sn + y[i + d / 2] * c;
                }
                out
            };
            for h in 0..n_head {
                let want = normed_roped(h, &qn);
                for i in 0..d {
                    close(q[h * d + i], want[i], &format!("qkv_post q head {h} [{i}]"));
                }
            }
            let row0 = pos * n_kv * d; // the cache row's first f16
            for j in 0..n_kv {
                let want = normed_roped(n_head + j, &kn);
                for i in 0..d {
                    close(unpack(&kw, row0 + j * d + i), f16_round(want[i]), &format!("qkv_post k head {j} [{i}]"));
                    assert_eq!(unpack(&vw, row0 + j * d + i), f16_round(raw[(n_head + n_kv + j) * d + i]), "{compute:?} qkv_post v head {j} [{i}]");
                }
            }
        }
    }

    fn preset(rp: f64, window: usize, fp: f64, max: usize) -> Preset {
        Preset {
            name: "t".into(),
            temperature: 0.0,
            top_p: 1.0,
            repetition_penalty: rp,
            repetition_window: window,
            frequency_penalty: fp,
            max_new_tokens: max,
            gt_token_cap_factor: 2.0,
        }
    }

    /// the compute paths this adapter offers (the choose kernels are the same WGSL on every one)
    fn all_gpus() -> Vec<Gpu> {
        [Compute::F32, Compute::F16, Compute::F16Coop].into_iter().filter_map(|c| pollster::block_on(Gpu::with_compute(Some(c))).ok()).collect()
    }

    /// the GPU token choice (choose_pen → choose_max → choose_pick, Choice::record) against ocr-decode's choose on the
    /// same logits, step after step (the GPU keeps its own history + counts, the CPU its own): the real vocabulary
    /// and a small odd one; rp off / > 1 / < 1, windows 1 … 64 with seeds shorter and longer than the window and
    /// repeated tokens; fp 0 and > 0; logits with many exact ties (small integers, ±0), −inf (one step all −inf), a dominant token the
    /// penalties must push down (repeats), plain random values
    #[test]
    fn gpu_choice_matches_choose() {
        use ocr_decode::sampling::{Rng, choose};
        use std::collections::HashMap;
        for gpu in all_gpus() {
            let mut s = 77u64;
            for (n_vocab, rp, window, fp, seed_hist) in [
                (151936usize, 1.1, 20usize, 0.0, vec![5i32; 30]),
                (151936, 1.1, 20, 0.1, vec![3, 3, 9]),
                (1000, 1.0, 20, 0.0, vec![]),
                (1000, 1.3, 1, 0.0, vec![1, 2]),
                (1000, 0.7, 7, 0.37, vec![0, 999, 0, 4, 4, 4, 17, 999]),
                (1000, 1.1, 64, 0.1, (0..100).map(|i| (i * 7) % 13).collect()),
                (257, 2.5, 3, 1.5, vec![256]),
            ] {
                let p = preset(rp, window, fp, 40);
                let ch = Choice::new(&gpu, &p, &seed_hist, n_vocab, 1, 1).unwrap();
                let (lb, tb) = (gpu.scratch("logits", n_vocab), gpu.scratch("tok", 1));
                let mut history = seed_hist.clone();
                let mut counts: HashMap<Token, u32> = HashMap::new();
                for i in 0..p.max_new_tokens {
                    let mode = i % 4;
                    let mut l: Vec<f32> = (0..n_vocab)
                        .map(|_| match mode {
                            0 => (lcg(&mut s) * 3.0).round(), // ties everywhere, −0 / +0 among them
                            1 => lcg(&mut s) * 20.0,
                            2 => if lcg(&mut s) > 0.9 { f32::NEG_INFINITY } else { (lcg(&mut s) * 2.0).round() - 1.0 },
                            _ => lcg(&mut s) * 1e-3,
                        })
                        .collect();
                    if i == 7 {
                        l.fill(f32::NEG_INFINITY); // all −inf: token 0 (the CPU's argmax start)
                    }
                    // the last chosen token (or a history one) dominates by a little: the penalties decide
                    if let Some(&t) = history.last() {
                        let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        l[t as usize] = m + 0.05;
                    }
                    gpu.queue.write_buffer(lb.buffer(), lb.offset(), bytemuck::cast_slice(&l));
                    let mut b = gpu.batch();
                    ch.record(&mut b, &lb, &tb, i).unwrap();
                    b.submit();
                    let got = pollster::block_on(gpu.read_f32(&tb, 1)).unwrap()[0].to_bits() as Token;
                    let want = choose(&mut l, &p, &history, &counts, &mut Rng(1));
                    assert_eq!(got, want, "{:?} vocab {n_vocab} rp {rp} window {window} fp {fp} step {i} (mode {mode})", gpu.compute);
                    history.push(want);
                    *counts.entry(want).or_insert(0) += 1;
                }
                // the history the CPU reads back = the chosen tokens in order
                let h = pollster::block_on(gpu.read_f32(&ch.hist, ch.seed_len + p.max_new_tokens)).unwrap();
                let ids: Vec<Token> = h[ch.seed_len..].iter().map(|v| v.to_bits() as Token).collect();
                assert_eq!(ids, history[seed_hist.len()..], "{:?} history", gpu.compute);
            }
        }
    }

    /// choose_pen's division is IEEE-754 f32 division rounded to nearest-even (div_rn in integers: WGSL's own `/` is
    /// only good to 2.5 ULP) and its product the f32 product, bit for bit: 8192 distinct window tokens over logits of
    /// every magnitude (positive: random bit patterns incl. subnormals, the largest finite, ±0, +inf; negative:
    /// normal values), for penalties around 1 and far from it. A product below f32's normal range is not compared
    /// (GPUs may flush subnormal results to zero; a logit × rp never gets near 1e-38)
    #[test]
    fn choose_pen_is_exact() {
        let gpu = f32_gpu();
        let n = 8192usize;
        let mut s = 3u64;
        let mut bits = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 32) as u32
        };
        let mut l: Vec<f32> = (0..n)
            .map(|i| {
                let b = bits();
                if i % 3 == 2 {
                    -f32::from_bits(0x0d80_0000 + b % (0x7f00_0000 - 0x0d80_0000)) // −1e-30 … −1.7e38
                } else {
                    f32::from_bits(b % 0x7f80_0000) // 0 … the largest finite, subnormals included
                }
            })
            .collect();
        l[..6].copy_from_slice(&[0.0, -0.0, f32::INFINITY, f32::MAX, f32::from_bits(1), f32::MIN_POSITIVE]);
        let hist: Vec<Token> = (0..n as Token).collect();
        for rp in [1.1f64, 0.9, 1.0000001, 3.0, 1e-3, 1e3, 1.7e-30, 6.0e29] {
            let p = preset(rp, n, 0.0, 1);
            let ch = Choice::new(&gpu, &p, &hist, n, 1, 1).unwrap();
            let lb = gpu.upload_f32("logits", &l);
            let mut b = gpu.batch();
            b.dispatch("choose_pen", &[n as u32, n as u32, (rp as f32).to_bits(), 0], &[&ch.hist, &lb], (1, 1, 1)).unwrap();
            b.submit();
            let got = pollster::block_on(gpu.read_f32(&lb, n)).unwrap();
            let r = rp as f32;
            for (i, (&g, &x)) in got.iter().zip(&l).enumerate() {
                let want = if x < 0.0 { x * r } else { x / r };
                if x < 0.0 && !want.is_normal() {
                    continue; // a subnormal product: GPUs may flush it to −0 (no logit is that small)
                }
                assert!(g.to_bits() == want.to_bits(), "rp {rp}: {x:e} ({:#010x}) → {g:e} ({:#010x}), want {want:e} ({:#010x}) [#{i}]", x.to_bits(), g.to_bits(), want.to_bits());
            }
        }
    }

    fn rel_l2(got: &[f32], want: &[f64]) -> f64 {
        let (mut d, mut w) = (0f64, 0f64);
        for (&g, &x) in got.iter().zip(want) {
            d += (g as f64 - x) * (g as f64 - x);
            w += x * x;
        }
        (d / w).sqrt()
    }

    /// the first n f16 values of a buffer as f32
    fn read_f16(gpu: &Gpu, buf: &Buf, n: usize) -> Vec<f32> {
        let words = pollster::block_on(gpu.read_f32(buf, n.div_ceil(2))).unwrap();
        words.iter().flat_map(|w| {
            let b = w.to_bits();
            [half::f16::from_bits(b as u16).to_f32(), half::f16::from_bits((b >> 16) as u16).to_f32()]
        }).take(n).collect()
    }

    /// the vision encoder's f16-path pieces: bf16 → f16 weights at load (exact = half's round-to-nearest-even, the
    /// out-of-range count), rms_norm_h, matmul_f16x (accumulate), ffn_f16 (SwiGLU fused on F16Coop) against CPU
    /// references, on every f16 path this adapter offers
    #[test]
    fn vision_f16_kernels_match_cpu() {
        for compute in [Compute::F16, Compute::F16Coop] {
            let Ok(gpu) = pollster::block_on(Gpu::with_compute(Some(compute))) else {
                eprintln!("{compute:?}: not offered by this adapter, skipped");
                continue;
            };
            let mut s = 7u64;
            // the conversion: ordinary values, ties, subnormals, the range edge and two out-of-range values
            let mut vals: Vec<f32> = (0..4000).map(|_| lcg(&mut s) * 3.0).collect();
            vals.extend([1e-6, -3e-7, 6e-8, 65280.0, -65280.0, 1.0009765625, 70000.0, -1e6]); // 65280: the largest bf16 within f16
            let bytes: Vec<u8> = vals.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect();
            let mut w = Weight { buf: gpu.upload("w", &bytes), ty: WeightType::Bf16, rows: 1, cols: vals.len() };
            let over = gpu.scratch("over", 1);
            gpu.weight_to_f16(&mut w, &over, 1.0).unwrap();
            // the two out-of-range values share one word: counted once (a word count)
            assert_eq!(pollster::block_on(gpu.read_f32(&over, 1)).unwrap()[0].to_bits(), 1, "{compute:?}: out-of-range count");
            let got = read_f16(&gpu, &w.buf, vals.len() - 2);
            for (i, (&g, &v)) in got.iter().zip(&vals).enumerate() {
                let want = half::f16::from_f32(half::bf16::from_f32(v).to_f32()).to_f32();
                assert!(g.to_bits() == want.to_bits(), "{compute:?}: f16 of bf16 {v} (#{i}): {g} vs {want}");
            }
            // one layer's matmul-side pieces: M off the 64-row tile (buffers padded), K = 128, N = F = 128
            let (m, d, f) = (100usize, 128usize, 128usize);
            let mp = m.div_ceil(64) * 64;
            let x: Vec<f32> = (0..mp * d).map(|i| if i < m * d { lcg(&mut s) * 4.0 } else { 0.0 }).collect();
            let nw: Vec<f32> = (0..d).map(|_| 1.0 + lcg(&mut s) * 0.2).collect();
            let wf = |s: &mut u64, n: usize| -> (Weight, Vec<f32>) {
                let v: Vec<f32> = (0..n * d).map(|_| lcg(s) * 0.2).collect();
                let bytes: Vec<u8> = v.iter().flat_map(|&a| half::bf16::from_f32(a).to_le_bytes()).collect();
                let mut w = Weight { buf: gpu.upload("w", &bytes), ty: WeightType::Bf16, rows: n, cols: d };
                gpu.weight_to_f16(&mut w, &over, 4.0).unwrap();
                (w, v.iter().map(|&a| half::f16::from_f32(half::bf16::from_f32(a).to_f32() * 4.0).to_f32()).collect())
            };
            let (wq, wqv) = wf(&mut s, d);
            let (wg, wgv) = wf(&mut s, f);
            let (wu, wuv) = wf(&mut s, f);
            let (xb, nb) = (gpu.upload_f32("x", &x), gpu.upload_f32("norm", &nw));
            let (h16, g16) = (gpu.scratch("h16", mp * d / 2), gpu.scratch("g16", mp * f / 2));
            let yb = gpu.upload_f32("y", &x); // accumulate onto the residual
            let (gs, us) = (gpu.scratch("gate", mp * f), gpu.scratch("up", mp * f));
            let mut b = gpu.batch();
            b.rms_norm_h(&xb, &nb, &h16, m, d, 1e-5).unwrap();
            b.matmul_f16x(&h16, &wq, &yb, m, true).unwrap();
            b.ffn_f16(&h16, &wg, &wu, &g16, m, 0.125, &gs, &us).unwrap();
            b.submit();
            let h = read_f16(&gpu, &h16, m * d);
            let hw: Vec<f32> = (0..m * d)
                .map(|i| {
                    let r = &x[(i / d) * d..(i / d + 1) * d];
                    let ms = r.iter().map(|v| v * v).sum::<f32>() / d as f32;
                    half::f16::from_f32(x[i] / (ms + 1e-5).sqrt() * nw[i % d]).to_f32()
                })
                .collect();
            let e = rel_l2(&h, &hw.iter().map(|&v| v as f64).collect::<Vec<_>>());
            assert!(e < 1e-3, "{compute:?} rms_norm_h: relative L2 error {e:.2e}");
            let dot = |r: usize, w: &[f32], c: usize| (0..d).map(|j| h[r * d + j] as f64 * w[c * d + j] as f64).sum::<f64>();
            let y = pollster::block_on(gpu.read_f32(&yb, m * d)).unwrap();
            let want: Vec<f64> = (0..m * d).map(|i| x[i] as f64 + dot(i / d, &wqv, i % d)).collect();
            let e = rel_l2(&y, &want);
            assert!(e < 1e-4, "{compute:?} matmul_f16x: relative L2 error {e:.2e}");
            let g = read_f16(&gpu, &g16, m * f);
            let want: Vec<f64> = (0..m * f)
                .map(|i| {
                    let (a, u) = (dot(i / f, &wgv, i % f), dot(i / f, &wuv, i % f));
                    a / (1.0 + (-a).exp()) * u * 0.125
                })
                .collect();
            let e = rel_l2(&g, &want);
            assert!(e < 2e-3, "{compute:?} ffn_f16: relative L2 error {e:.2e}");
        }
    }

    /// the f16 compute paths' kernels (F16: f16 tiles, tiled vision attention; F16Coop: matrix units; both: the
    /// split decode attention) against f64 CPU references — within f16 rounding (relative L2 error), on every path
    /// this adapter offers
    #[test]
    fn f16_paths_match_cpu() {
        for compute in [Compute::F16, Compute::F16Coop] {
            let Ok(gpu) = pollster::block_on(Gpu::with_compute(Some(compute))) else {
                eprintln!("{compute:?}: not offered by this adapter, skipped");
                continue;
            };
            let mut s = 31u64;
            // matmul: bf16 and q8_0 weights, M / N off the tiles, accumulate onto Y
            let (m, n, k) = (150usize, 200usize, 1024usize);
            let x: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
            let wf: Vec<f32> = (0..n * k).map(|_| lcg(&mut s) * 0.1).collect();
            let base: Vec<f32> = (0..m * n).map(|_| lcg(&mut s)).collect();
            let wb: Vec<u8> = wf.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect();
            let wbv: Vec<f32> = wf.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();
            let (wq, wqv) = q8_0(&wf);
            for (ty, bytes, wv) in [(WeightType::Bf16, wb, wbv), (WeightType::Q8_0, padded(wq), wqv)] {
                let w = Weight { buf: gpu.upload("w", &bytes), ty, rows: n, cols: k };
                let (xb, yb) = (gpu.upload_f32("x", &x), gpu.upload_f32("y", &base));
                let mut b = gpu.batch();
                b.matmul(&xb, &w, &yb, m, true).unwrap();
                b.submit();
                let y = pollster::block_on(gpu.read_f32(&yb, m * n)).unwrap();
                let want: Vec<f64> = (0..m * n)
                    .map(|i| base[i] as f64 + (0..k).map(|j| x[(i / n) * k + j] as f64 * wv[(i % n) * k + j] as f64).sum::<f64>())
                    .collect();
                let e = rel_l2(&y, &want);
                assert!(e < 2e-3, "{compute:?} matmul {ty:?}: relative L2 error {e:.2e}");
            }
            // vision attention (head dim 64, all keys): 70 tokens × 2 heads
            let (nt, nh, d) = (70usize, 2usize, 64usize);
            let scale = 1.0 / (d as f32).sqrt();
            let q: Vec<f32> = (0..nt * nh * d).map(|_| lcg(&mut s) * 2.0).collect();
            let kk: Vec<f32> = (0..nt * nh * d).map(|_| lcg(&mut s) * 2.0).collect();
            let vv: Vec<f32> = (0..nt * nh * d).map(|_| lcg(&mut s)).collect();
            let (qb, kb, vb, ob) = (gpu.upload_f32("q", &q), gpu.upload_f32("k", &kk), gpu.upload_f32("v", &vv), gpu.scratch("o", nt * nh * d));
            let (kernel, per) = if compute == Compute::F16Coop { ("attention_coop", 32) } else { ("attention_tile", 64) };
            let n_pad = nt.div_ceil(32) * 32;
            let (kh, vh) = (gpu.scratch("kh", nh * n_pad * d / 2), gpu.scratch("vh", nh * n_pad * d / 2));
            let mut b = gpu.batch();
            b.dispatch("heads_f16", &[nt as u32, nh as u32, n_pad as u32, 0], &[&kb, &vb, &kh, &vh], (stride_groups((nh * n_pad * d / 4) as u64), 1, 1)).unwrap();
            b.dispatch(kernel, &[nt as u32, nh as u32, scale.to_bits(), n_pad as u32], &[&qb, &kh, &vh, &ob], ((nt as u32).div_ceil(per), nh as u32, 1)).unwrap();
            b.submit();
            // the output is f16 (the o projection's input)
            let o = read_f16(&gpu, &ob, nt * nh * d);
            let mut want = vec![0f64; nt * nh * d];
            for t in 0..nt {
                for h in 0..nh {
                    let at = |a: &[f32], r: usize, i: usize| a[(r * nh + h) * d + i] as f64;
                    let sc: Vec<f64> = (0..nt).map(|j| (0..d).map(|i| at(&q, t, i) * at(&kk, j, i)).sum::<f64>() * scale as f64).collect();
                    let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = sc.iter().map(|v| (v - mx).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for i in 0..d {
                        want[(t * nh + h) * d + i] = (0..nt).map(|j| e[j] / z * at(&vv, j, i)).sum();
                    }
                }
            }
            let e = rel_l2(&o, &want);
            assert!(e < 5e-3, "{compute:?} {kernel}: relative L2 error {e:.2e}");
            // the split decode attention: 300 keys (3 chunks) of the f16 cache, 4 q heads on 2 kv heads
            let (n_head, n_kv, dh, n_pos) = (4usize, 2usize, 128usize, 300usize);
            let kvd = n_kv * dh;
            let kf: Vec<f32> = (0..n_pos * kvd).map(|_| lcg(&mut s) * 2.0).collect();
            let vf: Vec<f32> = (0..n_pos * kvd).map(|_| lcg(&mut s)).collect();
            let ql: Vec<f32> = (0..n_head * dh).map(|_| lcg(&mut s) * 2.0).collect();
            let (kc, vc) = (gpu.scratch("k", n_pos * kvd / 2), gpu.scratch("v", n_pos * kvd / 2));
            let (kfb, vfb, qb) = (gpu.upload_f32("kf", &kf), gpu.upload_f32("vf", &vf), gpu.upload_f32("q", &ql));
            let n_chunk = n_pos.div_ceil(crate::gpu::DECODE_CHUNK);
            let (part, od) = (gpu.scratch("part", n_head * n_chunk * 130), gpu.scratch("o", n_head * dh));
            let sc = 1.0 / (dh as f32).sqrt();
            let p = [n_pos as u32, n_head as u32, n_kv as u32, n_chunk as u32, sc.to_bits()];
            let mut b = gpu.batch();
            b.dispatch("kv_store", &[n_pos as u32, kvd as u32, 0], &[&kfb, &kc], (stride_groups((n_pos * kvd / 2) as u64), 1, 1)).unwrap();
            b.dispatch("kv_store", &[n_pos as u32, kvd as u32, 0], &[&vfb, &vc], (stride_groups((n_pos * kvd / 2) as u64), 1, 1)).unwrap();
            b.dispatch("attn_decode", &p, &[&qb, &kc, &vc, &part], (n_chunk as u32, n_kv as u32, 1)).unwrap();
            b.dispatch("attn_merge", &p, &[&part, &od], (n_head as u32, 1, 1)).unwrap();
            b.submit();
            let o = pollster::block_on(gpu.read_f32(&od, n_head * dh)).unwrap();
            let mut want = vec![0f64; n_head * dh];
            for h in 0..n_head {
                let kh = h / (n_head / n_kv);
                let sco: Vec<f64> = (0..n_pos)
                    .map(|j| (0..dh).map(|i| ql[h * dh + i] as f64 * f16_round(kf[j * kvd + kh * dh + i]) as f64).sum::<f64>() * sc as f64)
                    .collect();
                let mx = sco.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = sco.iter().map(|v| (v - mx).exp()).collect();
                let z: f64 = e.iter().sum();
                for i in 0..dh {
                    want[h * dh + i] = (0..n_pos).map(|j| e[j] / z * f16_round(vf[j * kvd + kh * dh + i]) as f64).sum();
                }
            }
            let e = rel_l2(&o, &want);
            assert!(e < 1e-5, "{compute:?} split decode attention: relative L2 error {e:.2e}");
        }
    }

    /// a small random qwen3-shaped text model (q8_0 weights, norms near 1): 5 layers (two KV growth groups), dim 256,
    /// 2 q heads on 1 kv head of 128, ffn 512, 1000 tokens — the prefill + decode graph of every path at test size
    fn tiny_text_model(gpu: &Gpu) -> TextModel {
        let (n_layer, dim, n_head, n_kv, d_head, n_ff, n_vocab) = (5usize, 256usize, 2usize, 1usize, 128usize, 512usize, 1000usize);
        let mut s = 31u64;
        let mut w = |rows: usize, cols: usize, scale: f32| {
            let v: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut s) * scale).collect();
            Weight { buf: gpu.upload("w", &padded(q8_0(&v).0)), ty: WeightType::Q8_0, rows, cols }
        };
        let norm = |n: usize, k: usize| gpu.upload_f32("norm", &(0..n).map(|i| 1.0 + 0.01 * ((i * 7 + k) % 13) as f32).collect::<Vec<f32>>());
        let (qd, kvd) = (n_head * d_head, n_kv * d_head);
        let layers = (0..n_layer)
            .map(|i| Layer {
                attn_norm: norm(dim, i),
                q: w(qd, dim, 0.2),
                k: w(kvd, dim, 0.2),
                v: w(kvd, dim, 0.2),
                o: w(dim, qd, 0.1),
                q_norm: norm(d_head, i + 1),
                k_norm: norm(d_head, i + 2),
                ffn_norm: norm(dim, i + 3),
                gate: w(n_ff, dim, 0.2),
                up: w(n_ff, dim, 0.2),
                down: w(dim, n_ff, 0.1),
            })
            .collect();
        let cfg = TextConfig { n_layer, dim, n_head, n_kv, d_head, n_ff, eps: 1e-6, rope_base: 1e6, n_vocab, n_ctx_train: 4096 };
        TextModel { tok_embd: w(n_vocab, dim, 1.0), output_norm: norm(dim, 0), layers, cfg }
    }

    /// a KV cache grown mid-read (every 1 / 3 / 61 rows: growths inside a run-ahead submission's window, across the
    /// 128-key attention chunks, the layer groups) gives the same logits, bit for bit, and the same tokens as one sized
    /// to the read up front — the CPU-choice step and the run-ahead GPU choice (4 × 2 in flight), on every path; the
    /// cache holds the rows reserved, counted in the ledger, and is gone with the session
    #[test]
    fn kv_growth_matches_presized_cache() {
        for gpu in all_gpus() {
            let m = tiny_text_model(&gpu);
            let mut seed = 5u64;
            let prompt: Vec<Token> = (0..20).map(|_| ((lcg(&mut seed) + 1.0) * 499.0) as Token).collect();
            let (n_new, n_steps) = (150usize, 40usize);
            let n_ctx = prompt.len() + n_new;
            let row_bytes = (2 * m.cfg.n_layer * m.cfg.n_kv * m.cfg.d_head / 2 * 4) as u64;
            let base_kv = gpu.mem_live().kv_cache;
            let mut want: Option<(Vec<Vec<f32>>, Vec<Token>)> = None;
            for grow in [n_ctx, 1, 3, 61] {
                // the CPU-choice path: prefill, then n_steps steps of the greedy token, every logits row kept
                let mut s = m.session_grown(&gpu, n_ctx, grow).unwrap();
                let mut logits = vec![pollster::block_on(m.prefill(&gpu, &mut s, &[Input::Tokens(&prompt)])).unwrap()];
                assert_eq!(s.kv_rows(), (prompt.len() + grow).min(n_ctx), "grow {grow}: rows after the prefill");
                for _ in 0..n_steps {
                    let l = logits.last().unwrap();
                    let t = (0..l.len()).fold(0, |a, i| if l[i] > l[a] { i } else { a });
                    logits.push(pollster::block_on(m.step(&gpu, &mut s, t as Token)).unwrap());
                }
                assert_eq!(gpu.mem_live().kv_cache - base_kv, s.kv_rows() as u64 * row_bytes, "grow {grow}: the ledger's KV cache");
                drop(s);
                assert_eq!(gpu.mem_live().kv_cache, base_kv, "grow {grow}: the KV cache outlived its session");
                // the GPU choice, run ahead
                let mut s = m.session_grown(&gpu, n_ctx, grow).unwrap();
                pollster::block_on(m.prefill(&gpu, &mut s, &[Input::Tokens(&prompt)])).unwrap();
                let p = preset(1.1, 20, 0.0, n_new);
                let ch = Choice::new(&gpu, &p, &prompt, m.cfg.n_vocab, 4, 2).unwrap();
                let mut toks: Vec<Token> = Vec::new();
                pollster::block_on(m.decode_gpu(&gpu, &mut s, &ch, n_new, |t| {
                    toks.push(t);
                    Ok(true)
                }))
                .unwrap();
                assert_eq!(toks.len(), n_new);
                // the last choice needs no row; the cache holds at most `grow` rows past the read
                assert_eq!(s.n_past, n_ctx - 1, "grow {grow}");
                assert!(s.kv_rows() >= s.n_past && s.kv_rows() <= (s.n_past + grow).min(n_ctx), "grow {grow}: {} rows", s.kv_rows());
                match &want {
                    None => want = Some((logits, toks)),
                    Some((wl, wt)) => {
                        for (i, (a, b)) in wl.iter().zip(&logits).enumerate() {
                            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{:?} grow {grow}: logits of step {i} differ", gpu.compute);
                        }
                        assert_eq!(wt, &toks, "{:?} grow {grow}: the GPU-chosen tokens differ", gpu.compute);
                    }
                }
            }
        }
    }
}
