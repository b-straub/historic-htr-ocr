// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The LightOnOCR vision encoder + projector on wgpu — llama.cpp's clip.cpp graph for PROJECTOR_TYPE_LIGHTONOCR
// (models/pixtral.cpp + clip_graph::build_vit), op for op:
//   pixels/255 → (x − mean)/std (clip_image_f32::normalize) → 14×14 patch conv, no bias (build_inp; normalise +
//   im2col on the GPU here, patchify.wgsl, f32 — llama.cpp's im2col rounds the pixels to f16 because the kernel is f32) → pre_ln RMSNorm →
//   24 × [RMSNorm ln1 → q/k/v (no bias) → 2D RoPE θ=10000 on q, k → softmax(q·kᵀ/√64)·v over ALL patches → attn_out,
//   + residual → RMSNorm ln2 → silu(gate)·up → down, + residual] (no post_ln) →
//   mm.input_norm RMSNorm → 2×2 unfold (im2col) → mm.patch_merger → mm.1 → gelu (tanh form) → mm.2.
// No [IMG_BREAK] tokens (LightOnOCR has none): n_tokens = (w/28)·(h/28).
// The image must already be baked (both sides on the 28 grid, longest ≤ image_size): then mtmd's longest-edge
// preprocess is the identity (llama.cpp built to downscale only) — any other size is an error here, not a resize.
// Weights: bf16 stay bf16 on the GPU (packed two per u32, unpacked exactly in the matmul — half the memory of f32,
// no f16 range / precision loss, no shader-f16 feature needed in browsers); the f32 tensors stay f32.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use anyhow::{Context, Result, bail};
use image::RgbImage;

use crate::gguf::{Gguf, TensorSource, TensorType};
#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::STAGING_BYTES;
use crate::gpu::{Batch, Buf, Compute, FFN_ROWS, Gpu, Precision, Uploader, Weight, WeightType, stride_groups};

/// clip.cpp sets rope_theta = 10000 for LIGHTONOCR in code (load_hparams); the GGUF carries no key for it
const ROPE_THETA: f32 = 10000.0;

/// the f16 paths keep the FFN's SwiGLU product (the down projection's input) divided by this and its weights times
/// this (powers of two: exact, the product unchanged): the product reached 3.6e4 in a test page's last layer
/// (≤ 1e3 on the other test pages), f16 ends at 65504 — stored ÷ 64 it has 117× headroom there; values below
/// 4e-3 lose relative precision (subnormal f16: absolute error ≤ 2e-6, nothing next to the sums)
const FFN_DOWN_SCALE: f32 = 64.0;

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub n_layer: usize,
    pub dim: usize,
    pub n_head: usize,
    pub d_head: usize,
    pub n_ff: usize,
    pub patch: usize,
    pub merge: usize,
    pub image_size: usize,
    pub proj_dim: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl VisionConfig {
    pub fn from_gguf<S: TensorSource>(g: &Gguf<S>) -> Result<Self> {
        let proj = g.str("clip.projector_type")?;
        if proj != "lightonocr" {
            bail!("projector {proj:?}: this engine runs lightonocr only");
        }
        if !g.bool("clip.use_silu")? {
            bail!("clip.use_silu is false: the encoder's FFN would not be SwiGLU");
        }
        let three = |k: &str| -> Result<[f32; 3]> {
            let v = g.f32_array(k)?;
            v.as_slice().try_into().with_context(|| format!("{k} has {} values", v.len()))
        };
        let dim = g.u32("clip.vision.embedding_length")? as usize;
        let n_head = g.u32("clip.vision.attention.head_count")? as usize;
        if dim % n_head != 0 || dim / n_head != 64 {
            bail!("head dim {dim}/{n_head}: the attention kernel is written for 64");
        }
        Ok(Self {
            n_layer: g.u32("clip.vision.block_count")? as usize,
            dim,
            n_head,
            d_head: dim / n_head,
            n_ff: g.u32("clip.vision.feed_forward_length")? as usize,
            patch: g.u32("clip.vision.patch_size")? as usize,
            merge: g.u32("clip.vision.spatial_merge_size")? as usize,
            image_size: g.u32("clip.vision.image_size")? as usize,
            proj_dim: g.u32("clip.vision.projection_dim")? as usize,
            eps: g.f32("clip.vision.attention.layer_norm_epsilon")?,
            rope_theta: ROPE_THETA,
            mean: three("clip.vision.image_mean")?,
            std: three("clip.vision.image_std")?,
        })
    }
}

struct Layer {
    ln1: Buf,
    ln2: Buf,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
}

pub struct Vision {
    pub cfg: VisionConfig,
    /// the pixel normalisation per channel: lut[c·256 + v] = (v / 255 − mean[c]) / std[c] (patchify.wgsl)
    lut: Buf,
    patch_embd: Weight,
    pre_ln: Buf,
    layers: Vec<Layer>,
    input_norm: Buf,
    merger: Weight,
    mm1: Weight,
    mm2: Weight,
}

pub struct Embeddings {
    pub n_tokens: usize,
    pub dim: usize,
    /// row-major [n_tokens][dim]
    pub data: Vec<f32>,
}

/// a weight matrix; with `over` (the f16 paths) a bf16 one is converted to f16 on the GPU (out-of-range count → over),
/// ffn_down times FFN_DOWN_SCALE
async fn weight<S: TensorSource>(gpu: &Gpu, up: &mut Uploader, g: &Gguf<S>, name: &str, rows: usize, cols: usize, over: Option<&Buf>) -> Result<Weight> {
    let t = g.info(name)?;
    // ggml order: the last dimension is the output row, the others (contiguous) make one row
    let (last, inner) = t.shape.split_last().context("tensor without shape")?;
    if *last as usize != rows || inner.iter().product::<u64>() as usize != cols {
        bail!("{name}: shape {:?}, expected {cols} values × {rows} rows", t.shape);
    }
    let ty = match t.ty {
        TensorType::BF16 => WeightType::Bf16,
        TensorType::F32 => WeightType::F32,
        other => bail!("{name}: {other:?} weights are not supported in the vision encoder"),
    };
    let mut w = Weight { buf: gpu.upload_tensor(up, &g.source, t).await?, ty, rows, cols };
    if let (Some(over), WeightType::Bf16) = (over, ty) {
        let scale = if name.ends_with(".ffn_down.weight") { FFN_DOWN_SCALE } else { 1.0 };
        gpu.weight_to_f16(&mut w, over, scale)?;
    }
    Ok(w)
}

async fn vector<S: TensorSource>(gpu: &Gpu, up: &mut Uploader, g: &Gguf<S>, name: &str, n: usize) -> Result<Buf> {
    let t = g.info(name)?;
    if t.ty != TensorType::F32 || t.shape != [n as u64] {
        bail!("{name}: {:?} {:?}, expected F32 [{n}]", t.ty, t.shape);
    }
    gpu.upload_tensor(up, &g.source, t).await
}

/// clip_image_f32 from_u8 + normalize for every byte value: [c·256 + v] = (v / 255 − mean[c]) / std[c] (f32)
fn pixel_lut(c: &VisionConfig) -> Vec<f32> {
    (0..3).flat_map(|ch| (0..256).map(move |v| (v as f32 / 255.0 - c.mean[ch]) / c.std[ch])).collect()
}

/// cos / sin of the 2D RoPE as ggml_rope_cache_init computes them (f32, θ multiplied up step by step):
/// [half][pos][k], half 0 = rows (freq_scale 1), half 1 = columns (freq_scale = base^(−2/d))
pub fn rope_table(theta: f32, d_head: usize, max_pos: usize) -> Vec<[f32; 2]> {
    let n_dims = d_head / 2;
    let theta_scale = theta.powf(-2.0 / n_dims as f32);
    let mut out = Vec::with_capacity(2 * max_pos * n_dims / 2);
    for half in 0..2 {
        let freq_scale = if half == 0 { 1.0 } else { theta.powf(-2.0 / d_head as f32) };
        for pos in 0..max_pos {
            let mut t = pos as f32;
            for _ in 0..n_dims / 2 {
                let a = freq_scale * t;
                out.push([a.cos(), a.sin()]);
                t *= theta_scale;
            }
        }
    }
    out
}

impl Vision {
    /// the weights streamed file → GPU through one staging buffer of `staging` bytes (STAGING_BYTES by default)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(gpu: &Gpu, path: &Path, staging: Option<u64>) -> Result<Self> {
        let g = Gguf::open(path)?;
        let mut up = gpu.uploader(staging.unwrap_or(STAGING_BYTES));
        pollster::block_on(Self::from_gguf(gpu, &g, &mut up))
    }

    /// the weights of an opened mmproj GGUF (any TensorSource: a file, OPFS, HTTP ranges) streamed through `up`
    pub async fn from_gguf<S: TensorSource>(gpu: &Gpu, g: &Gguf<S>, up: &mut Uploader) -> Result<Self> {
        let cfg = VisionConfig::from_gguf(g)?;
        let (d, f) = (cfg.dim, cfg.n_ff);
        // the f16 paths compute the matmuls from f16 weights: bf16 → f16 once here (each matmul rounded every tile so
        // before); a weight beyond f16's range refuses the load (it would be inf)
        let over_buf = (gpu.compute != Compute::F32).then(|| gpu.scratch("f16 range", 1));
        let over = over_buf.as_ref();
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let n = |s: &str| format!("v.blk.{i}.{s}.weight");
            layers.push(Layer {
                ln1: vector(gpu, up, g, &n("ln1"), d).await?,
                ln2: vector(gpu, up, g, &n("ln2"), d).await?,
                q: weight(gpu, up, g, &n("attn_q"), d, d, over).await?,
                k: weight(gpu, up, g, &n("attn_k"), d, d, over).await?,
                v: weight(gpu, up, g, &n("attn_v"), d, d, over).await?,
                o: weight(gpu, up, g, &n("attn_out"), d, d, over).await?,
                gate: weight(gpu, up, g, &n("ffn_gate"), f, d, over).await?,
                up: weight(gpu, up, g, &n("ffn_up"), f, d, over).await?,
                down: weight(gpu, up, g, &n("ffn_down"), d, f, over).await?,
            });
        }
        // tensors the graph would use if present: none of them may exist unread
        for absent in ["v.patch_embd.bias", "v.post_ln.weight", "mm.1.bias", "mm.2.bias", "v.token_embd.img_break", "mm.input_norm.bias"] {
            if g.has(absent) {
                bail!("{absent} is in the GGUF but not part of this engine's graph");
            }
        }
        let pp = 3 * cfg.patch * cfg.patch;
        let mm = cfg.merge * cfg.merge;
        let v = Self {
            lut: gpu.upload_f32("pixel lut", &pixel_lut(&cfg)),
            patch_embd: weight(gpu, up, g, "v.patch_embd.weight", d, pp, over).await?,
            pre_ln: vector(gpu, up, g, "v.pre_ln.weight", d).await?,
            layers,
            input_norm: vector(gpu, up, g, "mm.input_norm.weight", d).await?,
            merger: weight(gpu, up, g, "mm.patch_merger.weight", d, d * mm, over).await?,
            mm1: weight(gpu, up, g, "mm.1.weight", cfg.proj_dim, d, over).await?,
            mm2: weight(gpu, up, g, "mm.2.weight", cfg.proj_dim, cfg.proj_dim, over).await?,
            cfg,
        };
        if let Some(o) = &over_buf {
            let n = gpu.read_f32(o, 1).await?[0].to_bits();
            if n > 0 {
                bail!("{n} words of vision weights hold values beyond f16's range (65504): this model cannot run on the {:?} compute path (OCR_WGPU_COMPUTE=f32 natively)", gpu.compute);
            }
        }
        gpu.wait().await?;
        Ok(v)
    }

    /// the patch grid (px, py) of a baked image; any other size is an error, not a resize
    fn grid(&self, img: &RgbImage) -> Result<(usize, usize)> {
        let c = &self.cfg;
        let (w, h) = (img.width() as usize, img.height() as usize);
        let grid = c.patch * c.merge;
        if w % grid != 0 || h % grid != 0 || w.max(h) > c.image_size || w == 0 || h == 0 {
            bail!("image {w}x{h} is not baked (both sides on the {grid} px grid, longest ≤ {})", c.image_size);
        }
        Ok((w / c.patch, h / c.patch))
    }

    /// the image's embeddings for the text model ([n_tokens][proj_dim]); `layers` = Some(n) stops after n encoder
    /// layers and returns the residual stream [n_patches][dim] instead (Some(0) = after pre_ln) — the bisect hook
    /// `prec`: Precision::F32 normally; LlamaMetal emulates llama.cpp Metal's rounding (gpu.rs) for the bisect
    pub async fn encode(&self, gpu: &Gpu, img: &RgbImage, layers: Option<usize>, prec: Precision) -> Result<Embeddings> {
        let (buf, n_tokens, dim) = self.run(gpu, img, layers, prec)?;
        Ok(Embeddings { n_tokens, dim, data: gpu.read_f32(&buf, n_tokens * dim).await? })
    }

    /// the image's embeddings left on the GPU for the text decoder: (buffer [n_tokens][proj_dim] f32 — padded, so it
    /// may be larger —, n_tokens, proj_dim)
    pub fn encode_gpu(&self, gpu: &Gpu, img: &RgbImage) -> Result<(Buf, usize, usize)> {
        self.run(gpu, img, None, Precision::F32)
    }

    /// the encoder's dispatches, submitted: (the buffer holding the result, rows, row width)
    fn run(&self, gpu: &Gpu, img: &RgbImage, layers: Option<usize>, prec: Precision) -> Result<(Buf, usize, usize)> {
        if gpu.compute != Compute::F32 && prec == Precision::F32 {
            return self.run_f16(gpu, img, layers);
        }
        let c = &self.cfg;
        let (px, py) = self.grid(img)?;
        let n = px * py;
        let (d, f) = (c.dim, c.n_ff);
        let pp = 3 * c.patch * c.patch;
        let rgb = upload_pixels(gpu, img);
        let inp = gpu.scratch("patches", n * pp);
        let x = gpu.scratch("x", n * d);
        let hb = gpu.scratch("h", n * d);
        let (q, k, v, a) = (gpu.scratch("q", n * d), gpu.scratch("k", n * d), gpu.scratch("v", n * d), gpu.scratch("a", n * d));
        let (g, u) = (gpu.scratch("gate", n * f), gpu.scratch("up", n * f));
        let max_pos = px.max(py);
        let csb = self.rope(gpu, max_pos);
        let n_layers = self.n_layers(layers)?;

        let mut b = gpu.batch();
        let patchify = [img.width(), px as u32, n as u32, c.patch as u32];
        b.dispatch("patchify", &patchify, &[&rgb, &self.lut, &inp], (stride_groups((n * pp) as u64), 1, 1))?;
        b.matmul_p(&inp, &self.patch_embd, &hb, n, false, prec)?;
        b.rms_norm(&hb, &self.pre_ln, &x, n, d, c.eps)?;
        let rope = [n as u32, c.n_head as u32, c.d_head as u32, px as u32, max_pos as u32];
        let rope_groups = (stride_groups((n * c.n_head * c.d_head / 2) as u64), 1, 1);
        let attn = [n as u32, c.n_head as u32, (1.0 / (c.d_head as f32).sqrt()).to_bits(), (prec == Precision::LlamaMetal) as u32];
        for l in &self.layers[..n_layers] {
            b.rms_norm(&x, &l.ln1, &hb, n, d, c.eps)?;
            b.matmul_p(&hb, &l.q, &q, n, false, prec)?;
            b.matmul_p(&hb, &l.k, &k, n, false, prec)?;
            b.matmul_p(&hb, &l.v, &v, n, false, prec)?;
            b.dispatch("rope2d", &rope, &[&csb, &q], rope_groups)?;
            b.dispatch("rope2d", &rope, &[&csb, &k], rope_groups)?;
            b.dispatch("attention", &attn, &[&q, &k, &v, &a], ((n as u32).div_ceil(64), c.n_head as u32, 1))?;
            b.matmul_p(&a, &l.o, &x, n, true, prec)?;
            b.rms_norm(&x, &l.ln2, &hb, n, d, c.eps)?;
            b.matmul_p(&hb, &l.gate, &g, n, false, prec)?;
            b.matmul_p(&hb, &l.up, &u, n, false, prec)?;
            b.dispatch("swiglu", &[(n * f) as u32], &[&u, &g], (stride_groups((n * f) as u64), 1, 1))?;
            b.matmul_p(&g, &l.down, &x, n, true, prec)?;
        }
        if layers.is_some() {
            b.submit();
            return Ok((x, n, d));
        }
        let (out, n_out) = self.project(&mut b, &x, [&hb, &g, &q, &k, &v], px, py, prec)?;
        b.submit();
        Ok((out, n_out, c.proj_dim))
    }

    /// the f16 paths (F16 / F16Coop): what stays f32 and what is f16 —
    ///   f32: the residual stream x (it reaches ~1.6e5 in a few channels, beyond f16's 65504: an f16 mmproj
    ///        overflowed to NaN on CUDA), the norms' sums, every matmul / attention accumulator, q / k / v and RoPE,
    ///        the softmax (max, exp, sum), the projector tail;
    ///   f16: the matmul inputs — the normed activations (rms_norm_h), the attention output, the SwiGLU product — and
    ///        the weights (bf16 → f16 at load), K / V for the attention, P in P·V (attention_coop). These are exactly
    ///        the values the f16 kernels rounded on their way into the tiles / matrix units before (round 3 moved the
    ///        rounding into the producers: half the bytes, same numbers).
    /// Rows are padded to whole 64-row tiles (matmul_coopd reads / writes them; they hold zeros or unused values).
    fn run_f16(&self, gpu: &Gpu, img: &RgbImage, layers: Option<usize>) -> Result<(Buf, usize, usize)> {
        let c = &self.cfg;
        let (px, py) = self.grid(img)?;
        let n = px * py;
        let mp = n.div_ceil(64) * 64;
        let (d, f) = (c.dim, c.n_ff);
        let pp = 3 * c.patch * c.patch;
        let prec = Precision::F32;
        let rgb = upload_pixels(gpu, img);
        let inp = gpu.scratch("patches", n * pp);
        let x = gpu.scratch("x", mp * d);
        let (q, k, v) = (gpu.scratch("q", mp * d), gpu.scratch("k", mp * d), gpu.scratch("v", mp * d));
        let h16 = gpu.scratch("h f16", mp * d / 2);
        let a16 = gpu.scratch("a f16", mp * d / 2);
        let g16 = gpu.scratch("g f16", mp * f / 2);
        // F16 (no fused FFN kernel): the gate / up products in f32, one row block at a time (gpu.rs FFN_ROWS)
        let (g, u) = if gpu.compute == Compute::F16 { (gpu.scratch("gate", FFN_ROWS * f), gpu.scratch("up", FFN_ROWS * f)) } else { (gpu.scratch("-", 1), gpu.scratch("-", 1)) };
        // K / V as f16 head-major, padded to whole key blocks (heads_f16.wgsl) for the attention
        let n_pad = n.div_ceil(32) * 32;
        let (kh, vh) = (gpu.scratch("k f16", c.n_head * n_pad * c.d_head / 2), gpu.scratch("v f16", c.n_head * n_pad * c.d_head / 2));
        let heads = [n as u32, c.n_head as u32, n_pad as u32, 0];
        let heads_groups = (stride_groups((c.n_head * n_pad * c.d_head / 4) as u64), 1, 1);
        let max_pos = px.max(py);
        let csb = self.rope(gpu, max_pos);
        let n_layers = self.n_layers(layers)?;

        let mut b = gpu.batch();
        let patchify = [img.width(), px as u32, n as u32, c.patch as u32];
        b.dispatch("patchify", &patchify, &[&rgb, &self.lut, &inp], (stride_groups((n * pp) as u64), 1, 1))?;
        b.matmul_p(&inp, &self.patch_embd, &q, n, false, prec)?;
        b.rms_norm(&q, &self.pre_ln, &x, n, d, c.eps)?;
        let rope = [n as u32, c.n_head as u32, c.d_head as u32, px as u32, max_pos as u32];
        let rope_groups = (stride_groups((n * c.n_head * c.d_head / 2) as u64), 1, 1);
        let attn = [n as u32, c.n_head as u32, (1.0 / (c.d_head as f32).sqrt()).to_bits(), n_pad as u32];
        for l in &self.layers[..n_layers] {
            b.rms_norm_h(&x, &l.ln1, &h16, n, d, c.eps)?;
            b.matmul_f16x(&h16, &l.q, &q, n, false)?;
            b.matmul_f16x(&h16, &l.k, &k, n, false)?;
            b.matmul_f16x(&h16, &l.v, &v, n, false)?;
            b.dispatch("rope2d", &rope, &[&csb, &q], rope_groups)?;
            b.dispatch("rope2d", &rope, &[&csb, &k], rope_groups)?;
            b.dispatch("heads_f16", &heads, &[&k, &v, &kh, &vh], heads_groups)?;
            if gpu.compute == Compute::F16Coop {
                b.dispatch("attention_coop", &attn, &[&q, &kh, &vh, &a16], ((n as u32).div_ceil(32), c.n_head as u32, 1))?;
            } else {
                b.dispatch("attention_tile", &attn, &[&q, &kh, &vh, &a16], ((n as u32).div_ceil(64), c.n_head as u32, 1))?;
            }
            b.matmul_f16x(&a16, &l.o, &x, n, true)?;
            b.rms_norm_h(&x, &l.ln2, &h16, n, d, c.eps)?;
            b.ffn_f16(&h16, &l.gate, &l.up, &g16, n, 1.0 / FFN_DOWN_SCALE, &g, &u)?;
            b.matmul_f16x(&g16, &l.down, &x, n, true)?;
        }
        if layers.is_some() {
            b.submit();
            return Ok((x, n, d));
        }
        // the projector tail with f32 activations (the merger / mm weights f16): q, k, v reused
        let (out, n_out) = self.project(&mut b, &x, [&q, &k, &v, &q, &k], px, py, prec)?;
        b.submit();
        Ok((out, n_out, c.proj_dim))
    }


    /// mm.input_norm → 2×2 unfold → patch merger → mm.1 → gelu → mm.2 on the residual x: bufs = [norm, unfolded
    /// (n · d), merged, mm.1 out, mm.2 out] (a buffer may come again once its value is dead); (the result, n_out)
    #[allow(clippy::too_many_arguments)]
    fn project(&self, b: &mut Batch, x: &Buf, bufs: [&Buf; 5], px: usize, py: usize, prec: Precision) -> Result<(Buf, usize)> {
        let c = &self.cfg;
        let (d, f) = (c.dim, c.n_ff);
        let n = px * py;
        let [norm, unf, merged, m1, out] = bufs;
        let mm = c.merge * c.merge;
        if c.merge != 2 || f < d * mm {
            bail!("merge {} / n_ff {f}: the unfold kernel is 2×2", c.merge);
        }
        let n_out = n / mm;
        b.rms_norm(x, &self.input_norm, norm, n, d, c.eps)?;
        // the unfolded tokens [n_out][d·4] = n · d values
        b.dispatch("unfold2x2", &[px as u32, py as u32, d as u32], &[norm, unf], (stride_groups((n * d) as u64), 1, 1))?;
        b.matmul_p(unf, &self.merger, merged, n_out, false, prec)?;
        b.matmul_p(merged, &self.mm1, m1, n_out, false, prec)?;
        b.dispatch("gelu", &[(n_out * c.proj_dim) as u32], &[m1], (stride_groups((n_out * c.proj_dim) as u64), 1, 1))?;
        b.matmul_p(m1, &self.mm2, out, n_out, false, prec)?;
        Ok((out.clone(), n_out))
    }

    fn rope(&self, gpu: &Gpu, max_pos: usize) -> Buf {
        let c = &self.cfg;
        let cs: Vec<f32> = rope_table(c.rope_theta, c.d_head, max_pos).into_iter().flatten().collect();
        gpu.upload_f32("rope", &cs)
    }

    fn n_layers(&self, layers: Option<usize>) -> Result<usize> {
        let n = layers.unwrap_or(self.cfg.n_layer);
        if n > self.cfg.n_layer {
            bail!("{n} layers asked, the encoder has {}", self.cfg.n_layer);
        }
        Ok(n)
    }
}

/// the RGB bytes as they are (padded to whole words); patchify.wgsl normalises + im2cols them on the GPU
fn upload_pixels(gpu: &Gpu, img: &RgbImage) -> Buf {
    let raw = img.as_raw();
    if raw.len() % 4 == 0 {
        gpu.upload("pixels", raw)
    } else {
        let mut padded = raw.clone();
        padded.resize(raw.len().div_ceil(4) * 4, 0);
        gpu.upload("pixels", &padded)
    }
}
