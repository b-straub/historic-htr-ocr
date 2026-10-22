// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// One baked image → a streamed read, on wgpu alone: the vision encoder (vision.rs) → the prompt → the text decoder
// (text.rs) → ocr-decode's read loop.
// The prompt reproduces llama.cpp's mtmd prompt token for token (checked against llama.cpp's tokenizer):
//   PREFIX · <|im_start|> · the image's embeddings · <|im_end|> · SUFFIX
// — mtmd wraps a LIGHTONOCR image in <|im_start|> … <|im_end|> text chunks (mtmd.cpp, PROJECTOR_TYPE_LIGHTONOCR);
// positions run on over the image (n_pos = n_tokens). The repetition window starts on n_image × <|image_pad|> + SUFFIX
// (as hf/window_rep_penalty.py does for Transformers / vLLM).
// The engine is async (the browser's WebGPU answers through the event loop): `from_sources` + `load_weights` +
// `read_inspect_async` are THE implementation; `open` / `read` / `read_inspect` are the native blocking wrappers
// (pollster).
// Weights per read (Safari bills a page's WebGPU memory to it — 3.0 GB after a read; reloading the weights is cheap):
// the Engine between reads is the device with its compiled kernels (Gpu), the tokenizer and the model's
// format — NO weights. `load_and_read` loads them STAGED (vision → encode → vision freed → text: never both sets
// at once; files → GPU, bf16 → f16 on the f16 paths), reads and frees them, a failed or cancelled read too (the Model is a local of that call: every way out drops it, and a dropped Buf
// is destroyed). What stays costs little memory and would cost time to rebuild: the device + pipelines (created and
// warmed up once, gpu.rs Gpu::create), the kernels' parameter ring (0.3 MiB), the tokenizer (CPU).
// Token choice (TokenChoice, decided per read): a greedy preset chooses on the GPU — the penalties + argmax in kernels
// (text.rs Choice), the decode run ahead of the CPU (text.rs decode_gpu), only the chosen ids read back; a sampled
// preset, or a caller that taps every step's logits (the parity tools), keeps ocr-decode's generate on the read-back
// logits. Both feed the same ocr-decode Emitter (EOG, UTF-8 streaming, cancel), and ReadStats says which one ran.
// GPU memory (gpu.rs Buf / Ledger): the load reports the weights (vision / text) and the resident bytes (ModelMemory),
// every read its resident bytes, its peak and each phase's peak (ReadStats.gpu_memory) — counted, not estimated. The
// load first checks the largest weight against the device's buffer limits (Gpu::check_fits): a GPU that cannot hold
// it is refused with the numbers before anything is uploaded. Free memory cannot be checked: no API tells it.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use web_time::Instant;

use anyhow::{Result, bail};
use image::RgbImage;
use ocr_decode::presets::Preset;
use ocr_decode::prompt::{IMAGE_PAD, PREFIX, SUFFIX};
use ocr_decode::sampling::{Emitter, Flow, Phase, ReadStats, Step, Token, generate};

use crate::gguf::{Gguf, TensorInfo, TensorSource};
use crate::gpu::{Buf, Gpu, STAGING_BYTES, Uploader, tensor_buffer_bytes};
use crate::text::{Choice, Input, Session, TextModel, gpu_choice_capable};
use crate::tokenizer::Tokenizer;
use crate::vision::Vision;

pub const TEXT_GGUF: &str = "text-q8_0.gguf";
pub const MMPROJ_GGUF: &str = "mmproj-bf16.gguf";

/// mtmd.cpp: the text chunks around a LIGHTONOCR image
const IMAGE_BEGIN: &str = "<|im_start|>";
const IMAGE_END: &str = "<|im_end|>";

/// the image as the decoder sees it: the pixels (encoded here) or embeddings from elsewhere ([n][dim] f32 — the parity
/// check feeds llama.cpp's mtmd embeddings)
pub enum ImageInput<'a> {
    Pixels(&'a RgbImage),
    Embeddings(&'a [f32]),
}

/// the vision stage of a staged read: the image's embeddings on the GPU, the vision weights already freed
struct VisionStage {
    img_buf: Buf,
    n_img: usize,
    encode_ms: f64,
    vision_weights: u64,
    resident: u64,
    load_ms: f64,
    load_peak: u64,
}

/// the run-ahead decode's defaults: choices recorded per submission × submissions in flight (natively
/// OCR_WGPU_RUNAHEAD=<chunk>[,<depth>] overrides them — a measuring knob)
pub const RUN_AHEAD_CHUNK: usize = 4;
pub const RUN_AHEAD_DEPTH: usize = 2;

/// where a read's next token is chosen — a capability decision per read (Engine::token_choice), reported in ReadStats
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TokenChoice {
    /// penalties + argmax on the GPU, the decode run ahead of the CPU (greedy presets)
    Gpu,
    /// ocr-decode's choose on the logits read back each step (sampled presets, a logits tap, or forced)
    Cpu,
}

impl TokenChoice {
    pub fn label(self) -> &'static str {
        match self {
            TokenChoice::Gpu => "gpu",
            TokenChoice::Cpu => "cpu",
        }
    }
}

/// the per-step logits a caller taps (the parity tools): `(step, raw logits)` — a read with a tap chooses on the CPU
pub type LogitsTap<'a> = &'a mut dyn FnMut(usize, &[f32]);

/// the model's GPU memory after a weight load (bytes, counted by the ledger)
#[derive(Clone, Copy, Debug)]
pub struct ModelMemory {
    pub vision_weights: u64,
    pub text_weights: u64,
    /// every buffer alive after the load: the weights + the kernels' uniform params + small tables
    pub resident: u64,
    /// the most bytes alive at once during the load
    pub load_peak: u64,
}

/// one weight load: its wall time, the parts of it, its GPU memory
#[derive(Clone, Copy, Debug)]
pub struct WeightLoad {
    /// the whole load: the files read, uploaded, the bf16 weights converted, the checks
    pub ms: f64,
    /// of it: reading the model files (TensorSource::read_at)
    pub read_ms: f64,
    /// of it: waiting for the GPU to take each staging chunk (the copy to the GPU)
    pub wait_ms: f64,
    pub memory: ModelMemory,
}

/// the model's weights on the GPU — they live for one read (Engine::load_and_read), dropped (= destroyed) after it
pub struct Model {
    vision: Vision,
    text: TextModel,
    pub load: WeightLoad,
}

/// a read with its own weight load (Engine::load_and_read)
#[derive(Clone, Debug)]
pub struct LoadedRead {
    pub load: WeightLoad,
    pub stats: ReadStats,
    /// the GPU memory alive after the read, its weights freed (the parameter ring: ≈ 0.3 MiB)
    pub idle_bytes: u64,
}

/// the engine between reads: the device with its compiled kernels, the tokenizer, the model's format — no weights
pub struct Engine {
    pub gpu: Gpu,
    /// the formats the weights are stored in ("q8_0 text · bf16 vision", ocr_gguf::weights_label) — the compute format
    /// is gpu.compute_label()
    pub weights: String,
    tok: Tokenizer,
}

struct WgpuStep<'a> {
    gpu: &'a Gpu,
    text: &'a TextModel,
    tok: &'a Tokenizer,
    s: Session,
    last: Vec<f32>,
}

impl Step for WgpuStep<'_> {
    fn n_vocab(&self) -> usize {
        self.text.cfg.n_vocab
    }

    fn logits(&mut self, out: &mut [f32]) -> Result<()> {
        out.copy_from_slice(&self.last);
        Ok(())
    }

    async fn feed(&mut self, t: Token) -> Result<()> {
        self.last = self.text.step(self.gpu, &mut self.s, t).await?;
        Ok(())
    }

    fn is_eog(&self, t: Token) -> bool {
        self.tok.is_eog(t)
    }

    fn piece(&self, t: Token, out: &mut Vec<u8>) -> Result<()> {
        self.tok.piece(t, out)
    }
}

/// the two model files of a model dir (text-q8_0.gguf + mmproj-bf16.gguf, like ocr-core), headers parsed — the
/// weights are read from them at each read
#[cfg(not(target_arch = "wasm32"))]
pub struct ModelFiles {
    pub text: Gguf<crate::gguf::FileSource>,
    pub mmproj: Gguf<crate::gguf::FileSource>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ModelFiles {
    pub fn open(dir: &Path) -> Result<Self> {
        Ok(Self { text: Gguf::open(&dir.join(TEXT_GGUF))?, mmproj: Gguf::open(&dir.join(MMPROJ_GGUF))? })
    }
}

impl Engine {
    /// the engine for a model dir's files (no weights uploaded: each read loads them)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(files: &ModelFiles) -> Result<Self> {
        pollster::block_on(Self::from_sources(&files.text, &files.mmproj))
    }

    /// the engine for two GGUFs from any TensorSource (a file, an OPFS access handle, HTTP ranges): the device with its
    /// kernels, the pre-check of the largest weight against the device's limits, the tokenizer — no weights
    pub async fn from_sources<T: TensorSource, M: TensorSource>(text: &Gguf<T>, mmproj: &Gguf<M>) -> Result<Self> {
        let weights = crate::gguf::weights_label(text, mmproj)?;
        let gpu = Gpu::new_async().await?;
        // the pre-check, before anything is uploaded: every weight is ONE buffer bound whole to a kernel, so the largest
        // must fit maxBufferSize and maxStorageBufferBindingSize (the activations depend on the image: Batch::dispatch
        // checks each binding with its numbers)
        let largest = largest_tensor(&text.tensors).into_iter().chain(largest_tensor(&mmproj.tensors)).max_by_key(|t| tensor_buffer_bytes(t));
        let Some(largest) = largest else { bail!("the model files hold no tensors") };
        gpu.check_fits(&format!("the model's largest weight ({})", largest.name), tensor_buffer_bytes(largest))?;
        // the weight pool's blocks hold the largest tensor: the text weights fill the freed vision blocks (with 64 MiB
        // blocks the text's 157.6 MiB token_embd needed a block of its own on top — 990 MiB for 790 MiB of need)
        gpu.weight_block(tensor_buffer_bytes(largest));
        let tok = Tokenizer::from_gguf(text)?;
        gpu.alive()?;
        Ok(Self { gpu, weights, tok })
    }

    /// the weights onto the GPU, streamed through one staging chunk; they live as long as the Model
    pub async fn load_weights<T: TensorSource, M: TensorSource>(&self, text: &Gguf<T>, mmproj: &Gguf<M>) -> Result<Model> {
        self.check_files(text, mmproj)?;
        let gpu = &self.gpu;
        let t0 = Instant::now();
        gpu.mem_begin_load();
        let base = gpu.mem_live().weights;
        let mut up = gpu.uploader(STAGING_BYTES);
        let vision = Vision::from_gguf(gpu, mmproj, &mut up).await?;
        let vision_weights = gpu.mem_live().weights - base;
        let text = TextModel::load(gpu, text, &mut up).await?;
        let t1 = Instant::now();
        gpu.upload_flush(&mut up).await?;
        up.wait_ms += t1.elapsed().as_secs_f64() * 1e3;
        if self.tok.n_vocab() != text.cfg.n_vocab {
            bail!("{} tokens in the vocabulary, {} rows in token_embd", self.tok.n_vocab(), text.cfg.n_vocab);
        }
        gpu.alive()?;
        let live = gpu.mem_live();
        let memory = ModelMemory { vision_weights, text_weights: live.weights - base - vision_weights, resident: live.total(), load_peak: gpu.mem_peak() };
        let load = WeightLoad { ms: t0.elapsed().as_secs_f64() * 1e3, read_ms: up.read_ms, wait_ms: up.wait_ms, memory };
        Ok(Model { vision, text, load })
    }

    /// the files are the model the engine was opened with (its weight formats), the device alive
    fn check_files<T: TensorSource, M: TensorSource>(&self, text: &Gguf<T>, mmproj: &Gguf<M>) -> Result<()> {
        self.gpu.alive()?;
        let label = crate::gguf::weights_label(text, mmproj)?;
        if label != self.weights {
            bail!("the model files changed since the engine was opened: {label}, was {}", self.weights);
        }
        Ok(())
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(&mut self, files: &ModelFiles, img: &RgbImage, p: &Preset, seed: u64, on_phase: impl FnMut(Phase), on_text: impl FnMut(&str) -> bool) -> Result<LoadedRead> {
        pollster::block_on(self.load_and_read(&files.text, &files.mmproj, ImageInput::Pixels(img), p, seed, on_phase, on_text, None))
    }

    /// where a read with preset `p` chooses its tokens: on the GPU when the preset allows it (text.rs
    /// gpu_choice_capable: greedy) and no logits tap asks for every step's logits; natively OCR_WGPU_TOKEN_CHOICE=cpu
    /// forces the CPU path (the A/B baseline), =gpu refuses a preset that cannot (never a silent fallback)
    pub fn token_choice(&self, p: &Preset, tapped: bool) -> Result<TokenChoice> {
        #[cfg(not(target_arch = "wasm32"))]
        match std::env::var("OCR_WGPU_TOKEN_CHOICE").ok().as_deref() {
            None => {}
            Some("cpu") => return Ok(TokenChoice::Cpu),
            Some("gpu") => {
                if let Err(why) = gpu_choice_capable(p) {
                    bail!("OCR_WGPU_TOKEN_CHOICE=gpu, preset '{}': {why}", p.name);
                }
                if tapped {
                    bail!("OCR_WGPU_TOKEN_CHOICE=gpu with a logits tap: the tap needs every step's logits on the CPU");
                }
                return Ok(TokenChoice::Gpu);
            }
            Some(v) => bail!("OCR_WGPU_TOKEN_CHOICE={v}: cpu or gpu"),
        }
        Ok(if gpu_choice_capable(p).is_ok() && !tapped { TokenChoice::Gpu } else { TokenChoice::Cpu })
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::too_many_arguments)]
    pub fn read_inspect(
        &mut self,
        files: &ModelFiles,
        image: ImageInput<'_>,
        p: &Preset,
        seed: u64,
        on_phase: impl FnMut(Phase),
        on_text: impl FnMut(&str) -> bool,
        on_logits: Option<LogitsTap<'_>>,
    ) -> Result<LoadedRead> {
        pollster::block_on(self.load_and_read(&files.text, &files.mmproj, image, p, seed, on_phase, on_text, on_logits))
    }

    /// the vision weights loaded, the image encoded, the vision weights (and the encoder's work) freed — the staged
    /// read's first half; `base` = the weight bytes alive before
    async fn stage_vision<M: TensorSource>(&self, mmproj: &Gguf<M>, img: &RgbImage, up: &mut Uploader, base: u64, on_phase: &mut impl FnMut(Phase)) -> Result<VisionStage> {
        let gpu = &self.gpu;
        let t0 = Instant::now();
        gpu.mem_begin_load();
        let vision = Vision::from_gguf(gpu, mmproj, up).await?;
        let t1 = Instant::now();
        gpu.upload_flush(up).await?;
        up.wait_ms += t1.elapsed().as_secs_f64() * 1e3;
        let load_ms = t0.elapsed().as_secs_f64() * 1e3;
        let load_peak = gpu.mem_peak();
        let vision_weights = gpu.mem_live().weights - base;
        let resident = gpu.mem_live().total();
        let t2 = Instant::now();
        on_phase(Phase::EncodingImage);
        gpu.phase("vision");
        let (img_buf, n_img, _) = vision.encode_gpu(gpu, img)?;
        gpu.wait().await?;
        let encode_ms = t2.elapsed().as_secs_f64() * 1e3;
        drop(vision);
        gpu.phase("-");
        Ok(VisionStage { img_buf, n_img, encode_ms, vision_weights, resident, load_ms, load_peak })
    }

    /// THE read of the hosts: the weights loaded from the two GGUFs, the read, the weights freed — also when the read
    /// fails or is cancelled (the weights are locals: every way out of this call drops them). Pixels are read STAGED:
    /// the vision weights load, the image is encoded, the vision weights are freed, then the text weights load — the
    /// two weight sets are never on the GPU together (the peak: the larger of vision weights + encoder work and text
    /// weights + KV cache, not their sum; iPadOS killed the page at the second read's load). Given embeddings need no
    /// vision: the text weights only.
    #[allow(clippy::too_many_arguments)]
    pub async fn load_and_read<T: TensorSource, M: TensorSource>(
        &mut self,
        text: &Gguf<T>,
        mmproj: &Gguf<M>,
        image: ImageInput<'_>,
        p: &Preset,
        seed: u64,
        mut on_phase: impl FnMut(Phase),
        on_text: impl FnMut(&str) -> bool,
        on_logits: Option<LogitsTap<'_>>,
    ) -> Result<LoadedRead> {
        self.check_files(text, mmproj)?;
        let choice = self.token_choice(p, on_logits.is_some())?;
        let gpu = &self.gpu;
        let base = gpu.mem_live().weights;
        let mut up = gpu.uploader(STAGING_BYTES);
        let mut load_ms = 0.0;
        let mut load_peak = 0;
        gpu.mem_begin_read();
        let (img_buf, n_img, encode_ms, vision_weights, resident) = match image {
            ImageInput::Pixels(img) => {
                // the vision weights (and the encoder's work) go before the text loads
                let v = self.stage_vision(mmproj, img, &mut up, base, &mut on_phase).await?;
                load_ms += v.load_ms;
                load_peak = load_peak.max(v.load_peak);
                on_phase(Phase::LoadingText { encode_ms: v.encode_ms });
                (Some(v.img_buf), v.n_img, Some(v.encode_ms), v.vision_weights, v.resident)
            }
            ImageInput::Embeddings(_) => (None, 0, None, 0, gpu.mem_live().total()),
        };
        let t3 = Instant::now();
        gpu.mem_begin_load();
        let text_model = TextModel::load(gpu, text, &mut up).await?;
        let t4 = Instant::now();
        gpu.upload_flush(&mut up).await?;
        up.wait_ms += t4.elapsed().as_secs_f64() * 1e3;
        load_ms += t3.elapsed().as_secs_f64() * 1e3;
        load_peak = load_peak.max(gpu.mem_peak());
        if self.tok.n_vocab() != text_model.cfg.n_vocab {
            bail!("{} tokens in the vocabulary, {} rows in token_embd", self.tok.n_vocab(), text_model.cfg.n_vocab);
        }
        gpu.alive()?;
        let live = gpu.mem_live();
        let text_weights = live.weights - base;
        let memory = ModelMemory { vision_weights, text_weights, resident: resident.max(live.total()), load_peak };
        let load = WeightLoad { ms: load_ms, read_ms: up.read_ms, wait_ms: up.wait_ms, memory };
        let dim = text_model.cfg.dim;
        let (img_buf, n_img) = match (img_buf, image) {
            (Some(b), _) => (b, n_img),
            (None, ImageInput::Embeddings(e)) => {
                if e.is_empty() || e.len() % dim != 0 {
                    bail!("{} embedding values are no whole rows of {dim}", e.len());
                }
                (gpu.upload_f32("image embeddings", e), e.len() / dim)
            }
            (None, ImageInput::Pixels(_)) => bail!("pixels read without their encoding"),
        };
        let read = self.read_text(&text_model, img_buf, n_img, encode_ms, choice, resident, p, seed, on_phase, on_text, on_logits).await;
        drop(text_model);
        let stats = read?;
        Ok(LoadedRead { load, stats, idle_bytes: self.gpu.mem_live().total() })
    }

    /// one read with loaded weights + `on_logits(step, logits)` (each step's raw logits — it makes the read choose on
    /// the CPU) + the image as pixels or as given embeddings; `on_phase` hears EncodingImage → ReadingPrompt → Decoding
    /// as each starts
    #[allow(clippy::too_many_arguments)]
    pub async fn read_inspect_async(
        &mut self,
        model: &Model,
        image: ImageInput<'_>,
        p: &Preset,
        seed: u64,
        mut on_phase: impl FnMut(Phase),
        on_text: impl FnMut(&str) -> bool,
        on_logits: Option<LogitsTap<'_>>,
    ) -> Result<ReadStats> {
        let choice = self.token_choice(p, on_logits.is_some())?;
        // a dead device (lost, or an uncaptured GPU error) fails here with its reason, never hangs
        self.gpu.alive()?;
        self.gpu.mem_begin_read();
        let resident = self.gpu.mem_live().total();
        let t0 = Instant::now();
        let dim = model.text.cfg.dim;
        on_phase(Phase::EncodingImage);
        self.gpu.phase("vision");
        let (img_buf, n_img) = match image {
            ImageInput::Pixels(img) => {
                let (buf, n, _) = model.vision.encode_gpu(&self.gpu, img)?;
                (buf, n)
            }
            ImageInput::Embeddings(e) => {
                if e.is_empty() || e.len() % dim != 0 {
                    bail!("{} embedding values are no whole rows of {dim}", e.len());
                }
                (self.gpu.upload_f32("image embeddings", e), e.len() / dim)
            }
        };
        self.gpu.wait().await?;
        let encode_ms = t0.elapsed().as_secs_f64() * 1e3;
        self.read_text(&model.text, img_buf, n_img, Some(encode_ms), choice, resident, p, seed, on_phase, on_text, on_logits).await
    }

    /// the text part of a read: the prompt around the image's embeddings (img_buf, n_img rows), the prefill, the
    /// decode; `resident` = the bytes alive at the read's start (ReadStats.gpu_memory)
    #[allow(clippy::too_many_arguments)]
    async fn read_text(
        &mut self,
        text: &TextModel,
        img_buf: Buf,
        n_img: usize,
        encode_ms: Option<f64>,
        choice: TokenChoice,
        resident: u64,
        p: &Preset,
        seed: u64,
        mut on_phase: impl FnMut(Phase),
        on_text: impl FnMut(&str) -> bool,
        mut on_logits: Option<LogitsTap<'_>>,
    ) -> Result<ReadStats> {
        let prefix = self.tok.encode(PREFIX)?;
        let (beg, end) = (self.tok.encode(IMAGE_BEGIN)?, self.tok.encode(IMAGE_END)?);
        let suffix = self.tok.encode(SUFFIX)?;
        let n_prompt = prefix.len() + beg.len() + n_img + end.len() + suffix.len();
        // the session (KV cache) is the prefill's memory: the phase starts before it is made
        self.gpu.phase("prefill");
        let mut s = text.session_for(&self.gpu, n_prompt + p.max_new_tokens, n_img)?;
        let t1 = Instant::now();
        on_phase(Phase::ReadingPrompt { encode_ms });
        let last = text.prefill(
            &self.gpu,
            &mut s,
            &[Input::Tokens(&prefix), Input::Tokens(&beg), Input::Embeddings { buf: &img_buf, n: n_img }, Input::Tokens(&end), Input::Tokens(&suffix)],
            )
            .await?;
        drop(img_buf);
        let prefill_ms = t1.elapsed().as_secs_f64() * 1e3;
        on_phase(Phase::Decoding { encode_ms, prefill_ms });

        let pad = self.tok.id(IMAGE_PAD)?;
        let history: Vec<Token> = std::iter::repeat_n(pad, n_img).chain(suffix.iter().copied()).collect();
        let t2 = Instant::now();
        self.gpu.phase("decode");
        let o = match choice {
            TokenChoice::Cpu => {
                let mut step = WgpuStep { gpu: &self.gpu, text, tok: &self.tok, s, last };
                let tap = |i: usize, l: &[f32]| {
                    if let Some(f) = on_logits.as_mut() {
                        f(i, l)
                    }
                };
                generate(&mut step, p, seed, history, on_text, tap).await?
            }
            TokenChoice::Gpu => {
                drop(last); // the first token is chosen from the prompt's logits still on the GPU (s.logits)
                let (chunk, depth) = run_ahead()?;
                let ch = Choice::new(&self.gpu, p, &history, text.cfg.n_vocab, chunk, depth)?;
                let mut em = Emitter::new(on_text);
                let mut flow = Flow::Go;
                let tok = &self.tok;
                text
                    .decode_gpu(&self.gpu, &mut s, &ch, p.max_new_tokens, |t| {
                        flow = em.push(tok.is_eog(t), |b| tok.piece(t, b))?;
                        Ok(flow == Flow::Go)
                    })
                    .await?;
                em.finish(flow)?
            }
        };
        let gpu_memory = self.gpu.mem_read(resident)?;
        self.gpu.phase("-");
        Ok(ReadStats {
            image_tokens: n_img,
            prompt_tokens: n_prompt,
            encode_ms,
            prefill_ms,
            decode_ms: t2.elapsed().as_secs_f64() * 1e3,
            out_tokens: o.out_tokens,
            stopped_on_eog: o.stopped_on_eog,
            cancelled: o.cancelled,
            token_choice: Some(choice.label()),
            gpu_memory: Some(gpu_memory),
        })
    }
}

/// the largest tensor's buffer of one GGUF (None: no tensors)
fn largest_tensor(tensors: &[TensorInfo]) -> Option<&TensorInfo> {
    tensors.iter().max_by_key(|t| tensor_buffer_bytes(t))
}

/// the run-ahead's (chunk, depth): the defaults, natively OCR_WGPU_RUNAHEAD=<chunk>[,<depth>]
fn run_ahead() -> Result<(usize, usize)> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(v) = std::env::var("OCR_WGPU_RUNAHEAD") {
        use anyhow::Context;
        let parse = |x: &str| x.trim().parse::<usize>().with_context(|| format!("OCR_WGPU_RUNAHEAD={v}: <chunk>[,<depth>]"));
        return Ok(match v.split_once(',') {
            Some((c, d)) => (parse(c)?, parse(d)?),
            None => (parse(&v)?, RUN_AHEAD_DEPTH),
        });
    }
    Ok((RUN_AHEAD_CHUNK, RUN_AHEAD_DEPTH))
}
