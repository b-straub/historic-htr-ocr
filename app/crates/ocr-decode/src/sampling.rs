// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The decoding step, applied on the logits exactly as app/presets.json defines it: windowed repetition penalty over
// the history (prompt tail + output), frequency penalty over the output so far, then greedy or temperature + top-p
// with the page's seed — and the read loop around it (stop on end-of-generation or maxNewTokens, stream complete
// UTF-8). An engine runs THIS loop behind the Step trait (the wgpu text decoder; a llama.cpp context fits it too).
// The loop is async because the browser's WebGPU hands the logits back only through the event loop; native callers
// drive it with pollster::block_on.

use std::collections::{HashMap, HashSet};
use std::future::Future;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::presets::Preset;

/// a token id (llama.cpp's llama_token is an i32)
pub type Token = i32;

/// SplitMix64 — the sampled read's random source, seeded per page (seedRule); not MLX's generator (same settings,
/// not the same sample, like the app's Local engine)
pub struct Rng(pub u64);

impl Rng {
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// one decoding step on a copy of the logits: the presets' knobs in order rp → fp → greedy / T + top-p
pub fn choose(logits: &mut [f32], p: &Preset, history: &[Token], counts: &HashMap<Token, u32>, rng: &mut Rng) -> Token {
    if p.repetition_penalty != 1.0 {
        let rp = p.repetition_penalty as f32;
        let mut seen = HashSet::new();
        for &t in &history[history.len().saturating_sub(p.repetition_window)..] {
            if seen.insert(t) {
                let l = &mut logits[t as usize];
                *l = if *l < 0.0 { *l * rp } else { *l / rp };
            }
        }
    }
    if p.frequency_penalty > 0.0 {
        for (&t, &c) in counts {
            logits[t as usize] -= p.frequency_penalty as f32 * c as f32;
        }
    }
    if p.greedy() {
        let mut best = 0usize;
        for (i, &l) in logits.iter().enumerate() {
            if l > logits[best] {
                best = i;
            }
        }
        return best as Token;
    }
    let t = p.temperature as f32;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f64)> = logits.iter().enumerate().map(|(i, &l)| (i, (((l - max) / t) as f64).exp())).collect();
    let sum: f64 = probs.iter().map(|x| x.1).sum();
    probs.iter_mut().for_each(|x| x.1 /= sum);
    probs.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut keep = probs.len();
    if p.top_p < 1.0 {
        let mut acc = 0.0;
        for (k, x) in probs.iter().enumerate() {
            acc += x.1;
            if acc >= p.top_p {
                keep = k + 1;
                break;
            }
        }
    }
    let kept = &probs[..keep];
    let total: f64 = kept.iter().map(|x| x.1).sum();
    let mut r = rng.next_f64() * total;
    for x in kept {
        r -= x.1;
        if r <= 0.0 {
            return x.0 as Token;
        }
    }
    kept[keep - 1].0 as Token
}

/// what an engine gives the read loop
pub trait Step {
    fn n_vocab(&self) -> usize;
    /// the logits of the last position fed (after the prompt: its last token)
    fn logits(&mut self, out: &mut [f32]) -> Result<()>;
    /// append one token and evaluate it (async: the wgpu engine awaits the GPU's logits)
    fn feed(&mut self, t: Token) -> impl Future<Output = Result<()>>;
    fn is_eog(&self, t: Token) -> bool;
    /// the token's bytes (llama_token_to_piece, special = false)
    fn piece(&self, t: Token, out: &mut Vec<u8>) -> Result<()>;
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ReadStats {
    pub image_tokens: usize,
    /// every prompt position (text + image)
    pub prompt_tokens: usize,
    /// the image encoder alone — None where the engine cannot separate it: llama.cpp's mtmd encodes the image inside
    /// its prompt evaluation (mtmd_helper_eval_chunk_single), so there `prefill_ms` includes the encode
    pub encode_ms: Option<f64>,
    /// the prompt's prefill up to the first token's logits (with `encode_ms` None: including the image encode)
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub out_tokens: usize,
    /// false = stopped by maxNewTokens (a runaway) or cancelled
    pub stopped_on_eog: bool,
    pub cancelled: bool,
    /// where the next token was chosen: "gpu" (the wgpu engine's greedy presets: penalties + argmax in a kernel, the
    /// decode run ahead of the CPU) or "cpu" (ocr-decode's choose on the read-back logits); None = not reported
    /// (llama.cpp, which always chooses on the CPU)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_choice: Option<&'static str>,
    /// the GPU memory this read used, as the wgpu engine counts it (GpuBytes); None = not counted (llama.cpp)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory: Option<ReadMemory>,
}

/// GPU memory as the wgpu engine counts it: the bytes of every buffer it asked wgpu for, while the buffer lives, by
/// what it holds. Not in it: the driver's rounding of a buffer, its internal staging (queue.write_buffer copies),
/// pipelines / shader code, the browser's own copies — so the process uses more than this. No browser tells a page
/// how much GPU memory is free (no such API; Safari has not even navigator.deviceMemory): this is what the engine
/// needs, not what the device has.
#[derive(Serialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GpuBytes {
    /// the model's weights (resident from the load on)
    pub weights: u64,
    /// the text decoder's KV cache (one read's)
    pub kv_cache: u64,
    /// activations, scratch, per-read tables and uploads, the kernels' uniform params
    pub work: u64,
    /// buffers the CPU maps to read GPU results (logits, chosen tokens)
    pub readback: u64,
}

impl GpuBytes {
    pub fn total(&self) -> u64 {
        self.weights + self.kv_cache + self.work + self.readback
    }
}

/// the highest point of one phase of a read: the bytes alive at that moment
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PhaseMemory {
    /// "vision" | "prefill" | "decode"
    pub phase: &'static str,
    pub peak: GpuBytes,
}

/// a read's GPU memory: resident at its start, its peak (the highest phase), each phase's peak
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ReadMemory {
    pub resident_bytes: u64,
    pub peak_bytes: u64,
    pub peak_phase: &'static str,
    pub phases: Vec<PhaseMemory>,
}

/// where a read is (the engines report each phase as it starts — the UI's live status line); the times of the phases
/// already done ride along. JSON: { "phase": "encodingImage" | "readingPrompt" | "decoding", "encodeMs"?, "prefillMs"? }
#[derive(Serialize, Clone, Copy, Debug, PartialEq)]
#[serde(tag = "phase", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum Phase {
    /// the vision encoder runs (the wgpu engine; llama.cpp's encode is part of ReadingPrompt)
    EncodingImage,
    /// the prompt's prefill (encode_ms: the encoder's time when it ran separately)
    ReadingPrompt { encode_ms: Option<f64> },
    /// the first token is being chosen; text streams from here on
    Decoding { encode_ms: Option<f64>, prefill_ms: f64 },
}

pub struct Outcome {
    pub out_tokens: usize,
    pub stopped_on_eog: bool,
    pub cancelled: bool,
}

/// what the read does with a chosen token: Go on, or stop on end-of-generation / a cancel
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Flow {
    Go,
    Eog,
    Cancelled,
}

/// the read loop's token side, shared by `generate` (the token chosen on the CPU) and an engine that chooses on the
/// GPU (ocr-wgpu's run-ahead decode): stop on end-of-generation, append the token's bytes, stream the complete UTF-8
/// so far (a token may end inside a multi-byte character), count the output tokens; `on_text` returns false to cancel
pub struct Emitter<F: FnMut(&str) -> bool> {
    on_text: F,
    bytes: Vec<u8>,
    sent: usize,
    pub out_tokens: usize,
}

impl<F: FnMut(&str) -> bool> Emitter<F> {
    pub fn new(on_text: F) -> Self {
        Self { on_text, bytes: Vec::new(), sent: 0, out_tokens: 0 }
    }

    /// one chosen token: `eog` = it ends the read (nothing emitted); else `piece` appends its bytes
    pub fn push(&mut self, eog: bool, piece: impl FnOnce(&mut Vec<u8>) -> Result<()>) -> Result<Flow> {
        if eog {
            return Ok(Flow::Eog);
        }
        piece(&mut self.bytes)?;
        self.out_tokens += 1;
        let valid = match std::str::from_utf8(&self.bytes[self.sent..]) {
            Ok(s) => s.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(e) => bail!("the read is not UTF-8 at byte {}", self.sent + e.valid_up_to()),
        };
        if valid > 0 {
            let s = std::str::from_utf8(&self.bytes[self.sent..self.sent + valid]).context("utf8")?;
            self.sent += valid;
            if !(self.on_text)(s) {
                return Ok(Flow::Cancelled);
            }
        }
        Ok(Flow::Go)
    }

    /// the read's end (`last` = the Flow that ended it; Go = the token limit)
    pub fn finish(self, last: Flow) -> Result<Outcome> {
        let cancelled = last == Flow::Cancelled;
        if self.sent < self.bytes.len() && !cancelled {
            bail!("the read ends inside a UTF-8 character");
        }
        Ok(Outcome { out_tokens: self.out_tokens, stopped_on_eog: last == Flow::Eog, cancelled })
    }
}

/// The read after the prompt: `history` = the repetition window's start (the prompt tail). `on_text` gets each new
/// piece of valid UTF-8 and returns false to stop (cancel); `on_logits(step, logits)` sees each step's raw logits.
pub async fn generate(
    m: &mut impl Step,
    p: &Preset,
    seed: u64,
    mut history: Vec<Token>,
    on_text: impl FnMut(&str) -> bool,
    mut on_logits: impl FnMut(usize, &[f32]),
) -> Result<Outcome> {
    let mut counts: HashMap<Token, u32> = HashMap::new();
    let mut rng = Rng(seed);
    let mut logits = vec![0f32; m.n_vocab()];
    let mut em = Emitter::new(on_text);
    let mut flow = Flow::Go;
    while em.out_tokens < p.max_new_tokens {
        m.logits(&mut logits)?;
        on_logits(em.out_tokens, &logits);
        let tok = choose(&mut logits, p, &history, &counts, &mut rng);
        flow = em.push(m.is_eog(tok), |b| m.piece(tok, b))?;
        if flow != Flow::Go {
            break;
        }
        history.push(tok);
        *counts.entry(tok).or_insert(0) += 1;
        m.feed(tok).await?;
    }
    em.finish(flow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preset(rp: f64, fp: f64) -> Preset {
        Preset {
            name: "t".into(),
            temperature: 0.0,
            top_p: 1.0,
            repetition_penalty: rp,
            repetition_window: 2,
            frequency_penalty: fp,
            max_new_tokens: 10,
            gt_token_cap_factor: 2.0,
        }
    }

    #[test]
    fn windowed_rp_hits_distinct_tokens_in_window_only() {
        // token 0 is outside the window of 2, token 1 twice inside counts once
        let mut l = vec![2.0f32, 2.2, -1.0];
        let t = choose(&mut l, &preset(1.1, 0.0), &[0, 1, 1], &HashMap::new(), &mut Rng(1));
        assert_eq!(t, 0);
        assert!((l[1] - 2.0).abs() < 1e-6);
    }

    /// the phase events' JSON as ui/app.js reads it
    #[test]
    fn phase_json_shape() {
        let j = |p: Phase| serde_json::to_string(&p).unwrap();
        assert_eq!(j(Phase::EncodingImage), r#"{"phase":"encodingImage"}"#);
        assert_eq!(j(Phase::ReadingPrompt { encode_ms: None }), r#"{"phase":"readingPrompt","encodeMs":null}"#);
        assert_eq!(j(Phase::Decoding { encode_ms: Some(1.5), prefill_ms: 2.0 }), r#"{"phase":"decoding","encodeMs":1.5,"prefillMs":2.0}"#);
    }

    #[test]
    fn frequency_penalty_counts_output() {
        let mut l = vec![1.0f32, 1.05];
        let counts = HashMap::from([(1, 2u32)]);
        assert_eq!(choose(&mut l, &preset(1.0, 0.1), &[], &counts, &mut Rng(1)), 0);
    }
}
