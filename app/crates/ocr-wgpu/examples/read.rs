// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// Read one baked image on wgpu alone (vision encoder → text decoder → ocr-decode's read loop) and, with --ref, compare
// with a reference read dumped from llama.cpp (<prefix>.txt + <prefix>.logits0.bin; the dump tool is not part of this
// repository):
//   logits  the first step's raw logits: cosine, max abs diff, argmax, top-10 overlap
//   read    identical? else the first differing character + both continuations, the CER between the two reads,
//           and the first differing token step with both engines' top-2 margins there
// --ref-embd <ref.bin> feeds llama.cpp's mtmd embeddings (f32, dumped from llama.cpp) instead of this engine's
// vision encoder: the text decoder's parity on its own.
// Writes <out>.txt (the read) and, with the logits tapped, <out>.logits0.bin when --out is given. The logits are tapped
// (every step's logits read back, so the token is chosen on the CPU) with --ref or --logits; without them a greedy
// preset chooses on the GPU (the app's path) — the timing line says which ("token choice"). An image off the 28 grid (or over 1540 px) is
// baked first, exactly as the app does (ocr-core imageops). Each read loads the weights and frees them after it (the
// app's path, engine.rs load_and_read): "open" = the device + kernels + tokenizer, "load" = the read's weight load.
// --repeat N reads it N times on the same engine and prints each read's timing (the first read vs the warm ones), its
// weight load, whether its text equals the first read's and the GPU memory still alive after it (the weights and the
// KV cache must go with the read: the parameter ring only); the comparisons use the last read. --cancel-after N stops each
// read at its N-th piece of text (the Stop button's path: the text callback returns false).
// The model folder (text-q8_0.gguf + mmproj-bf16.gguf): --model, else the environment's OCR_LITE_MODEL_DIR.
//   cargo run --release -p ocr-wgpu --example read -- <image> [--model <dir>] [--preset NAME] [--max-tokens N]
//       [--ref <read_ref prefix>] [--ref-embd <ref.bin>] [--out <prefix>] [--repeat N] [--logits] [--cancel-after N]
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ocr_decode::presets::{Presets, seed_for_stem};
use ocr_wgpu::engine::{Engine, ImageInput, ModelFiles};

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("cannot read {}", p.display()))?;
    if b.len() % 4 != 0 {
        bail!("{} is not f32 data", p.display());
    }
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn top_k(l: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..l.len()).collect();
    idx.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
    idx.truncate(k);
    idx
}

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + (ca != cb) as usize).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn snippet(c: &[char], from: usize) -> String {
    c[from.min(c.len())..(from + 40).min(c.len())].iter().collect::<String>().replace('\n', "⏎")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        bail!("usage: read <image.png> [--model <dir>] [--preset NAME] [--max-tokens N] [--ref <prefix>] [--ref-embd <ref.bin>] [--out <prefix>] [--repeat N] [--logits] [--cancel-after N]");
    }
    let img_path = PathBuf::from(&args[1]);
    let mut model = std::env::var_os("OCR_LITE_MODEL_DIR").map(PathBuf::from);
    let presets = Presets::load(None)?;
    let mut name = presets.default_name.clone();
    let (mut max_tokens, mut reference, mut ref_embd, mut out) = (None::<usize>, None::<PathBuf>, None::<PathBuf>, None::<PathBuf>);
    let mut repeat = 1usize;
    let mut cancel_after = None::<usize>;
    let mut tap_logits = false;
    let mut then = None::<PathBuf>;
    let mut i = 2;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--logits" {
            tap_logits = true;
            i += 1;
            continue;
        }
        let val = args.get(i + 1).cloned().with_context(|| format!("{flag} needs a value"))?;
        match flag {
            "--model" => model = Some(PathBuf::from(val)),
            "--preset" => name = val,
            "--max-tokens" => max_tokens = Some(val.parse()?),
            "--ref" => reference = Some(PathBuf::from(val)),
            "--ref-embd" => ref_embd = Some(PathBuf::from(val)),
            "--out" => out = Some(PathBuf::from(val)),
            "--repeat" => repeat = val.parse::<usize>()?.max(1),
            "--cancel-after" => cancel_after = Some(val.parse()?),
            "--then" => then = Some(PathBuf::from(val)),
            a => bail!("unknown argument {a}"),
        }
        i += 2;
    }
    let mut p = presets.named(&name)?.clone();
    if let Some(n) = max_tokens {
        p.max_new_tokens = n;
    }
    let tap_logits = tap_logits || reference.is_some();
    let img = ocr_core::imageops::load(&img_path)?.to_rgb8();
    let img = if img.width() % 28 != 0 || img.height() % 28 != 0 || img.width().max(img.height()) > 1540 {
        let (baked, _) = ocr_core::imageops::bake(&img, ocr_core::imageops::Bake::PIXTRAL);
        eprintln!("baked {}x{} → {}x{}", img.width(), img.height(), baked.width(), baked.height());
        baked
    } else {
        img
    };
    let stem = img_path.file_stem().context("image name")?.to_string_lossy().into_owned();

    let t = Instant::now();
    let model = model.context("no model folder: --model <dir> or OCR_LITE_MODEL_DIR")?;
    let files = ModelFiles::open(&model)?;
    let mut engine = Engine::open(&files)?;
    let open_ms = t.elapsed().as_secs_f64() * 1e3;
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    let embd = match &ref_embd {
        Some(p) => Some(read_f32(p)?),
        None => None,
    };
    let mut run = 0usize;
    let mut text1: Option<String> = None;
    let (text, first, top2, read) = loop {
        run += 1;
        let input = match &embd {
            Some(e) => ImageInput::Embeddings(e),
            None => ImageInput::Pixels(&img),
        };
        let mut text = String::new();
        let mut pieces = 0usize;
        let mut first: Vec<f32> = Vec::new();
        let mut top2: Vec<[f32; 4]> = Vec::new();
        let t_read = Instant::now();
        let mut tap = |step: usize, l: &[f32]| {
            if step == 0 {
                first = l.to_vec();
            }
            // the per-step top-2 is for the comparison with a reference only (a full sort per token would slow the read)
            if reference.is_some() {
                let t = top_k(l, 2);
                top2.push([t[0] as f32, l[t[0]], t[1] as f32, l[t[1]]]);
            }
        };
        let read = engine.read_inspect(
            &files,
            input,
            &p,
            seed_for_stem(&stem),
            |_| {},
            |s| {
                text.push_str(s);
                pieces += 1;
                cancel_after.is_none_or(|n| pieces < n)
            },
            if tap_logits { Some(&mut tap) } else { None },
        )?;
        let stats = &read.stats;
        if repeat > 1 {
            let same = text1.get_or_insert_with(|| text.clone()) == &text;
            println!(
                "read {run}/{repeat}: load {:.0} ms · encode {:.0} ms · prefill {:.0} ms · {} tokens in {:.0} ms ({:.1} tok/s, token choice {}) · eog {} · cancelled {} · {} chars · wall {:.0} ms · same text as read 1: {same} · alive after the read {:.1} MiB",
                read.load.ms,
                stats.encode_ms.unwrap_or(f64::NAN),
                stats.prefill_ms,
                stats.out_tokens,
                stats.decode_ms,
                stats.out_tokens as f64 / (stats.decode_ms / 1e3),
                stats.token_choice.unwrap_or("?"),
                stats.stopped_on_eog,
                stats.cancelled,
                text.chars().count(),
                t_read.elapsed().as_secs_f64() * 1e3,
                mib(read.idle_bytes)
            );
        }
        if run >= repeat {
            break (text, first, top2, read);
        }
    };
    let stats = &read.stats;
    if let Some(r) = engine.gpu.profile_report() {
        eprint!("{r}");
    }
    println!(
        "{} {}x{} on {} ({:?}, compute {:?} = {}, weights {}, subgroups {}, shader checks {}), {}{}: {} image + {} prompt tokens · open {open_ms:.0} ms · load {:.0} ms (files {:.0} · gpu copy {:.0}) · encode {:.0} ms · prefill {:.0} ms · {} tokens in {:.0} ms ({:.1} tok/s, token choice {}) · eog {}",
        stem,
        img.width(),
        img.height(),
        engine.gpu.info.name,
        engine.gpu.info.backend,
        engine.gpu.compute,
        engine.gpu.compute_label(),
        engine.weights,
        if engine.gpu.subgroups { "on" } else { "off" },
        if ocr_wgpu::gpu::SHADER_CHECKS { "on" } else { "off" },
        p.name,
        if ref_embd.is_some() { ", llama.cpp embeddings" } else { "" },
        stats.image_tokens,
        stats.prompt_tokens,
        read.load.ms,
        read.load.read_ms,
        read.load.wait_ms,
        stats.encode_ms.unwrap_or(f64::NAN),
        stats.prefill_ms,
        stats.out_tokens,
        stats.decode_ms,
        stats.out_tokens as f64 / (stats.decode_ms / 1e3),
        stats.token_choice.unwrap_or("?"),
        stats.stopped_on_eog
    );
    // the GPU memory the engine counted (gpu.rs Buf): weights, resident, the load's and this read's peaks, per phase,
    // what is still alive after the read (the weights and the KV cache go with the read)
    let m = &read.load.memory;
    let rm = stats.gpu_memory.as_ref().context("the wgpu read reported no GPU memory")?;
    let phases: Vec<String> = rm
        .phases
        .iter()
        .map(|p| format!("{} {:.1} (weights {:.1} · kv {:.1} · work {:.1} · readback {:.1})", p.phase, mib(p.peak.total()), mib(p.peak.weights), mib(p.peak.kv_cache), mib(p.peak.work), mib(p.peak.readback)))
        .collect();
    println!(
        "gpu memory MiB: weights {:.1} (vision {:.1} · text {:.1}) · resident {:.1} · load peak {:.1} · read peak {:.1} in {} · after read {:.1} · phases: {}",
        mib(m.vision_weights + m.text_weights),
        mib(m.vision_weights),
        mib(m.text_weights),
        mib(m.resident),
        mib(m.load_peak),
        mib(rm.peak_bytes),
        rm.peak_phase,
        mib(read.idle_bytes),
        phases.join(" | ")
    );
    println!("memory pool MiB (held by the GPU for the engine, slices in use or not): {:.1}", mib(engine.gpu.pool_bytes()));
    // --then <image>: one more read on the same engine (the pool as the first left it) — a small page, then a large one
    if let Some(next) = &then {
        let img2 = ocr_core::imageops::load(next)?.to_rgb8();
        let (img2, _) = ocr_core::imageops::bake(&img2, ocr_core::imageops::Bake::PIXTRAL);
        let stem2 = next.file_stem().context("image name")?.to_string_lossy().into_owned();
        let r2 = engine.read_inspect(&files, ImageInput::Pixels(&img2), &p, seed_for_stem(&stem2), |_| {}, |_| true, None)?;
        let g = r2.stats.gpu_memory.as_ref().context("the wgpu read reported no GPU memory")?;
        println!("then {stem2} {}x{}: {} image tokens · {} tokens · read peak {:.1} MiB in {} · memory pool MiB: {:.1}", img2.width(), img2.height(), r2.stats.image_tokens, r2.stats.out_tokens, mib(g.peak_bytes), g.peak_phase, mib(engine.gpu.pool_bytes()));
    }
    if let Some(o) = &out {
        std::fs::write(o.with_extension("txt"), &text)?;
        if tap_logits {
            std::fs::write(o.with_extension("logits0.bin"), first.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        }
    }
    let Some(r) = reference else {
        println!("{text}");
        return Ok(());
    };

    // logits of the first step
    let rl = read_f32(&r.with_extension("logits0.bin"))?;
    if rl.len() != first.len() {
        bail!("reference logits have {} values, wgpu {}", rl.len(), first.len());
    }
    let (mut ab, mut aa, mut bb, mut max_abs) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in first.iter().zip(&rl) {
        let (x, y) = (x as f64, y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
        max_abs = max_abs.max((x - y).abs());
    }
    let (tw, tr) = (top_k(&first, 10), top_k(&rl, 10));
    let overlap = tw.iter().filter(|t| tr.contains(t)).count();
    println!(
        "logits step 0: cosine {:.7} (1 − cos {:.1e}) · max abs diff {:.4} (max|ref| {:.2}) · argmax wgpu {} llama.cpp {} · top-10 overlap {overlap}/10 · top-2 margin wgpu {:.3} llama.cpp {:.3}",
        ab / (aa.sqrt() * bb.sqrt()),
        1.0 - ab / (aa.sqrt() * bb.sqrt()),
        max_abs,
        rl.iter().fold(0f32, |m, v| m.max(v.abs())),
        tw[0],
        tr[0],
        first[tw[0]] - first[tw[1]],
        rl[tr[0]] - rl[tr[1]]
    );

    // the reads
    let rtext = std::fs::read_to_string(r.with_extension("txt"))?;
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(r.with_extension("json"))?)?;
    let (a, b): (Vec<char>, Vec<char>) = (text.chars().collect(), rtext.chars().collect());
    let cer = levenshtein(&a, &b) as f64 / b.len().max(1) as f64;
    let ref_stats = &meta["stats"];
    println!(
        "llama.cpp: {} tokens, prefill {:.0} ms, decode {:.0} ms ({:.1} tok/s), eog {}",
        ref_stats["outTokens"],
        ref_stats["prefillMs"].as_f64().unwrap_or(f64::NAN),
        ref_stats["decodeMs"].as_f64().unwrap_or(f64::NAN),
        ref_stats["outTokens"].as_f64().unwrap_or(f64::NAN) / (ref_stats["decodeMs"].as_f64().unwrap_or(f64::NAN) / 1e3),
        ref_stats["stoppedOnEog"]
    );
    if a == b {
        println!("read: IDENTICAL ({} chars, {} tokens)", a.len(), stats.out_tokens);
    } else {
        let pos = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
        println!("read: DIFFERENT at char {pos} of {} / {} · CER(wgpu vs llama.cpp) {:.4}", a.len(), b.len(), cer);
        println!("  wgpu      …{}", snippet(&a, pos.saturating_sub(10)));
        println!("  llama.cpp …{}", snippet(&b, pos.saturating_sub(10)));
        let rtop: Vec<[f64; 4]> = meta["top2"]
            .as_array()
            .context("top2")?
            .iter()
            .map(|v| {
                let x: Vec<f64> = v.as_array().map(|a| a.iter().filter_map(|y| y.as_f64()).collect()).unwrap_or_default();
                [x[0], x[1], x[2], x[3]]
            })
            .collect();
        if let Some(step) = top2.iter().zip(&rtop).position(|(w, l)| w[0] as f64 != l[0]) {
            let (w, l) = (top2[step], rtop[step]);
            println!(
                "  first raw-argmax difference at step {step}: wgpu {} ({:.3}) vs {} ({:.3}), margin {:.3} · llama.cpp {} ({:.3}) vs {} ({:.3}), margin {:.3}",
                w[0], w[1], w[2], w[3], w[1] - w[3], l[0], l[1], l[2], l[3], l[1] - l[3]
            );
        } else {
            println!("  raw argmax equal on all {} common steps (the difference comes from the penalties)", top2.len().min(rtop.len()));
        }
    }
    Ok(())
}
