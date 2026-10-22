// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// ocr-web: OCR Lite in the browser. One wasm module, two users:
//   main thread  the image commands (open → preview, prep ops, auto crop, bake, render, the saved page) on ocr-core —
//                ui/app.js calls them through web/bridge-web.js
//   worker       ocr-wgpu's engine on WebGPU (web/worker.js). The model is read from two JS-backed TensorSources: the
//                JS reader (an OPFS sync access handle, or synchronous HTTP range requests) fills a view on the
//                engine's ONE staging chunk in wasm memory — the model is never held in wasm or JS memory as a whole.
//                `open_engine` makes the device + kernels + tokenizer (kept); each `read` loads the weights from the
//                readers it is given, reads and frees them (ocr-wgpu load_and_read: Safari bills a page's WebGPU memory
//                to it) — between reads no weights are on the GPU.
//                A read streams its text through a JS callback; `cancel` flips a flag the read loop checks per token.
//                A fatal GPU event (device lost, an uncaptured GPU error — ocr-wgpu's set_fatal_hook) goes at once to
//                the worker's global `ocrGpuFatal(kind, message)`, a read running or not; the engine is then dead and
//                every further read fails with that message (gpu.rs Health).
// Values cross as JSON strings in the commands' serde shapes; pixels and file bytes as Uint8Array.
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use image::{DynamicImage, RgbImage};
use js_sys::{Function, Object, Reflect, Uint8Array};
use ocr_core::autocrop::{self, AutoCropOptions};
use ocr_core::imageops::{self, Bake, Op};
use ocr_core::render::Marks;
use ocr_decode::presets::{Preset, Presets, seed_for_stem};
use ocr_wgpu::engine::{Engine, ImageInput};
use ocr_wgpu::gguf::{Gguf, TensorSource};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

thread_local! {
    /// main thread: the opened image + its stem (the sampled read's seed)
    static IMAGE: RefCell<Option<(DynamicImage, String)>> = const { RefCell::new(None) };
    /// worker: the open engine (device + kernels + tokenizer, no weights) — taken out for the length of a read (no
    /// borrow is held across an await)
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
    /// worker: the open engine's device, reachable during a read (debug_destroy_device: the lost-device selftest)
    static DEVICE: RefCell<Option<ocr_wgpu::wgpu::Device>> = const { RefCell::new(None) };
}

/// worker: set by `cancel`, checked by the running read at every token
static CANCEL: AtomicBool = AtomicBool::new(false);

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(|p| error(&format!("ocr-web panicked: {p}"))));
    ocr_wgpu::gpu::set_fatal_hook(gpu_fatal);
}

/// a fatal GPU event → the global scope's `ocrGpuFatal(kind, message)` (web/worker.js posts it to the page); a scope
/// without the hook gets the message on the console
fn gpu_fatal(kind: &'static str, message: &str) {
    let hook = Reflect::get(&js_sys::global(), &JsValue::from_str("ocrGpuFatal")).ok().and_then(|f| f.dyn_into::<Function>().ok());
    match hook {
        Some(f) => {
            if let Err(e) = f.call2(&JsValue::NULL, &JsValue::from_str(kind), &JsValue::from_str(message)) {
                error(&format!("ocrGpuFatal threw on '{message}': {e:?}"));
            }
        }
        None => error(&format!("{message} (no ocrGpuFatal hook in this scope)")),
    }
}

fn js_err(e: anyhow::Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}

/// a JS exception as an anyhow error (its message when it is an Error object)
fn from_js(e: JsValue) -> anyhow::Error {
    match e.dyn_ref::<js_sys::Error>() {
        Some(err) => anyhow!("{}", String::from(err.message())),
        None => anyhow!("{e:?}"),
    }
}

fn set(o: &Object, key: &str, v: impl Into<JsValue>) -> Result<()> {
    Reflect::set(o, &JsValue::from_str(key), &v.into()).map_err(from_js)?;
    Ok(())
}

// ── main thread: the image commands ──────────────────────────────────

fn ops(json: &str) -> Result<Vec<Op>> {
    serde_json::from_str(json).context("prep ops")
}

/// the opened image with the prep chain applied
fn prepared(ops_json: &str) -> Result<(RgbImage, String)> {
    let ops = ops(ops_json)?;
    IMAGE.with(|i| match i.borrow().as_ref() {
        Some((img, stem)) => Ok((imageops::apply(img, &ops)?, stem.clone())),
        None => bail!("no image open"),
    })
}

/// the start: presets, marks, backend (the settings live in the browser's localStorage): { presets, marks, backend }
#[wasm_bindgen]
pub fn startup(user_presets: Option<String>) -> Result<String, JsError> {
    let presets = Presets::load(user_presets.as_deref()).map_err(js_err)?;
    let marks = Marks::profile().map_err(js_err)?;
    Ok(serde_json::json!({ "presets": presets, "marks": marks, "backend": "WebGPU" }).to_string())
}

/// the user's preset list checked against the shipped presets → Presets (the page stores it)
#[wasm_bindgen]
pub fn check_user_presets(user: &str) -> Result<String, JsError> {
    let p = Presets::load(Some(user)).map_err(js_err)?;
    serde_json::to_string(&p).map_err(|e| js_err(e.into()))
}

/// open an image from the file's bytes: { stem, w, h }
#[wasm_bindgen]
pub fn open_image(bytes: &[u8], name: &str) -> Result<String, JsError> {
    let img = imageops::load_bytes(bytes).with_context(|| name.to_owned()).map_err(js_err)?;
    let stem = Path::new(name).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let out = serde_json::json!({ "stem": stem, "w": img.width(), "h": img.height() }).to_string();
    IMAGE.with(|i| *i.borrow_mut() = Some((img, stem)));
    Ok(out)
}

/// the preview (`sharp` = the zoomed-in size): { jpeg: Uint8Array, w, h, bake } — the bridge turns the JPEG into a Blob URL
#[wasm_bindgen]
pub fn preview(ops_json: &str, sharp: bool) -> Result<Object, JsError> {
    (|| -> Result<Object> {
        let (img, _) = prepared(ops_json)?;
        let (w, h) = img.dimensions();
        let o = Object::new();
        set(&o, "jpeg", Uint8Array::from(imageops::preview_jpeg(&img, imageops::preview_longest(sharp))?.as_slice()))?;
        set(&o, "w", w)?;
        set(&o, "h", h)?;
        set(&o, "bake", serde_json::to_string(&imageops::plan(Bake::PIXTRAL, w, h))?)?;
        Ok(o)
    })()
    .map_err(js_err)
}

/// a read's preparation: the baked model input { rgb: Uint8Array, w, h, plan, seed } for the worker
#[wasm_bindgen]
pub fn bake(ops_json: &str) -> Result<Object, JsError> {
    (|| -> Result<Object> {
        let (img, stem) = prepared(ops_json)?;
        let (baked, plan) = imageops::bake(&img, Bake::PIXTRAL);
        let o = Object::new();
        set(&o, "w", baked.width())?;
        set(&o, "h", baked.height())?;
        set(&o, "rgb", Uint8Array::from(baked.as_raw().as_slice()))?;
        set(&o, "plan", serde_json::to_string(&plan)?)?;
        set(&o, "seed", seed_for_stem(&stem) as f64)?;
        Ok(o)
    })()
    .map_err(js_err)
}

/// the auto crop: CropRect
#[wasm_bindgen]
pub fn auto_crop(ops_json: &str, opts_json: &str) -> Result<String, JsError> {
    (|| -> Result<String> {
        let (img, _) = prepared(ops_json)?;
        let opts: AutoCropOptions = serde_json::from_str(opts_json).context("auto crop options")?;
        Ok(serde_json::to_string(&autocrop::auto_crop(&img, &opts)?)?)
    })()
    .map_err(js_err)
}

/// the markup → HTML
#[wasm_bindgen]
pub fn render(text: &str) -> Result<String, JsError> {
    Ok(Marks::profile().map_err(js_err)?.html(text))
}

/// the saved .html page (the bridge downloads it)
#[wasm_bindgen]
pub fn document(title: &str, text: &str) -> Result<String, JsError> {
    Ok(Marks::profile().map_err(js_err)?.document(title, text))
}

/// the baked model input as PNG bytes
#[wasm_bindgen]
pub fn baked_png(ops_json: &str) -> Result<Vec<u8>, JsError> {
    (|| -> Result<Vec<u8>> {
        let (img, _) = prepared(ops_json)?;
        imageops::png(&imageops::bake(&img, Bake::PIXTRAL).0)
    })()
    .map_err(js_err)
}

// ── worker: the engine ───────────────────────────────────────────────

/// a model file read through JS: `read(offset, view)` fills the Uint8Array view (on wasm memory) with the bytes at
/// `offset` and returns how many it wrote
struct JsSource {
    name: String,
    len: u64,
    read: Function,
}

impl TensorSource for JsSource {
    fn len(&self) -> Result<u64> {
        Ok(self.len)
    }

    fn read_at(&self, off: u64, buf: &mut [u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        // SAFETY: the view is used only during this call, and the JS reader allocates nothing in wasm memory (a grown
        // memory would detach the view, not corrupt it)
        let view = unsafe { Uint8Array::view_mut_raw(buf.as_mut_ptr(), buf.len()) };
        let n = self.read.call2(&JsValue::NULL, &JsValue::from_f64(off as f64), &view).map_err(from_js)?;
        let n = n.as_f64().with_context(|| format!("{}: the reader returned no byte count", self.name))?;
        if n as usize != buf.len() {
            bail!("{}: {n} of {} bytes read at {off}", self.name, buf.len());
        }
        Ok(())
    }
}

fn source(name: &str, len: f64, read: Function) -> Result<Gguf<JsSource>> {
    Gguf::from_source(JsSource { name: name.to_owned(), len: len as u64, read }).with_context(|| format!("{name} is not a readable GGUF"))
}

/// open (or replace) the engine for the two model files — the device + kernels, the pre-check against its limits, the
/// tokenizer; no weights (each read loads them): { adapter, backend, compute, computeLabel, weights, subgroups,
/// shaderChecks, maxStorageBufferBindingSize, maxBufferSize, modelBytes, idleBytes (GPU memory alive between reads) }
#[wasm_bindgen]
pub async fn open_engine(text_len: f64, text_read: Function, mmproj_len: f64, mmproj_read: Function) -> Result<String, JsError> {
    async {
        close_engine(); // the old engine's GPU buffers first
        let text = source(ocr_wgpu::engine::TEXT_GGUF, text_len, text_read)?;
        let mmproj = source(ocr_wgpu::engine::MMPROJ_GGUF, mmproj_len, mmproj_read)?;
        let bytes = text.data_bytes() + mmproj.data_bytes();
        let eng = Engine::from_sources(&text, &mmproj).await?;
        let info = serde_json::json!({
            "adapter": eng.gpu.info.name,
            "backend": format!("{:?}", eng.gpu.info.backend),
            "compute": format!("{:?}", eng.gpu.compute),
            "computeLabel": eng.gpu.compute_label(),
            "weights": eng.weights,
            "subgroups": eng.gpu.subgroups,
            "shaderChecks": ocr_wgpu::gpu::SHADER_CHECKS,
            "maxStorageBufferBindingSize": eng.gpu.limits.max_storage_buffer_binding_size,
            "maxBufferSize": eng.gpu.limits.max_buffer_size,
            "modelBytes": bytes,
            "idleBytes": eng.gpu.mem_live().total(),
        });
        DEVICE.with(|d| *d.borrow_mut() = Some(eng.gpu.device.clone()));
        let mut eng = eng;
        eng.stage_pause = Some(Box::new(|| Box::pin(sleep_ms(STAGE_PAUSE_MS))));
        ENGINE.with(|e| *e.borrow_mut() = Some(eng));
        Ok(info.to_string())
    }
    .await
    .map_err(js_err)
}

#[wasm_bindgen]
pub fn engine_open() -> bool {
    ENGINE.with(|e| e.borrow().is_some())
}

/// PROBE: the pause after the staged load's vision half (the vision weights freed, the GPU idle), before the text
/// weights load — iOS killed the page at that load (iPhone: the first read); if the pause lets it through, iOS still
/// counted the freed memory for a while
const STAGE_PAUSE_MS: i32 = 3000;

/// a timer of the worker (setTimeout) as a future
async fn sleep_ms(ms: i32) -> Result<()> {
    let set_timeout: Function = Reflect::get(&js_sys::global(), &JsValue::from_str("setTimeout"))
        .map_err(from_js)?
        .dyn_into()
        .map_err(|_| anyhow!("setTimeout is no function"))?;
    let p = js_sys::Promise::new(&mut |resolve, reject| {
        if let Err(e) = set_timeout.call2(&JsValue::NULL, &resolve, &JsValue::from(ms)) {
            let _ = reject.call1(&JsValue::NULL, &e);
        }
    });
    wasm_bindgen_futures::JsFuture::from(p).await.map_err(from_js)?;
    Ok(())
}

/// close the engine (its device and GPU buffers go with it)
#[wasm_bindgen]
pub fn close_engine() {
    ENGINE.with(|e| e.borrow_mut().take());
    DEVICE.with(|d| d.borrow_mut().take());
}

/// the lost-device selftest (web/selftest.html?lose=1): destroys the loaded engine's GPU device — as a crashed GPU
/// process loses it — also while a read runs
#[wasm_bindgen]
pub fn debug_destroy_device() -> Result<(), JsError> {
    DEVICE.with(|d| match d.borrow().as_ref() {
        Some(dev) => {
            dev.destroy();
            Ok(())
        }
        None => Err(JsError::new("no engine open: no GPU device to destroy")),
    })
}

/// a Stop: the running (or the next queued) read ends at its next token
#[wasm_bindgen]
pub fn cancel() {
    CANCEL.store(true, Ordering::SeqCst);
}

/// called when a read is queued: a Stop during the model load still counts
#[wasm_bindgen]
pub fn reset_cancel() {
    CANCEL.store(false, Ordering::SeqCst);
}

/// the model files of one read: each a reader `(offset, view) → bytes written` over a handle the worker opened for
/// this read (OPFS) or a URL (HTTP ranges)
#[wasm_bindgen]
pub struct ReadFiles {
    text_len: f64,
    text_read: Function,
    mmproj_len: f64,
    mmproj_read: Function,
}

#[wasm_bindgen]
impl ReadFiles {
    #[wasm_bindgen(constructor)]
    pub fn new(text_len: f64, text_read: Function, mmproj_len: f64, mmproj_read: Function) -> ReadFiles {
        ReadFiles { text_len, text_read, mmproj_len, mmproj_read }
    }
}

/// read one baked image (RGB, w × h) with the weights loaded from `files` for this read and freed after it (a failed
/// or cancelled read too): each phase as it starts goes to `on_phase(json)` (ocr-decode Phase + loadMs), each new piece
/// of text to `on_text(piece)`; resolves to the ReadStats JSON + loadMs (the files' headers + the weight load) + load
/// { parseMs, ms, readMs, waitMs, weightsBytes, visionWeightsBytes, textWeightsBytes, residentBytes, loadPeakBytes } +
/// idleBytes (the GPU memory alive after the read)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn read(files: ReadFiles, rgb: Vec<u8>, w: u32, h: u32, preset_json: String, seed: f64, on_phase: Function, on_text: Function) -> Result<String, JsError> {
    let Some(mut eng) = ENGINE.with(|e| e.borrow_mut().take()) else {
        return Err(JsError::new("no engine open (or a read is already running)"));
    };
    let r = run_read(&mut eng, files, rgb, w, h, &preset_json, seed, &on_phase, &on_text).await;
    ENGINE.with(|e| *e.borrow_mut() = Some(eng));
    r.map_err(js_err)
}

#[allow(clippy::too_many_arguments)]
async fn run_read(eng: &mut Engine, files: ReadFiles, rgb: Vec<u8>, w: u32, h: u32, preset_json: &str, seed: f64, on_phase: &Function, on_text: &Function) -> Result<String> {
    let img = RgbImage::from_raw(w, h, rgb).with_context(|| format!("{w}×{h} RGB: wrong byte count"))?;
    let p: Preset = serde_json::from_str(preset_json).context("preset")?;
    p.validate()?;
    let t0 = web_time::Instant::now();
    let text = source(ocr_wgpu::engine::TEXT_GGUF, files.text_len, files.text_read)?;
    let mmproj = source(ocr_wgpu::engine::MMPROJ_GGUF, files.mmproj_len, files.mmproj_read)?;
    let parse_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut cb_err: Option<anyhow::Error> = None;
    let mut phase_err: Option<anyhow::Error> = None;
    // the load's time rides on every phase (the status line: "load 0.9 s · image … · prefill …"): the first phase
    // starts right after the load
    let mut load_ms: Option<f64> = None;
    let r = eng
        .load_and_read(
            &text,
            &mmproj,
            ImageInput::Pixels(&img),
            &p,
            seed as u64,
            |ph| {
                let ms = *load_ms.get_or_insert_with(|| t0.elapsed().as_secs_f64() * 1e3);
                let r = phase_json(&ph, ms).and_then(|j| on_phase.call1(&JsValue::NULL, &JsValue::from_str(&j)).map_err(from_js));
                if let Err(e) = r {
                    phase_err.get_or_insert(e.context("the phase callback failed"));
                }
            },
            |s| match on_text.call1(&JsValue::NULL, &JsValue::from_str(s)) {
                Ok(_) => !CANCEL.load(Ordering::SeqCst),
                Err(e) => {
                    cb_err = Some(from_js(e).context("the text callback threw"));
                    false
                }
            },
            None,
        )
        .await?;
    if let Some(e) = cb_err.or(phase_err) {
        return Err(e);
    }
    let m = &r.load.memory;
    let mut out = serde_json::to_value(&r.stats)?;
    let o = out.as_object_mut().context("ReadStats is no JSON object")?;
    o.insert("loadMs".into(), (parse_ms + r.load.ms).into());
    o.insert(
        "load".into(),
        serde_json::json!({
            "parseMs": parse_ms,
            "ms": r.load.ms,
            "readMs": r.load.read_ms,
            "waitMs": r.load.wait_ms,
            "weightsBytes": m.vision_weights + m.text_weights,
            "visionWeightsBytes": m.vision_weights,
            "textWeightsBytes": m.text_weights,
            "residentBytes": m.resident,
            "loadPeakBytes": m.load_peak,
        }),
    );
    o.insert("idleBytes".into(), r.idle_bytes.into());
    Ok(out.to_string())
}

/// a Phase's JSON + loadMs
fn phase_json(ph: &ocr_decode::sampling::Phase, load_ms: f64) -> Result<String> {
    let mut v = serde_json::to_value(ph)?;
    v.as_object_mut().context("Phase is no JSON object")?.insert("loadMs".into(), load_ms.into());
    Ok(v.to_string())
}

/// the wasm linear memory's size in bytes (the evidence that the model is not in it)
#[wasm_bindgen]
pub fn wasm_memory_bytes() -> Result<f64, JsError> {
    let mem = wasm_bindgen::memory().dyn_into::<js_sys::WebAssembly::Memory>().map_err(|_| JsError::new("wasm memory is no WebAssembly.Memory"))?;
    let buf = mem.buffer().dyn_into::<js_sys::ArrayBuffer>().map_err(|_| JsError::new("wasm memory buffer is no ArrayBuffer"))?;
    Ok(buf.byte_length() as f64)
}
