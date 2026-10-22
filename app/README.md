# OCR Lite

A small web app that reads a page image with **LightOnOCR** entirely on the device: one image in → crop / prepare →
OCR on the GPU (WebGPU) → the read as rendered, editable markup → save. The engine is a purpose-built Rust + wgpu port
of the model (no llama.cpp, no Python), compiled to WebAssembly; the model is downloaded once from Hugging Face into
the browser's storage (OPFS). Images never leave the machine.

Live: <https://b-straub.github.io/historic-htr-ocr/app/> · the output format: [site/markup.html](../site/markup.html).

## Layout

| Path | What |
|---|---|
| `crates/ocr-decode` | the decoding presets (compiled in from `presets.json`) + the decoding step and read loop — pure Rust |
| `crates/ocr-core` | image ops (rotate, contrast, pencil, crop), the bake (≤ 1540 px, white pad to the 28 px grid), the auto crop, the markup → HTML renderer (marks from `markup.json`) |
| `crates/ocr-gguf` | the GGUF reader: header + tensor table from a prefix of the file, tensor bytes by range — the model is never held in memory as a whole |
| `crates/ocr-wgpu` | the engine: vision encoder + text decoder as hand-written WGSL kernels (f32, f16, f16 matrix units), GPU token choice, GPU memory ledger; examples `read` (a native read), `gguf_info`, `adapter`, `cut` |
| `crates/ocr-web` | the browser build: ocr-core + ocr-wgpu behind wasm-bindgen (wasm32 only) |
| `ui/` | the page: `index.html`, `app.js` (the prep chain, the state table of every control), `editor.js` (WYSIWYG markup editing), `setup.js` (first-run setup), `scan.js` (the read's scan overlay), CSS |
| `web/` | the host side: `bridge-web.js` (commands, events, model storage, crash marker), `worker.js` (the engine worker: OPFS / HTTP range reads into the engine's staging buffer), `gpu-check.js` (the device check), `config.js` (**the model**), `sw.js` + `pwa.js` (service worker, "new version" banner), `icons/`, `build.sh`, `dist.py` / `dist.sh`, `serve.py`, `selftest.html` |
| `tests/` | Node unit tests (`editor_roundtrip.js`, `gpu_check.mjs`) and the browser tests (`browser/*.cjs`, headless Chrome) |
| `presets.json` | THE decoding presets — the app's, and the same an evaluation or a server-side read uses |
| `markup.json` | the inline marks of the model's output |
| `ci.sh` | every build and test step — CI and local runs use it alike |

## Build

Needs Rust stable with the `wasm32-unknown-unknown` target, `wasm-bindgen-cli` of exactly the version in `Cargo.lock`,
Python 3 and Node (22).

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version '=0.2.129' --locked
app/web/dist.sh            # wasm build (web/build.sh → web/pkg/) + the static site → app/dist/
app/ci.sh site             # the whole Pages site → _site/ (site/ at /, app/dist/ at /app/, the generated examples)
```

The wasm build remaps `~/.cargo`, the toolchain and the workspace (`--remap-path-prefix`), so no build-machine path
ends up in the binary. Every built JS / CSS file carries the licence line.

## Run locally

```sh
app/web/serve.py --dist _site --prefix /historic-htr-ocr/ --model-dir <folder>
# open http://127.0.0.1:8765/historic-htr-ocr/app/?modelBase=/model/
```

`serve.py --dist` serves a built site as GitHub Pages does (no extra headers, under the Pages sub-path); `--model-dir`
serves a local model pair (`text-q8_0.gguf` + `mmproj-bf16.gguf`) at `/model/`. Two test overrides are honoured **on
localhost only** (anywhere else they are ignored, and the console says so):
- `?modelBase=/model/` — the setup downloads the model from that folder instead of Hugging Face;
- `?gpuCheck=skip` — the setup's device check passes without running (the browser tests on machines without a GPU).

Without `--dist`, `serve.py` serves the sources themselves (`web/` first, then `ui/`) — the dev loop after
`web/build.sh`; `selftest.html?image=/data/x.png&expect=/data/x.txt[&source=opfs&fill=1][&preset=NAME][&repeat=N]`
(with `--data-dir`) reads one image and compares it with a native read (`cargo run --release -p ocr-wgpu --example
read -- x.png --model <folder> --out x`).

## The model

`web/config.js` is the one place that says where the model comes from: `MODEL` (name, Hugging Face repo, the repo's
file names) and `MODEL_SOURCES` (only `"hub"` is on; uploading files, another folder URL and `?model=` streaming are
switched off). It points at
[b-straub/LightOnOCR-3-1B-Historic-German-Handwriting](https://huggingface.co/b-straub/LightOnOCR-3-1B-Historic-German-Handwriting),
whose `text-q8_0.gguf` + `mmproj-bf16.gguf` the setup downloads. Another GGUF repo of the same architecture works by
changing `name` and `repo`, plus `remoteFiles` when its file names differ:

```js
name: "LightOnOCR-3-1B base model",
repo: "https://huggingface.co/prithivMLmods/LightOnOCR-3-1B-GGUF",
remoteFiles: { "text-q8_0.gguf": "LightOnOCR-3-1B.Q8_0.gguf", "mmproj-bf16.gguf": "LightOnOCR-3-1B.mmproj-bf16.gguf" },
```

A model stored in the browser is used only when its record (`model-source.json` in OPFS) says it came from an allowed
source; otherwise the setup downloads the model again.

## Test

```sh
app/ci.sh                  # all: test, build, site, browser
app/ci.sh test             # cargo test + the Node unit tests
app/ci.sh browser          # the browser tests on _site/ (after build + site)
```

- **cargo**: `ocr-decode`, `ocr-core`, `ocr-gguf` always. `ocr-wgpu`'s tests run kernels on the GPU: `ci.sh` asks the
  `adapter` example first — with an adapter they all run; without one (CI runners) they are compiled, only the CPU
  tests run, and the log says `ocr-wgpu GPU tests SKIPPED`. The tokenizer tests need the model: they are `#[ignore]`d
  and run with `OCR_LITE_MODEL_DIR=<folder>`.
- **Node**: the markup editor's round trip (jsdom) and the device check's pure part.
- **Browser** (`tests/browser/`, headless Chrome via puppeteer-core; `npm ci` in `app/tests/`, Chrome from `$CHROME`
  or the usual install): `ui` (zoom, crop mapping, pencil, About, layout), `setup` (the first-run setup with a stand-in
  model), `state` (every control's state), `pages` (the published site: pages, links, examples, CSP, service worker,
  manifest, a real Hugging Face download started and cancelled). The parts that need a read — the Read / Stop states,
  a read compared with the native read — run with `OCR_LITE_MODEL_DIR` (+ `OCR_LITE_READ_IMAGE`,
  `OCR_LITE_READ_EXPECT` for `pages`) and are reported as `SKIP` without it. Screenshots go to `test-output/`.

## CI and publishing

- **Every push and pull request = the full CI** (`.github/workflows/ci.yml`): `app/ci.sh test`, `build`, `site`,
  `browser` on ubuntu-24.04 (Rust stable + wasm32, wasm-bindgen-cli cached, Python, Node).
- **A tag `v*.*.*` = the full CI + publish** (`.github/workflows/pages.yml`): its first job IS `ci.yml` (called as a
  reusable workflow); the build job `needs` it and takes the very `_site` the CI built and tested; the deploy job
  `needs` the build. A tag whose tests fail never deploys; the manual run (workflow_dispatch) goes through the same gate.
  Pages source: Settings → Pages → Source "GitHub Actions".
- **Hosting elsewhere**: the dist is a plain static site; it needs HTTPS (WebGPU, OPFS, the service worker). For an
  Apache host it carries `.htaccess` (from `web/htaccess`, needs `AllowOverride FileInfo`): `.wasm` as
  `application/wasm`, `no-cache` for `index.html` / `sw.js` / the manifest, a year `immutable` for the hashed files,
  `nosniff`, `frame-ancestors 'none'`, compression. GitHub Pages ignores it. No COOP / COEP needed.
- **The version is the tag.** The About box shows the tag the build was made from (`v1.0.0`), the commit and the
  build time; a build without a tag on its commit says "untagged". No version number to keep in step elsewhere.

## Browsers and devices

Needs WebGPU in workers and OPFS sync access handles: Chrome / Edge 113+, Safari 26+. Run on Apple silicon Macs
(Chrome, Safari) and an iPad Pro (Safari); other GPUs are untested. The engine picks its compute path per device:
**F16Coop** (native Metal: f16 inputs on the GPU's matrix units, f32 sums), **F16** (shader-f16, the browsers),
**F32** (no shader-f16, or too little workgroup memory for the f16 kernels). Residual streams, norms, accumulators and
the softmax stay f32 on every path.

Measured on an Apple M3 (10-core GPU), a full 980 × 1540 page (1925 image tokens): Chrome image encode ≈ 8.3 s,
prefill ≈ 1.8 s, decode ≈ 84 tokens/s; native wgpu (F16Coop) ≈ 5.9 s · 1.6 s · 86 tokens/s.

- **GPU memory**: no browser tells a page how much GPU memory is free, so the engine counts every buffer it creates
  (`ocr-wgpu` `Buf`, Settings shows it). Each read loads the weights in two stages — the vision weights (790 MiB),
  the image encode, the vision weights freed, then the text weights (604 MiB) — so the two sets are never on the GPU
  together: a full page peaks at about 1.2–1.4 GB (1.6 GB for a read that runs to its token cap; before the staging
  1.9–2.1 / 2.4 GB — iPadOS killed the page at the second read). After the read the weights are freed; between reads
  the GPU holds 0.3 MiB.
- **Crash marker**: before each heavy step a marker goes to localStorage; still set at the next start means the tab
  or its GPU process was killed (out of memory) — the device is flagged until "Try anyway".
- **Device check** (`web/gpu-check.js`, the setup's first step and Help → Check this device): WebGPU present, a real
  (not software) adapter, the limits the kernels need, and a short bandwidth benchmark → an estimated reading speed
  with hints (e.g. a laptop running the browser on its integrated GPU).

### WebGPU pitfalls

Browser bugs met by llama.cpp's WebGPU backend ([Llamas on the Web](https://reeselevine.github.io/llamas-on-the-web/))
and where OCR Lite stands:

- **Silent miscompile on Chrome / Metal** ([crbug 504928024](https://issues.chromium.org/issues/504928024)): with Dawn's
  robustness and integer div / mod polyfills (Chrome's default) a valid shader wrote wrong values, no error. Trigger: a
  private array indexed by a `/` `%` result, a 2-D private array passed by value to a helper (`noinline` in the
  generated Metal hid it). Our kernels pass no array by value and index with shifts and masks. Wrong numbers show only
  as wrong text: **after a shader change, and now and then after a Chrome / Safari / macOS update, run the real read
  compared with the native read** (`pages` with `OCR_LITE_MODEL_DIR`) in Chrome and in Safari.
- **iOS hang with command buffers in flight** ([WebKit 311598](https://bugs.webkit.org/show_bug.cgi?id=311598)): on
  Safari / iOS 26 two or more submissions before a wait could leave the wait stuck — intermittent, no error, gone with
  the Web Inspector attached; 26.4 "more stable". WebKit's advice: one command buffer per submit with milliseconds of
  work, few compute passes (each pass costs). Ours: one pass per submit, but the greedy decode runs ahead with 2
  submissions in flight (`ocr-wgpu` `RUN_AHEAD_DEPTH`). No stall watchdog: a hang would leave a read waiting. Seen
  working on the iPad (current iPadOS); older iOS untested.
- **Metal lowers the workgroup limit per shader** ([gpuweb 5315](https://github.com/gpuweb/gpuweb/issues/5315)): a
  complex kernel can get fewer threads than WebGPU reports (704 instead of 1024 seen) and fail silently. Our largest
  workgroup is 256.
- **naga: no `vec2<f16>` ↔ `u32` bitcast** ([wgpu 8896](https://github.com/gfx-rs/wgpu/issues/8896), Firefox): we
  unpack bf16 / q8_0 from `u32` with shifts, no such bitcast.
- **Timestamp queries are heavy on WebKit** (a shared event per pass): the per-kernel profiler is native only.

## Licences

The code: MIT ([LICENSE](../LICENSE)). The model is separate: a fine-tune of LightOnOCR-3-1B (Apache-2.0, LightOn),
released under CC BY-SA 4.0 on Hugging Face. The About box lists both, the training sources and the main third-party
components.
