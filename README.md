# Historic German Handwriting OCR

An open model that reads **whole pages of historic German handwriting** — Kurrent and Latin-script hands, mainly of
the 19th century — and writes them as written: line by line in page order, with struck-out, inserted and underlined
text kept as inline marks. The model, **LightOnOCR-3-1B · Historic German Handwriting**, is a fine-tune of
LightOnOCR-3-1B, trained only on openly licensed editions and archives, and released under CC BY-SA 4.0 on
[Hugging Face](https://huggingface.co/b-straub/LightOnOCR-3-1B-Historic-German-Handwriting).

**Try it: [OCR Lite](https://b-straub.github.io/historic-htr-ocr/app/)** — a small web app that runs the model entirely
on your device with WebGPU (Chrome / Edge 113+, Safari 26+; iPhone experimental): the model is downloaded once from
Hugging Face into the browser's storage, and your images never leave the machine. A tech preview: edits are not kept
yet.

The [project site](https://b-straub.github.io/historic-htr-ocr/) explains the output format and how the model was
trained and scored. Questions, problems and reading errors worth reporting:
[issues](https://github.com/b-straub/historic-htr-ocr/issues).

## What's new in v1.5.0

- **German and English**: OCR Lite follows the browser's language; Settings can fix it. A **How to** (the five steps,
  also shown before the first image) guides first-time users.
- **Guided crop**: when an image opens, **Auto** puts a rectangle around the writing by itself — fit it to whole lines
  of the text you want, then the green ✓; Read waits for it. A **book spread** is recognised: the ▾ beside Auto asks
  which page (Left, Right, Double). An archive's printed label below a book stays out of the crop.
- **Touch**: the rectangle has handles at its corners and edge middles; the one you hold lights up. A finger elsewhere
  moves the image, two fingers zoom. While a read runs the image stays still.
- **Scale after the read**: a crop is read at its own size. Afterwards OCR Lite measures how wide the characters reach
  the model; when they are smaller or larger than on the training pages, the **wand** lights up and reads again at a
  better size (its ▾ offers a little smaller, a little larger and ×1). Save → model input image shows what the model
  sees, at the current scale.
- **Expert mode** (Settings): off by default — the presets and technical details stay out of the way. The model shows
  as one entry with its version; a newer version is offered when it comes out.
- **Install**: a sheet per browser and device shows how to add OCR Lite as an app.
- **Firefox** is not supported (its storage and timer limits); OCR Lite runs in Chrome, Edge and Safari 26+.

## This repository

| Path | What |
|---|---|
| [`app/`](app/) | **OCR Lite** — the browser app: a Rust + wgpu engine for LightOnOCR compiled to WebAssembly, the UI, the tests. Build, test, run locally and the model setting: [app/README.md](app/README.md) |
| [`site/`](site/) | the project site (GitHub Pages): the pages, their style, the example inputs (`examples/`: the model's output + the page image; the viewer pages are generated at build) |
| [`hf/`](hf/) | the Hugging Face model repo's files: the model card (`README.md`), its images, `convert_output.py` (the output → TEI / an HTML viewer), `window_rep_penalty.py` (the windowed repetition penalty for vLLM) |
| `.github/workflows/` | `ci.yml` — the full build and tests on every push and pull request; `pages.yml` — on a tag `v*.*.*`: the same CI, then the Pages deploy (only when it passed) |

Nothing built is committed: the CI builds the app and the site (`app/ci.sh`).

## Licence

The code in this repository: MIT (see [LICENSE](LICENSE)). The model: CC BY-SA 4.0, on Hugging Face; its base model
LightOnOCR-3-1B is Apache-2.0 (© LightOn). The texts of the site (`site/`): CC BY 4.0.
