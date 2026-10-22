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
