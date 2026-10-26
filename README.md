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

## What's new in v1.3.0

- **Text size** for photos and close-ups: a select in the image head scales the page so its letters reach the model as
  tall as on the pages it learned from (smaller / average page / larger; *as is* leaves the image unchanged). A gauge
  on the image shows a word's letter height now and the height the read gets — and says when only a tighter crop can
  enlarge it further. The model size beside the image follows the choice at once.
- **One crop bar**, open when an image loads, over the image head (never over the image): **Draw** a rectangle by hand,
  or tap a page mode (**Auto, Left, Right, Double, Leaf**) to find the paper. Any rectangle can be adjusted by its edges
  and corners, then **Crop** or **Cancel**; **Reset** goes back to the original.
- **Gestures that stay put**: drawing a crop no longer zooms or moves the image when a second finger touches; two
  fingers zoom, also while cropping; on a computer the cursor shows where a rectangle can be resized or moved.
- **Phones and tablets**: Image / Text tabs where the two panes do not fit side by side, each keeping its zoom and
  scroll; pinch to zoom the text; a compact two-line image head held upright, Help and Settings beside the image on a
  phone's side.
- **Steadier reads**: a read starts from the fitted image and the zoom is capped — a strongly zoomed image no longer
  holds graphics memory beside the read (on an iPhone it likely ended the page).

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
