// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// OCR Lite's image worker (browser build): the opened image and every step on it — ocr-web's open_image, preview,
// bake, auto_crop, baked_png. The full-size image and the copies the steps make live in THIS worker's wasm memory,
// never in the page's: a wasm memory only grows, and the page's held ~400 MiB after a 24 MP camera photo was opened
// and cropped — on top of a read's GPU memory iOS ended the page at its 2048 MB. The page (bridge-web.js) keeps only
// the file as it came (compressed), ends this worker before a read (its memory goes with it) and starts a new one from
// that file when the image is needed again; the steps (ops) come with every request, so nothing is lost.
// Messages in: { id, cmd, … } — reply { id, ok, result | error }; the first message of a worker is always "open".
import init, * as ocr from "./pkg/ocr_web.js";

const ready = init();

/// cmd → (args) → [result, transfer list]
const commands = {
  open({ bytes, name }) {
    return [JSON.parse(ocr.open_image(new Uint8Array(bytes), name)), []];
  },
  preview({ ops, sharp }) {
    const p = ocr.preview(JSON.stringify(ops), sharp);
    return [{ jpeg: p.jpeg, w: p.w, h: p.h, bake: p.bake }, [p.jpeg.buffer]];
  },
  bake({ ops }) {
    const b = ocr.bake(JSON.stringify(ops));
    return [{ rgb: b.rgb, w: b.w, h: b.h, plan: b.plan, seed: b.seed }, [b.rgb.buffer]];
  },
  autoCrop({ ops, opts }) {
    return [JSON.parse(ocr.auto_crop(JSON.stringify(ops), JSON.stringify(opts))), []];
  },
  bakedPng({ ops }) {
    const png = ocr.baked_png(JSON.stringify(ops));
    return [png, [png.buffer]];
  },
};

self.onmessage = async (e) => {
  const { id, cmd, ...args } = e.data;
  try {
    await ready;
    const f = commands[cmd];
    if (!f) {
      throw new Error(`image worker: no command ${cmd}`);
    }
    const [result, transfer] = f(args);
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};
