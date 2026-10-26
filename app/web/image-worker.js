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

/// a file the wasm decoder cannot read (HEIC / HEIF: the image crate has no decoder for it) decoded by the browser —
/// Safari reads HEIC; createImageBitmap applies the photo's orientation, the 2D canvas gives sRGB pixels → open_rgba.
/// null when the browser cannot decode it either
async function browserDecode(bytes, name) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(new Blob([bytes]));
  } catch {
    return null;
  }
  const { width: w, height: h } = bitmap;
  const ctx = new OffscreenCanvas(w, h).getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, w, h).data;
  return JSON.parse(ocr.open_rgba(new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength), w, h, name));
}

/// cmd → (args) → [result, transfer list] (or a Promise of it)
const commands = {
  async open({ bytes, name }) {
    try {
      return [JSON.parse(ocr.open_image(new Uint8Array(bytes), name)), []];
    } catch (e) {
      const viaBrowser = await browserDecode(bytes, name);
      if (viaBrowser === null) {
        throw e;
      }
      return [viaBrowser, []];
    }
  },
  preview({ ops, sharp }) {
    const p = ocr.preview(JSON.stringify(ops), sharp);
    return [{ jpeg: p.jpeg, w: p.w, h: p.h, bake: p.bake }, [p.jpeg.buffer]];
  },
  bake({ ops }) {
    const b = ocr.bake(JSON.stringify(ops));
    return [{ rgb: b.rgb, w: b.w, h: b.h, plan: b.plan, seed: b.seed }, [b.rgb.buffer]];
  },
  /// the baked model input at the Text size `size` ("smaller" / "average" / "larger": the page's glyph height scaled to
  /// the trained pages') — as bake + `scale` (JSON: the decision { factor, summary, reason, warning })
  bakeNormalized({ ops, size }) {
    const b = ocr.bake_normalized(JSON.stringify(ops), size);
    return [{ rgb: b.rgb, w: b.w, h: b.h, plan: b.plan, seed: b.seed, scale: b.scale }, [b.rgb.buffer]];
  },
  /// the page's writing size for the Text size gauge (glyph height, a sample blob, the targets)
  glyphHeight({ ops }) {
    return [JSON.parse(ocr.glyph_height(JSON.stringify(ops))), []];
  },
  /// a region's baked model input (the editor: per-region reads; `input` = its contrast / paper fill / scale / turns)
  bakeRegion({ ops, polygon, input }) {
    const b = ocr.bake_region(JSON.stringify(ops), JSON.stringify(polygon), JSON.stringify(input));
    return [{ rgb: b.rgb, w: b.w, h: b.h, plan: b.plan, seed: b.seed }, [b.rgb.buffer]];
  },
  /// the page at its bake scale, unpadded (the editor's ink analysis)
  scaled({ ops }) {
    const s = ocr.scaled(JSON.stringify(ops));
    return [{ rgb: s.rgb, w: s.w, h: s.h }, [s.rgb.buffer]];
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
    const [result, transfer] = await f(args);
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};
