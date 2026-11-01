// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// OCR Lite UI: image + prep chain on the left, the read as an editable rendered page on the right (editor.js).
// Everything heavy happens in Rust (ocr-web's wasm); this file keeps the prep chain and wires events.
// The host is web/bridge-web.js (ocr-web wasm + a WebGPU worker) — commands, events, file pick / drop / save, the
// host's settings.
import { t, fixed, upTo, applyI18n, setChoice, getChoice, getLang } from "./i18n.js";
import { scanOverlay } from "./scan.js";
import { setupDialog, installSheet, INSTALL_PLATFORMS } from "./setup.js";
import { setTip } from "./tip.js";
import * as host from "./bridge-web.js";
const { invoke, listen } = host;
const $ = (id) => document.getElementById(id);
// a mouse / keyboard device (else a touch screen: no key hints)
const KEYS = !matchMedia("(hover: none) and (pointer: coarse)").matches;

// ── icons (24 px grid, stroke) ────────────────────────────────────────
const ICONS = {
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  rotl: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
  rotr: '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
  flip: '<path d="M12 3v18"/><path d="M8 7 4 12l4 5"/><path d="m16 7 4 5-4 5"/>',
  contrast: '<circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor"/>',
  pencil: '<path d="M4 20l1.2-4.6L15.6 5a2 2 0 0 1 2.8 0l.6.6a2 2 0 0 1 0 2.8L8.6 18.8z"/><path d="m13.5 7.1 3.4 3.4"/>',
  zoomin: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/><path d="M8 11h6"/><path d="M11 8v6"/>',
  zoomout: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/><path d="M8 11h6"/>',
  fit: '<path d="M4 9V5a1 1 0 0 1 1-1h4"/><path d="M15 4h4a1 1 0 0 1 1 1v4"/><path d="M20 15v4a1 1 0 0 1-1 1h-4"/><path d="M9 20H5a1 1 0 0 1-1-1v-4"/>',
  undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/>',
  redo: '<path d="m15 14 5-5-5-5"/><path d="M20 9H9a5 5 0 0 0 0 10h3"/>',
  reset: '<path d="M3 12a9 9 0 1 0 9-9"/><path d="M3 4v5h5"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
  play: '<path d="M7 4.5v15l12-7.5z" fill="currentColor"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
  save: '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
  table: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18"/><path d="M12 5v14"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/>',
  close: '<path d="M6 6l12 12M18 6 6 18"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.4a2.5 2.5 0 1 1 3.4 2.4c-.6.3-1 .8-1 1.5v.5"/><circle cx="12" cy="16.8" r=".6" fill="currentColor"/>',
  textsize: '<path d="m10 19 5-14 5 14"/><path d="M11.8 14h6.4"/><path d="m2 19 3.3-8 3.3 8"/><path d="M3.2 16.2h4.2"/>',
  wand: '<path d="M15 4V2"/><path d="M15 16v-2"/><path d="M8 9h2"/><path d="M20 9h2"/><path d="M17.8 11.8 19 13"/><path d="M15 9h.01"/><path d="M17.8 6.2 19 5"/><path d="m3 21 9-9"/><path d="M12.2 6.2 11 5"/>',
  book: '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z"/><path d="M4 20.5A2.5 2.5 0 0 0 6.5 23H20v-5"/><path d="M8 7h8"/><path d="M8 11h6"/>',
  type: '<path d="M4 6h16"/><path d="M4 11h11"/><path d="M4 16h7"/><path d="M18 13v8"/><path d="M16 13h4"/><path d="M16 21h4"/>',
  // the browsers' menu items, inline in the install steps (setup.js): Share (square with an arrow), ⋮, …
  share: '<path d="M12 3v12"/><path d="m8 7 4-4 4 4"/><path d="M6 11H5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-8a1 1 0 0 0-1-1h-1"/>',
  dots: '<circle cx="12" cy="5" r="1.4" fill="currentColor"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/><circle cx="12" cy="19" r="1.4" fill="currentColor"/>',
  installmon: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8"/><path d="M12 16v4"/><path d="M12 7v5"/><path d="m9.5 9.5 2.5 2.5 2.5-2.5"/>',
  appavail: '<rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><path d="M17 14v6"/><path d="M14 17h6"/>',
  hdots: '<circle cx="5" cy="12" r="1.4" fill="currentColor"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/><circle cx="19" cy="12" r="1.4" fill="currentColor"/>',
  autocrop: '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/><path d="m11 11 3 3" stroke-dasharray="1 2"/>',
};
function paintIcons(scope = document) {
  for (const el of scope.querySelectorAll("i[data-icon]")) {
    el.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[el.dataset.icon]}</svg>`;
  }
}

const S = {
  ops: [],          // the prep chain (imageops::Op), applied in order to the opened image
  prev: null,       // the last preview: { url, w, h, bake }
  readBake: null,   // the bake plan of the read's model input (Scale may differ from the preview's)
  readInputW: 0,    // the read's model input width (unpadded): the characters' measure after it
  // the scale, reset by every crop change: factor = what the read applies on top of the bake scale (1 = ×1, or the
  // 1540 px edge); cw = the last read's width per character at the model input, at = its step (the wand's menu),
  // undefined before a read
  size: { factor: 1, cw: undefined, at: undefined },
  sharp: false,     // the shown preview is the sharp (zoomed-in) one
  stem: "",
  reading: false,
  stopping: false,  // Stop pressed, the engine's ocr:done / ocr:error not yet in
  starting: false,  // Read pressed: the checks before the read (model in storage, its source) run
  finding: false,   // Find crop runs (auto_crop)
  model: { ready: false, why: t("why.checkingModel") }, // can a read start? why not (checkModel)
  streamed: "",
  settings: null,
  presets: null,    // { shipped, defaultName, user }
  marks: null,
  editor: null,
  backend: "",
};

// link = { href, label } or { run, label } (an action: run() does it and
// returns the status that follows, e.g. the unsupported-device notice's "Try anyway" — null: the status stays):
// appended to the message
function status(msg, kind = "", link = null) {
  $("status").textContent = msg;
  if (link && link.run) {
    const a = Object.assign(document.createElement("a"), { href: "#", textContent: link.label });
    a.onclick = (e) => {
      e.preventDefault();
      const next = link.run();
      if (next !== null) {
        status(next, "ok");
      }
    };
    $("status").append(" — ", a);
  } else if (link) {
    $("status").append(" — ", Object.assign(document.createElement("a"), { href: link.href, textContent: link.label }));
  }
  $("status").classList.toggle("err", kind === "err");
  $("dot").className = `dot ${kind}`;
}
async function call(cmd, args) {
  try {
    return await invoke(cmd, args);
  } catch (e) {
    status(String(e), "err");
    throw e;
  }
}

// ── breadcrumbs (rare iPad crashes on crop / contrast / zoom that cannot be forced) ──────────────────────
// The last actions live in sessionStorage — it survives this tab's reload after a crash, not a closed tab. "alive" is
// set on load and removed on pagehide (every normal unload / reload / navigation): a load that still finds it means the
// page before ended without unloading — a crash, or iOS dropping the tab. Then the status offers the trail to copy.
const CRUMBS = 40;
const crumbs = [];
let zoomCrumb = false; // the last crumb is a zoom step (a pinch makes many: kept as one, updated)
function crumb(what, isZoom = false) {
  const line = `${new Date().toISOString().slice(11, 23)} ${what}`;
  if (isZoom && zoomCrumb) {
    crumbs[crumbs.length - 1] = line;
  } else {
    crumbs.push(line);
  }
  zoomCrumb = isZoom;
  if (crumbs.length > CRUMBS) {
    crumbs.shift();
  }
  try { sessionStorage.setItem("ocrlite.crumbs", JSON.stringify(crumbs)); } catch {}
}
let lastCrash = null; // the trail of a page that ended without unloading
// the page before ended while the app was in the background ("ocrlite.away", set by wentAway, removed by cameBack): iOS
// closing a background app (iOS Safari sends no pagehide on an app switch) — no crash, a note in the status line
let closedAway = false;
try {
  if (sessionStorage.getItem("ocrlite.alive")) {
    if (sessionStorage.getItem("ocrlite.away")) {
      closedAway = true;
    } else {
      lastCrash = JSON.parse(sessionStorage.getItem("ocrlite.crumbs") ?? "[]");
    }
  }
  sessionStorage.removeItem("ocrlite.away");
  sessionStorage.setItem("ocrlite.alive", "1");
  sessionStorage.removeItem("ocrlite.crumbs");
} catch {}
window.addEventListener("pagehide", () => { try { sessionStorage.removeItem("ocrlite.alive"); } catch {} });
// after a crash: one dialog (not the status line) — what happened, the stored model checked by itself (iOS ending the
// app in the middle of a read left it damaged; the result appears in the dialog), and "Copy the last actions" (the
// breadcrumb trail, for a report)
async function verifyAfterCrash() {
  const line = $("crashCheck");
  let missing;
  try {
    missing = await host.setup.modelMissing();
  } catch (e) {
    line.classList.add("err");
    line.textContent = t("st.storage", { e });
    return;
  }
  if (missing) {
    return;
  }
  line.classList.remove("err");
  line.textContent = t("crash.checking");
  try {
    const r = await host.verifyModel();
    crumb(`model check: ${r.ok ? "intact" : `damaged (${r.damaged.join(", ")})`}`);
    line.textContent = r.text;
    line.classList.toggle("err", !r.ok);
    if (!r.ok) {
      status(r.text, "err");
    }
  } catch (e) {
    line.textContent = t("crash.checkFailed", { e: e instanceof Error ? e.message : e });
    line.classList.add("err");
  }
}
// `crash`: the run before died in a GPU step (the crash marker) or lost the GPU (host.takeCrashNote); lastCrash: the
// page before ended without unloading (its breadcrumb trail)
function showCrashNote(crash) {
  if (!crash && !lastCrash) {
    return;
  }
  $("crashWhat").textContent = crash ?? t("crash.what", { last: lastCrash.at(-1) ?? t("crash.none") });
  $("crashCheck").textContent = "";
  $("crashCopy").hidden = !lastCrash;
  $("crashCopy").textContent = t("crash.copy");
  $("crashCopy").onclick = () => {
    const trail = [`OCR Lite — the page before ended without unloading (${navigator.userAgent})`, ...lastCrash].join("\n");
    navigator.clipboard.writeText(trail).then(() => { $("crashCopy").textContent = t("crash.copied"); });
  };
  $("crashDlg").showModal();
  if (crash) {
    verifyAfterCrash();
  }
}

// ── image + prep chain ────────────────────────────────────────────────
// every open gets a number: a step's preview still under way when a new image opens is dropped (refresh)
let openSeq = 0;
async function openPath(src) {
  // Open is disabled while busy (applyState); a drop or the setup's "open an image" can still arrive
  if (isBusy()) {
    status(busyWhy(), "err");
    return;
  }
  let o;
  openSeq++;
  try {
    o = await host.openImage(src);
  } catch (e) {
    status(String(e), "err");
    throw e;
  }
  S.stem = o.stem;
  S.ops = [];
  uncropped = null;
  spreadSeen = false;
  drawCropSide();
  sizeKey = null;
  setTab("image");
  zoom = 1;
  crumb(`open ${o.stem} ${o.w}×${o.h}`);
  status(`${o.stem} · ${o.w} × ${o.h}`);
  await refresh();
  await textCrop(); // a crop is due: Auto puts its rectangle up at once
}
// every preview request gets a number; only the latest one's answer is shown (a sharp preview still being made when
// the chain changes is dropped)
let previewSeq = 0;
async function refresh() {
  cancelCrop();
  const seq = ++previewSeq;
  const old = S.prev;
  const sharp = S.sharp && zoom > 1; // zoomed in on a sharp preview: ask for the sharp one straight away
  const opened = openSeq;
  let p;
  try {
    p = await call("preview", { ops: S.ops, sharp });
  } catch (e) {
    if (opened !== openSeq) {
      return; // a new image opened meanwhile: its worker replaced this one, the step's preview is moot
    }
    throw e;
  }
  if (seq !== previewSeq) return;
  S.prev = p;
  S.sharp = sharp;
  if (!old || old.w !== p.w || old.h !== p.h) zoom = 1; // a crop / rotation: back to fit; a filter keeps the view
  $("img").src = p.url;
  $("stage").hidden = false;
  $("drop").hidden = true;
  layout();
  drawBakeInfo();
  applyState();
  measureSize(); // a new image / step: the Scale's range measured again on a crop
}
// contrast / pencil are on or off, never stacked: a second press takes the step out again (they change no geometry,
// so the crops after them keep their pixels)
const FILTERS = new Set(["autoContrast", "pencil"]);
// before the crop (a crop is due) a turn or a filter changes what Auto sees: its rectangle is found again
async function pushOp(op) {
  if (!S.prev) return;
  const again = op.op !== "crop" && !hasCrop();
  const on = S.ops.findIndex((o) => o.op === op.op);
  if (FILTERS.has(op.op) && on >= 0) {
    crumb(`${op.op} off`);
    S.ops.splice(on, 1);
  } else {
    crumb(op.op === "crop" ? `crop ${op.w}×${op.h} at ${op.x},${op.y}` : op.op);
    S.ops.push(op);
  }
  await refresh();
  if (again) {
    await textCrop();
  }
}
document.querySelectorAll("[data-op]").forEach((b) => b.addEventListener("click", () => pushOp({ op: b.dataset.op })));
$("imgUndoBtn").onclick = () => { if (S.ops.length) { crumb("undo"); S.ops.pop(); refresh(); } };
// back to the original: the uncropped image again, so Auto's rectangle comes up again (an undo of a crop does not)
$("resetBtn").onclick = async () => {
  if (S.ops.length) {
    crumb("reset");
    S.ops = [];
    uncropped = null;
    await refresh();
    await textCrop();
  }
};
async function pickImage() {
  crumb("picker open"); // iOS puts the app in the background while the picker is up
  const p = await host.pickImage();
  if (p) openPath(p);
}
$("openBtn").onclick = pickImage;
$("openLink").onclick = (e) => { e.preventDefault(); pickImage(); };
host.onDrop({
  enter: () => $("imgPane").classList.add("dragover"),
  leave: () => $("imgPane").classList.remove("dragover"),
  drop: (src) => openPath(src),
});

// ── zoom: the image's displayed size = fit × zoom, zoom 1 … 8; the pane scrolls when larger.
// Fit = the image inside the pane, never above the prepared image's own size. Everything over the image (crop rect,
// pill, scan overlay) is kept in prepared-image pixels and drawn × D, so a crop is exact at every zoom.
const PANE_PAD = 18;    // .imgwrap padding
const ZOOM_MAX = 8;
// the zoomed image's longest side at most this many DEVICE px: an iPhone ended the page in a read while a small crop
// was zoomed to 800 % (~7900 × 9500 device px, ~300 MB of layer beside the read's GPU memory) — the cap a guess,
// UNMEASURED; beyond the image's own pixels a zoom only magnifies anyway
const ZOOM_LAYER_MAX = 6144;
function zoomMax() {
  if (!S.prev) return ZOOM_MAX;
  const fit = fitScale() * Math.max(S.prev.w, S.prev.h) * (window.devicePixelRatio || 1);
  return Math.max(1, Math.min(ZOOM_MAX, ZOOM_LAYER_MAX / fit));
}
const ZOOM_STEP = 1.25;
let zoom = 1;
let D = 1;              // CSS px per prepared-image px
function fitScale() {
  const pane = $("imgPane");
  // offsetWidth (with the scrollbar): the fit must not shrink when the zoomed pane shows scrollbars
  const aw = pane.offsetWidth - 2 * PANE_PAD, ah = pane.offsetHeight - 2 * PANE_PAD;
  return Math.max(0.001, Math.min(1, aw / S.prev.w, ah / S.prev.h));
}
// anchor = { from, to } in client px: the image point under `from` before the change ends up under `to`
function layout(anchor = null) {
  if (!S.prev) return;
  const pane = $("imgPane"), img = $("img");
  if (pane.offsetWidth === 0) {
    return; // the image tab is away: its zoom and scroll stay for its return
  }
  let fx = 0, fy = 0;
  if (anchor) {
    const r = img.getBoundingClientRect();
    fx = (anchor.from.x - r.left) / r.width;
    fy = (anchor.from.y - r.top) / r.height;
  }
  D = fitScale() * zoom;
  // zoomed in: no rounded corners / shadow on the (then very large) image — suspect for the iPad's rare crashes: a
  // shadowed, clipped layer of many thousand device px (unconfirmed)
  $("stage").classList.toggle("zoomed", zoom > 1);
  img.style.width = `${S.prev.w * D}px`;
  img.style.height = `${S.prev.h * D}px`;
  if (anchor) {
    const r = img.getBoundingClientRect();
    pane.scrollLeft += r.left + fx * r.width - anchor.to.x;
    pane.scrollTop += r.top + fy * r.height - anchor.to.y;
  }
  $("zoomVal").textContent = `${Math.round(zoom * 100)} %`;
  drawOverlay(); // → applyState: the zoom buttons
  sharpen();
}
function zoomTo(z, from, to = from) {
  if (!S.prev || $("stage").hidden) return;
  z = Math.min(zoomMax(), Math.max(1, z));
  if (z < 1.005) z = 1;
  zoom = z;
  crumb(`zoom ${Math.round(z * 100)} %`, true);
  layout({ from, to });
}
function paneCentre() {
  const r = $("imgPane").getBoundingClientRect();
  return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
}
$("zoomInBtn").onclick = () => zoomTo(zoom * ZOOM_STEP, paneCentre());
$("zoomOutBtn").onclick = () => zoomTo(zoom / ZOOM_STEP, paneCentre());
$("zoomFitBtn").onclick = () => zoomTo(1, paneCentre());
new ResizeObserver(() => layout()).observe($("imgPane"));

// zoomed beyond the preview's own pixels: ONE sharper preview of the prepared image (imageops::preview_longest(true)),
// replacing the normal one (the browser keeps one preview at a time); a new prep chain starts again from the normal one
let sharpening = false;
async function sharpen() {
  const img = $("img");
  if (!S.prev || S.sharp || sharpening || !img.naturalWidth) return;
  const have = img.naturalWidth / S.prev.w; // preview px per prepared-image px
  if (have >= 1 || D * (window.devicePixelRatio || 1) <= have * 1.05) return;
  sharpening = true;
  const seq = ++previewSeq;
  try {
    const p = await call("preview", { ops: S.ops, sharp: true });
    if (seq !== previewSeq) return;
    S.prev = { ...S.prev, url: p.url };
    S.sharp = true;
    crumb("sharp preview");
    img.src = p.url;
  } finally {
    sharpening = false;
  }
}
$("img").addEventListener("load", () => sharpen());

// crop: only in the crop mode (Auto puts a rectangle up — explicitly, since a zoom gesture drew crops); there
// press-drag-release adjusts the rectangle or draws a new one; Crop / Enter applies, Cancel / Esc drops it and leaves the
// mode. Outside the mode a drag pans the image.
let cropMode = false;
// Auto on a cropped image starts from the uncropped one: the chain before (Cancel puts it back), else null
let uncropped = null;
let pending = null; // { x, y, w, h } in prepared-image pixels
let drag = null;    // the rectangle being drawn: { id, start, cur, prev } (points in prepared-image pixels)
// the rectangle a drag gives: a new one from where it began (mode "new"), or the drawn one (d.base) with the grabbed
// edges moved ("l" / "r" / "t" / "b", a corner = two) or all of it ("move"); kept on the image
function dragBox(d) {
  if (d.mode === "new") {
    return { x: Math.min(d.start.x, d.cur.x), y: Math.min(d.start.y, d.cur.y), w: Math.abs(d.cur.x - d.start.x), h: Math.abs(d.cur.y - d.start.y) };
  }
  const b = d.base, W = S.prev.w, H = S.prev.h;
  let dx = d.cur.x - d.start.x, dy = d.cur.y - d.start.y;
  if (d.mode === "move") {
    dx = Math.min(Math.max(dx, -b.x), W - b.x - b.w);
    dy = Math.min(Math.max(dy, -b.y), H - b.y - b.h);
    return { x: b.x + dx, y: b.y + dy, w: b.w, h: b.h };
  }
  let x0 = b.x, y0 = b.y, x1 = b.x + b.w, y1 = b.y + b.h;
  if (d.mode.includes("l")) x0 += dx;
  if (d.mode.includes("r")) x1 += dx;
  if (d.mode.includes("t")) y0 += dy;
  if (d.mode.includes("b")) y1 += dy;
  [x0, x1] = [Math.max(0, Math.min(x0, x1)), Math.min(W, Math.max(x0, x1))];
  [y0, y1] = [Math.max(0, Math.min(y0, y1)), Math.min(H, Math.max(y0, y1))];
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}
// where a pointer lands on the drawn rectangle (image px): its edges / corners within reach (`px` screen px), inside =
// move, else null (a new rectangle)
const GRAB_PX = 22, GRAB_TOUCH_PX = 28;
function grabOf(b, p, px) {
  const r = px / D;
  const inX = p.x > b.x - r && p.x < b.x + b.w + r, inY = p.y > b.y - r && p.y < b.y + b.h + r;
  if (!inX || !inY) return null;
  let m = "";
  if (Math.abs(p.x - b.x) < r) m += "l";
  else if (Math.abs(p.x - (b.x + b.w)) < r) m += "r";
  if (Math.abs(p.y - b.y) < r) m += "t";
  else if (Math.abs(p.y - (b.y + b.h)) < r) m += "b";
  if (m !== "") return m;
  return p.x > b.x && p.x < b.x + b.w && p.y > b.y && p.y < b.y + b.h ? "move" : null;
}
// the rect (drawing or pending) and the pill at the current zoom
// ── Image / Text tabs: upright on a narrow screen one card at a time (style.css); each keeps its zoom, scroll and
// fit while away — the hidden pane is neither laid out nor fitted (layout / fitText), its scroll put back on return
// tab mode (body.stacked): a narrow screen upright, and a phone on its side (too little width for two panes) — there
// the tabs sit in the bar after the logo, upright in a bar at the bottom
const STACKED = matchMedia("(max-width: 820px) and (orientation: portrait), (orientation: landscape) and (max-height: 500px)");
const COMPACT = matchMedia("(orientation: landscape) and (max-height: 500px)");
const tabsHome = { parent: $("paneTabs").parentNode, next: $("paneTabs").nextSibling };
function placeTabs() {
  document.body.classList.toggle("stacked", STACKED.matches);
  if (COMPACT.matches) {
    $("aboutBtn").after($("paneTabs"));
  } else {
    tabsHome.parent.insertBefore($("paneTabs"), tabsHome.next);
  }
}
const tabScroll = { image: { left: 0, top: 0 }, text: { left: 0, top: 0 } };
const tabScroller = (t) => (t === "image" ? $("imgPane") : markupView ? $("src") : $("editor"));
function setTab(t) {
  const now = document.body.dataset.tab;
  if (now === t) {
    return;
  }
  if (now === "image" || now === "text") {
    const el = tabScroller(now);
    tabScroll[now] = { left: el.scrollLeft, top: el.scrollTop };
  }
  document.body.dataset.tab = t;
  for (const b of document.querySelectorAll("#paneTabs [data-tab]")) {
    b.classList.toggle("on", b.dataset.tab === t);
    b.setAttribute("aria-selected", String(b.dataset.tab === t));
  }
  crumb(`tab ${t}`);
  if (!STACKED.matches) {
    return;
  }
  requestAnimationFrame(() => {
    if (t === "image") {
      layout();
    } else {
      fitText();
    }
    const el = tabScroller(t);
    el.scrollLeft = tabScroll[t].left;
    el.scrollTop = tabScroll[t].top;
  });
}
for (const b of document.querySelectorAll("#paneTabs [data-tab]")) {
  b.onclick = () => setTab(b.dataset.tab);
}
placeTabs();
setTab("image");
// into / out of tab mode: the tabs placed, both panes laid out again
const retab = () => {
  placeTabs();
  requestAnimationFrame(() => {
    layout();
    fitText();
  });
};
STACKED.addEventListener("change", retab);
COMPACT.addEventListener("change", retab);

// ── Scale: no slider — a crop is read at ×1 (its own pixels; past the 1540 px edge at the edge). After a read its lines
// measure the model input's width per character — the median over the lines of input width / the line's characters —
// against the training pages' (the same per page over the free training pages, bake px: p25 19 · p33 20 · median 23 ·
// p67 27 · p75 29). The band sits lower — middle 20 (p33), ends 18 and 23: a training page's width carries its margins,
// a tight crop has few, so the same hand gives fewer px a character there (user: the lower range read better). After
// a read the wand reads again at the band's middle; its ▾ opens a menu: the middle, the lower and upper end — each a factor rounded to 0.1 (0.25 from ×1.5 on), never past the 1540 px
// edge; a step reads again with it. Outside the band the wand is lit and the status line says why; on while a step
// other than ×1 is in use. S.size.factor = the factor on top of the bake scale (bake_scaled), 1 = ×1 or the edge ──
const BAND_LO = 18, BAND_MID = 20, BAND_HI = 23, LINE_MIN_LINES = 3;
const LONGEST = 1540;
// the prepared image's bake scale (bake px per own px) and the factor from its own pixels to the 1540 px edge
const sizeOf = (p) => ({ bake: p.bake.scaledW / p.w, edge: LONGEST / Math.max(p.w, p.h) });
// a factor rounded for the menu (0.1, from ×1.5 on 0.25), never past the edge (nor below 0.1)
function roundStep(f, edge) {
  const g = f >= 1.5 ? 0.25 : 0.1;
  const s = Math.max(0.1, Math.round(f / g) * g);
  return s > edge ? edge : s;
}
// a step in words: ×1/3, ×1, ×2.5, the edge's factor ×0.51
const stepText = (s) => {
  const n = Math.round(1 / s);
  return s < 1 && n >= 2 && Math.abs(1 / s - n) < 1e-6 ? `×1/${n}` : `×${upTo(s, 2)}`;
};
const hasCrop = () => S.ops.some((o) => o.op === "crop");
// a result without a measure ({ none, code } of text_crop, ocr-web) in words: from its code; `none` is the detail for
// the trail
const noneWhy = (r) => t(`none.${r.code}`, r);
// the scale is geometry: a crop or a turn starts again at ×1 — contrast / pencil keep it
const GEOM = new Set(["crop", "rotateLeft", "rotateRight", "rotate180"]);
let sizeKey = null; // the geometric steps of the current scale; null = a new image
function measureSize() {
  const key = JSON.stringify(S.ops.filter((o) => GEOM.has(o.op)));
  if (key !== sizeKey) {
    sizeKey = key;
    S.size = { factor: 1, cw: undefined, at: undefined };
  }
  drawBakeInfo();
  applyState();
}
// the wand's menu after a read: the middle of the training's band, its lower and upper end (merged when they meet) and
// ×1 — each { step, px (the characters' width it gives), key }
function scaleChoices() {
  const c = S.size.cw, at = S.size.at, edge = sizeOf(S.prev).edge, out = [];
  for (const [target, key] of [[BAND_MID, "scale.menu.mid"], [BAND_LO, "scale.menu.lo"], [BAND_HI, "scale.menu.hi"]]) {
    const step = roundStep(at * target / c, edge);
    if (!out.some((o) => Math.abs(o.step - step) < 1e-6)) {
      out.push({ step, px: Math.round(c * step / at), key });
    }
  }
  const one = Math.min(1, edge);
  if (!out.some((o) => Math.abs(o.step - one) < 1e-6)) {
    out.push({ step: one, px: Math.round(c * one / at), key: "scale.menu.one" });
  }
  return out;
}
// the wand after a read: the middle step (scaleChoices' first) and the read again — in use already: back to ×1 (no
// read); lit when the read's characters are outside the training's band, on while a step other than ×1 is in use
$("scaleAutoBtn").onclick = () => {
  const r = sizeOf(S.prev), mid = scaleChoices()[0], now = S.size.factor * r.bake;
  if (Math.abs(mid.step - now) < 1e-6) {
    S.size.factor = 1;
    crumb("scale by the lines off: ×1");
    drawBakeInfo();
    applyState();
    return;
  }
  S.size.factor = mid.step / r.bake;
  crumb(`scale by the lines ${stepText(mid.step)}`);
  drawBakeInfo();
  applyState();
  $("readBtn").click();
};
// its ▾: the menu (scaleChoices; the current step ✓) — a step reads again with it, ×1 only sets it back
$("scaleMenuBtn").onclick = (e) => {
  e.stopPropagation();
  const menu = $("scaleMenu");
  if (!menu.hidden) {
    menu.hidden = true;
    return;
  }
  const now = S.size.factor * sizeOf(S.prev).bake;
  const head = Object.assign(document.createElement("span"), { className: "menuhead",
    textContent: t("scale.menu.head", { cw: Math.round(S.size.cw), lo: BAND_LO, hi: BAND_HI }) });
  const items = scaleChoices().map((o) => {
    const b = Object.assign(document.createElement("button"), { role: "menuitemradio",
      textContent: t(o.key, { step: stepText(o.step), px: o.px }) });
    const on = Math.abs(o.step - now) < 1e-6;
    b.classList.toggle("on", on);
    b.setAttribute("aria-checked", String(on));
    b.onclick = () => {
      menu.hidden = true;
      S.size.factor = o.step / sizeOf(S.prev).bake;
      crumb(`scale by the lines ${stepText(o.step)}`);
      drawBakeInfo();
      applyState();
      if (o.key !== "scale.menu.one") {
        $("readBtn").click();
      }
    };
    return b;
  });
  menu.replaceChildren(head, ...items);
  menu.hidden = false;
};
document.addEventListener("click", () => { $("scaleMenu").hidden = true; });
// after a read: its width per character at the model input (S.size.cw at S.size.at, the wand's menu) — the first
// read on a crop measures, later ones keep it (the menu stays the same whichever step is in use; a crop change
// measures again); returns the status line's note when outside the training's band ("" inside or without a measure)
function lineSuggest(text, inputW) {
  if (!hasCrop()) {
    return "";
  }
  if (S.size.cw !== undefined) {
    return scaleOff() ? t("done.scale", { cw: Math.round(cwNow()), lo: BAND_LO, hi: BAND_HI, step: stepText(scaleChoices()[0].step) }) : "";
  }
  const lens = text.split("\n").map((l) => l.replace(/<[^>]*>/g, "").trim().length).filter((n) => n > 0);
  if (lens.length < LINE_MIN_LINES) {
    return "";
  }
  const per = lens.map((n) => inputW / n).sort((a, b) => a - b);
  const m = per.length >> 1, cw = per.length % 2 ? per[m] : (per[m - 1] + per[m]) / 2;
  const now = S.size.factor * sizeOf(S.prev).bake;
  S.size.cw = cw;
  S.size.at = now;
  crumb(`lines ${lens.length}, ${cw.toFixed(1)} px a character at ${stepText(now)} (training band ${BAND_LO} … ${BAND_HI})`);
  if (!scaleOff()) {
    return "";
  }
  return t("done.scale", { cw: Math.round(cw), lo: BAND_LO, hi: BAND_HI, step: stepText(scaleChoices()[0].step) });
}
// the width per character at the step in use (from the crop's measure)
const cwNow = () => S.size.cw * S.size.factor * sizeOf(S.prev).bake / S.size.at;
// the wand lit: the characters at the step in use outside the training's band
function scaleOff() {
  if (S.size.cw === undefined) {
    return false;
  }
  const c = cwNow();
  return c < BAND_LO || c > BAND_HI;
}
// the status after the tools: the crop (or the image) → the model input — "1300 × 800 → 1560 × 960" — and the scale
// from the image's own pixels ("×2"); the model input in its tooltip (a tap shows it in the status line). The scaled
// size = the preview's bake scale × the factor, as bake_scaled makes it (imageops::plan pads it to 56 px and the 28 px
// grid)
function drawBakeInfo() {
  const p = S.prev;
  if (p === null) {
    return;
  }
  // a rectangle up: its size (what Crop gives)
  const rb = drag ? dragBox(drag) : pending;
  if (rb) {
    $("bakeInfo").textContent = `${rb.w} × ${rb.h}`;
    setTip($("bakeInfo"), t("bake.rect", { w: p.w, h: p.h, rw: rb.w, rh: rb.h, x: rb.x, y: rb.y }));
    $("scaleInfo").textContent = "";
    return;
  }
  const b = p.bake, f = S.size.factor;
  const w = f === 1 ? b.scaledW : Math.max(1, Math.round(b.scaledW * f)), h = f === 1 ? b.scaledH : Math.max(1, Math.round(b.scaledH * f));
  const by = stepText(w / p.w);
  // the Read row: the image (or crop) → the model input ("from" goes on a narrow phone); the Adjust row: the scale
  $("bakeInfo").replaceChildren(Object.assign(document.createElement("span"), { className: "from", textContent: `${p.w} × ${p.h} → ` }), `${w} × ${h}`);
  $("scaleInfo").textContent = by;
  let fw = w, fh = h;
  if (Math.min(fw, fh) < 56) {
    if (fw <= fh) { fw = 56; } else { fh = 56; }
  }
  const tip = t("bake.title", { w: p.w, h: p.h, by, sw: w, sh: h, mw: Math.ceil(fw / 28) * 28, mh: Math.ceil(fh / 28) * 28 });
  setTip($("bakeInfo"), tip);
  // the scale's tip too: on a phone upright (tab mode) the size line goes and this is where the sizes are
  setTip($("scaleInfo"), tip);
}
function drawOverlay() {
  const rect = $("rect");
  const b = drag ? dragBox(drag) : pending;
  if (!b) {
    rect.hidden = true;
  } else {
    const c = { x: b.x * D, y: b.y * D, w: b.w * D, h: b.h * D };
    Object.assign(rect.style, { left: `${c.x}px`, top: `${c.y}px`, width: `${c.w}px`, height: `${c.h}px` });
    rect.hidden = false;
  }
  // the handle a pointer holds (its edge or corner; a move holds none)
  const held = drag !== null && drag.mode !== "new" && drag.mode !== "move" ? drag.mode : "";
  for (const h of rect.querySelectorAll(".hd")) {
    h.classList.toggle("held", [...h.classList].includes(held));
  }
  drawBakeInfo();
  applyState(); // every change of the crop rect (and of the zoom, via layout) passes here
}
function setCropMode(on) {
  cropMode = on;
  $("stage").classList.toggle("cropping", on);
  applyState();
}
$("cropApplyBtn").onclick = () => applyCrop();
$("cropCancelBtn").onclick = () => dropCrop();
// Cancel / Esc: the rectangle gone — a crop taken off for it (Auto on a cropped image) back
function dropCrop() {
  cancelCrop();
  if (uncropped !== null) {
    S.ops = uncropped;
    uncropped = null;
    refresh();
  }
}
function cancelCrop() {
  pending = null;
  drag = null;
  drawOverlay();
  setCropMode(false);
}
function applyCrop() {
  if (!pending) return;
  const op = { op: "crop", ...pending };
  cancelCrop();
  uncropped = null;
  pushOp(op);
}

// pointers on the pane: mouse / pen / one finger draw the crop in the crop mode, else pan the image; two fingers
// pinch-zoom about their midpoint and pan (Pointer Events — iPadOS Safari, Android Chrome); a double click / double tap = fit.
// A crop and a zoom never mix: once a rectangle is under way (the finger moved PINCH_GRACE_PX or held PINCH_GRACE_MS)
// and GESTURE_QUIET_MS after it, another finger, a pinch or a double tap leave the view as it is — a second finger
// sooner is a pinch (the rectangle dropped); after a pinch a finger landing again within that time draws nothing.
// A finger with a rectangle up: on a handle (corner or edge) it adjusts the rectangle — the handle lit while held, a
// second finger then leaves the view as it is; anywhere else it pans and two fingers pinch (the mouse: inside moves,
// outside draws anew)
const GESTURE_QUIET_MS = 500, PINCH_GRACE_MS = 250, PINCH_GRACE_PX = 10;
(() => {
  const pane = $("imgPane"), stage = $("stage"), img = $("img");
  const touches = new Map(); // pointerId → client point
  let pinch = null;          // { dist0, zoom0, mid }
  let lastTap = null;        // { t, x, y } of the last short tap
  let pan = null;            // { id, x, y, left, top, t } — a drag outside the crop mode
  let cropEnd = -Infinity, pinchEnd = -Infinity; // when the last crop drag / pinch ended
  const cropUnderWay = () => drag !== null && (performance.now() - drag.t > PINCH_GRACE_MS || Math.hypot(drag.cx - drag.x, drag.cy - drag.y) > PINCH_GRACE_PX);
  // a read locks the view too: no zoom, no pan (see applyState)
  const zoomLocked = () => isBusy() || cropUnderWay() || performance.now() - cropEnd < GESTURE_QUIET_MS;
  // a short tap: two in a row = fit (touch / pen; the mouse has dblclick)
  const tap = (ev, t0) => {
    if (ev.pointerType === "mouse" || performance.now() - t0 >= 300) return;
    const now = performance.now();
    if (lastTap && now - lastTap.t < 350 && Math.hypot(ev.clientX - lastTap.x, ev.clientY - lastTap.y) < 30) {
      lastTap = null;
      zoomTo(1, paneCentre());
    } else {
      lastTap = { t: now, x: ev.clientX, y: ev.clientY };
    }
  };
  const imgPos = (ev) => {
    const r = img.getBoundingClientRect();
    return { x: Math.min(Math.max((ev.clientX - r.left) / D, 0), S.prev.w), y: Math.min(Math.max((ev.clientY - r.top) / D, 0), S.prev.h) };
  };
  const two = () => {
    const [a, b] = [...touches.values()];
    return { mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, dist: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)) };
  };
  const dropDrag = () => {
    if (!drag) return;
    pending = drag.prev; // the crop proposal before the finger came down stays
    drag = null;
    drawOverlay();
  };
  pane.addEventListener("pointerdown", (ev) => {
    if (!S.prev || stage.hidden || isBusy()) return;
    if (ev.pointerType === "touch") {
      if (ev.isPrimary) {
        // a fresh first finger: no finger is down — one whose end the pane missed (it lifted outside) is gone
        touches.clear();
        pinch = null;
      }
      touches.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
      if (touches.size >= 2) {
        if (zoomLocked()) {
          return; // a second finger while cropping: the rectangle goes on, the view stays
        }
        dropDrag();
        pan = null;
        if (touches.size === 2) {
          const t = two();
          pinch = { dist0: t.dist, zoom0: zoom, mid: t.mid };
        }
        return;
      }
    }
    // the crop mode takes the whole pane (a rectangle may begin beside the image — kept on it); a pan only the image
    if (ev.button !== 0 || pinch || (!cropMode && !stage.contains(ev.target))) return;
    ev.preventDefault();
    pane.setPointerCapture(ev.pointerId);
    if (!cropMode) {
      pan = { id: ev.pointerId, x: ev.clientX, y: ev.clientY, left: pane.scrollLeft, top: pane.scrollTop, t: performance.now() };
      return;
    }
    if (performance.now() - pinchEnd < GESTURE_QUIET_MS) {
      return; // a finger of the pinch just ended landing again: no rectangle
    }
    const p = imgPos(ev);
    const touch = ev.pointerType === "touch";
    const grab = pending === null ? null : grabOf(pending, p, touch ? GRAB_TOUCH_PX : GRAB_PX);
    if (touch && pending !== null && (grab === null || grab === "move")) {
      // no handle under the finger: a pan (a pinch when a second finger follows)
      pan = { id: ev.pointerId, x: ev.clientX, y: ev.clientY, left: pane.scrollLeft, top: pane.scrollTop, t: performance.now() };
      return;
    }
    drag = { id: ev.pointerId, mode: grab === null ? "new" : grab, base: pending, start: p, cur: p, prev: pending, t: performance.now(), x: ev.clientX, y: ev.clientY, cx: ev.clientX, cy: ev.clientY };
    pending = null;
    $("rect").classList.remove("auto");
    drawOverlay();
  });
  // a mouse / pen over the rectangle (crop mode): the cursor of what a press does there — resize at an edge or corner,
  // move inside, draw anew elsewhere (the crop mode's crosshair)
  const GRAB_CURSOR = { l: "ew-resize", r: "ew-resize", t: "ns-resize", b: "ns-resize", lt: "nwse-resize", rb: "nwse-resize", lb: "nesw-resize", rt: "nesw-resize", move: "move" };
  pane.addEventListener("pointermove", (ev) => {
    if (ev.pointerType !== "touch" && drag === null && pan === null && S.prev) {
      const g = cropMode && pending !== null ? grabOf(pending, imgPos(ev), GRAB_PX) : null;
      stage.style.cursor = g === null ? "" : GRAB_CURSOR[g];
    }
    if (touches.has(ev.pointerId)) touches.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    if (pinch && touches.size === 2) {
      const t = two();
      zoomTo(pinch.zoom0 * (t.dist / pinch.dist0), pinch.mid, t.mid);
      pinch.mid = t.mid;
      return;
    }
    if (pan && ev.pointerId === pan.id) {
      pane.scrollLeft = pan.left - (ev.clientX - pan.x);
      pane.scrollTop = pan.top - (ev.clientY - pan.y);
      return;
    }
    if (drag && ev.pointerId === drag.id) {
      drag.cx = ev.clientX;
      drag.cy = ev.clientY;
      drag.cur = imgPos(ev);
      drawOverlay();
    }
  });
  const end = (ev, cancelled) => {
    touches.delete(ev.pointerId);
    if (pinch && touches.size < 2) {
      pinch = null;
      pinchEnd = performance.now();
    }
    if (pan && ev.pointerId === pan.id) {
      const p = pan;
      pan = null;
      if (!cancelled && Math.hypot(ev.clientX - p.x, ev.clientY - p.y) < 8) tap(ev, p.t);
      return;
    }
    if (!drag || ev.pointerId !== drag.id) return;
    cropEnd = performance.now();
    if (cancelled) { dropDrag(); return; }
    drag.cur = imgPos(ev);
    const d = drag, b = dragBox(d);
    drag = null;
    if (d.mode !== "new" && (b.w * D < 8 || b.h * D < 8)) {
      pending = d.base; // squeezed to nothing: the rectangle as it was
      drawOverlay();
      return;
    }
    if (b.w * D < 8 || b.h * D < 8) {
      // a tap in the crop mode drops the rect drawn so far, the mode stays (no double tap = fit: the view stays)
      pending = null;
      drawOverlay();
      return;
    }
    const x0 = Math.round(b.x), y0 = Math.round(b.y);
    const x1 = Math.min(S.prev.w, Math.round(b.x + b.w)), y1 = Math.min(S.prev.h, Math.round(b.y + b.h));
    pending = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
    drawOverlay();
  };
  pane.addEventListener("pointerup", (ev) => end(ev, false));
  pane.addEventListener("pointercancel", (ev) => end(ev, true));
  pane.addEventListener("dblclick", (ev) => { if (!cropMode && !isBusy() && stage.contains(ev.target)) zoomTo(1, paneCentre()); });
  // ⌘ / ctrl + wheel and the trackpad pinch (Chrome / Edge send it as a ctrl + wheel) zoom about the pointer;
  // a plain wheel scrolls the pane
  pane.addEventListener("wheel", (ev) => {
    if (!S.prev || stage.hidden || !(ev.ctrlKey || ev.metaKey)) return;
    ev.preventDefault();
    if (zoomLocked()) return;
    const dy = ev.deltaMode === 1 ? ev.deltaY * 16 : ev.deltaY;
    const p = { x: ev.clientX, y: ev.clientY };
    zoomTo(zoom * ZOOM_STEP ** (-dy / 100), p);
  }, { passive: false });
  // Safari's trackpad pinch (macOS) comes as gesture events, not ctrl + wheel; on iPadOS they come WITH the touches
  // (handled above) — there only the page zoom is stopped
  let gesture0 = 1;
  pane.addEventListener("gesturestart", (ev) => { ev.preventDefault(); gesture0 = zoom; });
  pane.addEventListener("gesturechange", (ev) => {
    ev.preventDefault();
    if (touches.size || zoomLocked()) return;
    zoomTo(gesture0 * ev.scale, { x: ev.clientX, y: ev.clientY });
  });
  pane.addEventListener("gestureend", (ev) => ev.preventDefault());
})();
// Auto: a rectangle (dashed) in the crop mode — its edges adjust it, ✓ applies it: the text crop (the page's main
// text block, ocr-layout crop.rs), the whole image when no writing is found. On a cropped image it starts from the
// uncropped one (its other steps kept)
function showPending(r) {
  drag = null;
  pending = { ...r };
  $("rect").classList.add("auto");
  setCropMode(true);
  drawOverlay();
}
async function uncrop() {
  if (!S.ops.some((o) => o.op === "crop")) {
    return;
  }
  uncropped = S.ops;
  S.ops = S.ops.filter((o) => o.op !== "crop");
  await refresh();
}
// the text crop's mode (▾ beside Auto): Auto, or a side of a book spread — Left, Right, Double (both pages); the button
// names it and repeats it; kept for the next images of this page (a new page starts at Auto)
let cropSide = "auto";
// this image is a book spread (Auto found writing on both sides of a spine): Auto is no choice for it — the ▾ menu
// opens by itself with the question and Auto greyed out (most people never read the status line)
let spreadSeen = false;
function drawCropSide() {
  const b = $("textCropBtn");
  b.dataset.i18n = `crop.side.${cropSide}`;
  b.textContent = t(b.dataset.i18n);
  b.dataset.i18nTip = cropSide === "auto" ? "crop.auto.title" : `crop.side.${cropSide}.tip`;
  setTip(b, t(b.dataset.i18nTip));
  for (const m of $("sideMenu").querySelectorAll("button")) {
    m.classList.toggle("on", m.dataset.side === cropSide);
    m.setAttribute("aria-checked", String(m.dataset.side === cropSide));
    m.disabled = spreadSeen && m.dataset.side === "auto";
  }
  $("sideMenuHead").hidden = !spreadSeen;
  applyState(); // the button's own tip back under a why
}
$("sideBtn").onclick = (e) => {
  e.stopPropagation();
  $("sideMenu").hidden = !$("sideMenu").hidden;
};
document.addEventListener("click", () => { $("sideMenu").hidden = true; });
for (const m of $("sideMenu").querySelectorAll("button")) {
  m.onclick = () => {
    $("sideMenu").hidden = true;
    cropSide = m.dataset.side;
    crumb(`crop side ${cropSide}`);
    drawCropSide();
    $("textCropBtn").click();
  };
}
// Auto: the text crop in the chosen mode — by itself when an image opens (and after Back to the original), and on a
// press (again after ✕, or on a cropped image: from the uncropped one)
async function textCrop() {
  await uncrop();
  status(t("st.finding"), "busy");
  S.finding = true;
  applyState();
  let r;
  try {
    r = await call("text_crop_side", { ops: S.ops, side: cropSide });
  } finally {
    S.finding = false;
    applyState();
  }
  if (r.none !== undefined) {
    // no writing found: the whole image to adjust by hand
    showPending({ x: 0, y: 0, w: S.prev.w, h: S.prev.h });
    crumb(`text crop: ${r.none}`);
    status(t("st.cropNone", { why: noneWhy(r) }), "ok");
    return;
  }
  showPending(r);
  if (cropSide === "auto" && r.spread) {
    spreadSeen = true;
    drawCropSide();
    $("sideMenu").hidden = false;
  }
  // a side: the page gutter found (the crop is that page / both), or not (the crop is Auto's whole text block); Auto on
  // a book spread (writing on both sides of a spine): the hint to pick a page in the ▾ menu
  const side = cropSide !== "auto" ? t(r.gutter === null ? "st.noGutter" : "st.gutterFound") : r.spread ? t("st.spread") : "";
  status(`${t("st.cropFound")}${side}`, "ok");
}
$("textCropBtn").onclick = () => textCrop();

document.addEventListener("keydown", (e) => {
  if ((!pending && !cropMode) || document.activeElement === $("editor") || document.querySelector("dialog[open]")) return;
  if (e.key === "Enter") { e.preventDefault(); applyCrop(); }
  if (e.key === "Escape") { e.preventDefault(); dropCrop(); }
});

// ── text: rendered editor + markup view ───────────────────────────────
const renderHtml = (text) => invoke("render", { text });
const FMT_NAMED = new Set(["del", "ins", "u", "sup", "sub"]);
function buildFormatBar() {
  const sample = { del: "<del>S</del>", ins: "<ins>A</ins>", u: "<u>U</u>", sup: "x<sup>2</sup>", sub: "x<sub>2</sub>" };
  for (const p of S.marks.pairs) {
    const b = document.createElement("button");
    b.className = "ib fmt";
    b.dataset.tag = p.tag;
    b.innerHTML = sample[p.tag] ?? p.tag;
    b.addEventListener("mousedown", (e) => e.preventDefault()); // keep the editor's selection
    b.onclick = () => S.editor.toggle(p.tag).then(applyState);
    $("fmtSeg").append(b);
  }
  titleFormatBar();
}
// the format buttons' tips in the current language
function titleFormatBar() {
  for (const p of S.marks.pairs) {
    const b = $("fmtSeg").querySelector(`[data-tag="${p.tag}"]`);
    setTip(b, t("fmt.title", { name: FMT_NAMED.has(p.tag) ? t(`fmt.${p.tag}`) : p.tag, open: p.open, close: p.close }));
  }
}
for (const id of ["gapBtn", "tabBtn", "undoBtn", "redoBtn"]) $(id).addEventListener("mousedown", (e) => e.preventDefault());
$("gapBtn").onclick = () => S.editor.insert(S.marks.gap).then(applyState);
$("tabBtn").onclick = () => S.editor.insert(S.marks.cellSep).then(applyState);
$("undoBtn").onclick = () => S.editor.undo().then(applyState);
$("redoBtn").onclick = () => S.editor.redo().then(applyState);
document.addEventListener("selectionchange", () => {
  if (!S.editor || document.activeElement !== $("editor")) return;
  const on = new Set(S.editor.tagsAtCaret());
  for (const b of $("fmtSeg").children) b.classList.toggle("on", on.has(b.dataset.tag));
});

// Fit: every line as written (no wrapping), the font scaled down so the widest line or table fits the panel — a view
// setting (remembered in this browser), the text is not changed. Refitted whenever the text or the panel changes. On
// by default (it fits on every device); off only once turned off in this browser
const FIT_KEY = "ocrlite.fit";
const FIT_MIN_PX = 3; // Fit shows every line whole, however small — a pinch (the text zoom) makes it readable
let fitOn = true;
try { fitOn = localStorage.getItem(FIT_KEY) !== "0"; } catch {}
let fitFrame = 0;
// the text zoom: × the fitted (or the normal) size; a pinch / ⌘ + wheel on the text, kept across tabs; Fit resets it
const TEXT_ZOOM = [0.5, 6];
let textZoom = 1;
let srcBase = 0; // the markup view's own font size (px), read once
function fitText() {
  const ed = $("editor");
  if (ed.offsetWidth === 0 && $("src").offsetWidth === 0) {
    return; // the text tab is away: its fit stays for its return
  }
  if (srcBase === 0 && $("src").offsetWidth > 0) {
    srcBase = parseFloat(getComputedStyle($("src")).fontSize);
  }
  if (srcBase > 0) {
    $("src").style.fontSize = `${srcBase * textZoom}px`;
  }
  ed.classList.toggle("fit", fitOn);
  ed.style.fontSize = "";
  if (ed.hidden) {
    return;
  }
  const cs = getComputedStyle(ed);
  const base = parseFloat(cs.fontSize);
  let size = base;
  if (fitOn) {
    const pad = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight);
    const avail = ed.clientWidth - pad - 2; // a little margin: the last glyph clear of the edge
    // the width scales with the font size except borders and cell padding: three passes come close enough
    for (let i = 0; i < 3; i++) {
      const need = ed.scrollWidth - pad;
      if (need <= avail + 0.5) {
        break;
      }
      size = Math.max(FIT_MIN_PX, Math.min(base, (size * avail) / need));
      ed.style.fontSize = `${size}px`;
    }
  }
  if (textZoom !== 1) {
    ed.style.fontSize = `${size * textZoom}px`;
  }
}
/// the text zoom to `z`, the text under the client point `at` kept in place
function zoomText(z, at) {
  const el = markupView ? $("src") : $("editor");
  const z1 = Math.min(TEXT_ZOOM[1], Math.max(TEXT_ZOOM[0], z));
  if (Math.abs(z1 - textZoom) < 0.001) {
    return;
  }
  const r = el.getBoundingClientRect();
  const cx = el.scrollLeft + at.x - r.left, cy = el.scrollTop + at.y - r.top;
  const k = z1 / textZoom;
  textZoom = z1;
  fitText();
  el.scrollLeft = cx * k - (at.x - r.left);
  el.scrollTop = cy * k - (at.y - r.top);
}
// two fingers on the text pinch it (touch events: the browser's own one-finger pan stays; the second finger's moves are
// taken from it), ⌘ / ctrl + wheel zooms it
for (const el of [$("editor"), $("src")]) {
  let pinch = null;
  const two = (e) => {
    const [a, b] = [e.touches[0], e.touches[1]];
    return { d: Math.max(1, Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY)), mid: { x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 } };
  };
  el.addEventListener("touchstart", (e) => {
    if (e.touches.length === 2) {
      pinch = { d0: two(e).d, z0: textZoom };
    }
  }, { passive: true });
  el.addEventListener("touchmove", (e) => {
    if (pinch === null || e.touches.length !== 2) {
      return;
    }
    e.preventDefault();
    const t = two(e);
    zoomText(pinch.z0 * (t.d / pinch.d0), t.mid);
  }, { passive: false });
  const done = (e) => {
    if (e.touches.length < 2) {
      pinch = null;
    }
  };
  el.addEventListener("touchend", done);
  el.addEventListener("touchcancel", done);
  el.addEventListener("wheel", (e) => {
    if (!(e.ctrlKey || e.metaKey)) {
      return;
    }
    e.preventDefault();
    const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    zoomText(textZoom * ZOOM_STEP ** (-dy / 100), { x: e.clientX, y: e.clientY });
  }, { passive: false });
}
function refit() {
  cancelAnimationFrame(fitFrame);
  fitFrame = requestAnimationFrame(fitText);
}
new ResizeObserver(refit).observe($("editor"));
new MutationObserver(refit).observe($("editor"), { childList: true, subtree: true, characterData: true });
$("fitBtn").classList.toggle("on", fitOn);
$("fitBtn").onclick = () => {
  fitOn = !fitOn;
  textZoom = 1; // Fit / not: the zoom starts again from it
  $("fitBtn").classList.toggle("on", fitOn);
  try { localStorage.setItem(FIT_KEY, fitOn ? "1" : "0"); } catch {}
  refit();
};

// a tech preview: the first edit of a session says that edits are not kept (no saved work yet)
const EDIT_WARNED = "ocrlite.editWarned";
function warnEdit() {
  try {
    if (sessionStorage.getItem(EDIT_WARNED)) {
      return;
    }
    sessionStorage.setItem(EDIT_WARNED, "1");
  } catch {}
  alert(t("warn.edit"));
}
$("editor").addEventListener("input", warnEdit);
$("src").addEventListener("input", warnEdit);

let markupView = false;
$("markupBtn").onclick = async () => {
  markupView = !markupView;
  $("markupBtn").classList.toggle("on", markupView);
  if (markupView) {
    $("src").value = S.editor.markup;
  } else if ($("src").value !== S.editor.markup) {
    await S.editor.load($("src").value); // an edit in the markup view starts a new history base
  }
  $("src").hidden = !markupView;
  $("editor").hidden = markupView;
  for (const id of ["fmtSeg", "gapBtn", "tabBtn", "undoBtn", "redoBtn"]) $(id).closest(".seg").style.visibility = markupView ? "hidden" : "";
  applyState();
};
$("src").addEventListener("keydown", (e) => {
  if (e.key === "Tab" && !e.metaKey && !e.ctrlKey && !e.altKey) {
    e.preventDefault();
    $("src").setRangeText("\t", $("src").selectionStart, $("src").selectionEnd, "end");
  }
});
const currentMarkup = () => (markupView ? $("src").value : S.editor.markup);
$("src").addEventListener("input", () => applyState()); // text present / not, in the markup view

// ── read ──────────────────────────────────────────────────────────────
function setReading(on) {
  S.reading = on;
  S.stopping = false;
  crumb(on ? "read start" : "read end");
  S.editor.setEditable(!on);
  $("src").readOnly = on;
  if (!on) scan.stop();
  applyState();
  if (!on) {
    restartRead();
  }
}
let streamTimer = null;
function showStream() {
  if (streamTimer) return;
  streamTimer = setTimeout(() => {
    streamTimer = null;
    if (markupView) {
      $("src").value = S.streamed;
      $("src").scrollTop = $("src").scrollHeight;
    } else {
      S.editor.show(S.streamed).then(() => { $("editor").scrollTop = $("editor").scrollHeight; });
    }
  }, 120);
}
// iOS freezes a page in the background: a read there stalls, comes back scrambled or loses its GPU device (pausing it
// did not work), and closes background apps holding much memory. When the app goes away the engine is released
// (host.releaseEngine: its worker ended — GPU device and memory pool, about 1.8 GB, given back; a new worker opens the
// engine at the next read); a read running then ends and starts again from the beginning when the app is back.
// No single event tells it on every browser — several signals feed wentAway / cameBack:
//   visibilitychange   hidden / visible (the main one: app switch, home screen, lock)
//   pagehide/pageshow  the page put into / back from the back-forward cache
//   freeze / resume    Chromium's page lifecycle (a frozen background tab)
//   a late timer tick  the page was frozen without any of these (seen with iOS home-screen apps): wall-clock time
//                      jumped by more than AWAY_GAP_MS between two ticks of a 1 s timer
// blur / focus are not used: on a desktop they fire on every switch to another window.
let stoppedHidden = false;
let away = false;
let restartOnReturn = false;
const AWAY_GAP_MS = 5000;
let lastTick = Date.now();
function stopRead() {
  S.stopping = true;
  applyState();
  invoke("cancel");
}
function wentAway(why) {
  if (away) {
    return;
  }
  away = true;
  lastTick = Date.now();
  try { sessionStorage.setItem("ocrlite.away", "1"); } catch {}
  host.awayFromApp();
  if (S.reading || S.starting) {
    restartOnReturn = true;
    stoppedHidden = true;
  }
  // the read (if one runs) fails with "released" → ocr:error → setReading(false); restartRead waits for the return
  const released = host.releaseEngine();
  crumb(`background (${why})${released ? ": engine released" : ": engine kept (the worker is busy)"}`);
  if (!released && S.reading && !S.stopping) {
    stopRead();
  }
}
function cameBack(why) {
  if (!away) {
    return;
  }
  away = false;
  // the gap behind an event's return is not counted again by the timer
  lastTick = Date.now();
  try { sessionStorage.removeItem("ocrlite.away"); } catch {}
  crumb(`foreground (${why})`);
  restartRead();
}
// the read again once the stopped one has ended and the app is in the foreground
function restartRead() {
  if (!restartOnReturn || away || S.reading || S.starting || document.visibilityState === "hidden") {
    return;
  }
  restartOnReturn = false;
  crumb("read restarted after the background");
  setTimeout(() => $("readBtn").click(), 0);
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") {
    wentAway("hidden");
  } else {
    cameBack("visible");
  }
});
window.addEventListener("pagehide", () => wentAway("pagehide"));
window.addEventListener("pageshow", () => cameBack("pageshow"));
document.addEventListener("freeze", () => wentAway("freeze"));
document.addEventListener("resume", () => cameBack("resume"));
setInterval(() => {
  const now = Date.now();
  if (now - lastTick > AWAY_GAP_MS && !away) {
    wentAway(`frozen ${Math.round((now - lastTick) / 1000)} s`);
    cameBack("timer");
  }
  lastTick = now;
}, 1000);
$("readBtn").onclick = async () => {
  if (S.reading) {
    stopRead();
    return;
  }
  if (!S.prev) {
    throw new Error("Read without an image (applyState keeps it disabled then)");
  }
  // Read stays disabled while the checks run (a second click must not slip in before the read starts)
  S.starting = true;
  applyState();
  let go = false;
  try {
    go = await readChecks();
  } finally {
    if (!go) {
      S.starting = false;
      applyState();
    }
  }
  if (!go) {
    return;
  }
  S.streamed = "";
  stoppedHidden = false;
  // tab mode: the read shows on the image (its scan); the Text tab comes back with the first text (ocr:text)
  setTab("image");
  await S.editor.show("");
  $("src").value = "";
  S.starting = false; // straight into reading: Read never shows enabled in between
  // the image back to fit: a zoomed image's large layer is given back before the read takes its GPU memory
  if (zoom > 1) {
    crumb("zoom to fit for the read");
    zoomTo(1, paneCentre());
  }
  setReading(true);
  status(t("st.starting"), "busy");
  try {
    S.readBake = null;
    // the input's width for the lines' measure after it (unpadded, as drawBakeInfo shows it)
    const b = S.prev.bake;
    S.readInputW = S.size.factor === 1 ? b.scaledW : Math.max(1, Math.round(b.scaledW * S.size.factor));
    await call("read", { ops: S.ops, preset: readPreset(), factor: S.size.factor });
  } catch { setReading(false); }
};
// the checks before a read → true: it may start; false: refused, the status says why
async function readChecks() {
  // without the model in this browser's storage: refused with the way to the setup (never a failing read) — Read is
  // disabled then already (checkModel); this asks again in case the storage changed behind the page's back
  let missing;
  try {
    missing = await host.setup.modelMissing();
  } catch (e) {
    status(t("st.storage", { e: e instanceof Error ? e.message : e }), "err");
    return false;
  }
  if (missing) {
    status(t("st.needModel"), "err", SETUP_LINK);
    await checkModel();
    return false;
  }
  // the host refuses a read on a device flagged unsupported (it ran out of memory before) or a dead GPU engine
  const blocked = host.blockedNote();
  if (blocked) {
    status(blocked.text, blocked.kind, blocked.link);
    return false;
  }
  // the model in this browser must come from an allowed source (bridge-web.js modelGate)
  let gate;
  try {
    gate = await host.modelGate();
  } catch (e) {
    status(t("st.checkFailed", { e: e instanceof Error ? e.message : e }), "err");
    return false;
  }
  if (!gate.ok) {
    status(gate.note.text, gate.note.kind, gate.note.link);
    await checkModel();
    return false;
  }
  return true;
}
listen("ocr:status", (e) => status(e.payload, "busy"));
listen("ocr:result", (e) => status(e.payload.text, e.payload.ok ? "ok" : "err"));
// the read's phase times so far (ocr-decode Phase / ReadStats): "load 0.9 s · image 6.5 s · prefill 1.5 s"; loadMs =
// the browser engine's weight load for this read (it frees the weights after each read); without encodeMs the engine
// (llama.cpp) encodes the image inside its prefill
const secs = (ms) => `${fixed(ms / 1000, 1)} s`;
function phaseTimes(r) {
  const out = [];
  if (r.loadMs != null) out.push(t("phase.load", { t: secs(r.loadMs) }));
  if (r.encodeMs != null) out.push(t("phase.image", { t: secs(r.encodeMs) }));
  if (r.prefillMs != null) out.push(`${t("phase.prefill", { t: secs(r.prefillMs) })}${r.encodeMs == null ? t("phase.withImage") : ""}`);
  return out;
}
const PHASES = new Set(["encodingImage", "loadingText", "readingPrompt", "decoding"]);
const scan = scanOverlay($("stage"), $("img"));
listen("ocr:phase", (e) => scan.phase(e.payload, S.readBake));
// a tap on the model size line: its detail (the tooltip) in the status line — a touch screen shows no tooltip
$("bakeInfo").onclick = () => {
  if ($("bakeInfo").dataset.tip !== undefined && $("bakeInfo").dataset.tip !== "") {
    status($("bakeInfo").dataset.tip, "ok");
  }
};
// the read's model input: its plan (the scan sweeps it) — the status already shows its size (drawBakeInfo)
listen("ocr:input", (e) => {
  const { plan, w, h, factor } = e.payload;
  S.readBake = plan;
  crumb(`input ${w}×${h}${factor === 1 ? "" : ` ×${factor.toFixed(2)}`}`);
});
listen("ocr:phase", (e) => crumb(e.payload.phase)); // read start … encodingImage = the vision weights' load
listen("ocr:phase", (e) => status([PHASES.has(e.payload.phase) ? t(`phase.${e.payload.phase}`) : e.payload.phase, ...phaseTimes(e.payload)].join(" · "), "busy"));
listen("ocr:text", (e) => {
  if (S.streamed === "") {
    setTab("text"); // the text begins: it streams in view (stacked: the Text tab)
  }
  S.streamed += e.payload;
  showStream();
});
listen("ocr:done", async (e) => {
  clearTimeout(streamTimer);
  streamTimer = null;
  await S.editor.load(S.streamed);
  if (markupView) $("src").value = S.streamed;
  setReading(false);
  // the text gets the focus: keys and the wheel scroll it at once — with a mouse / trackpad only (a touch screen would
  // open its keyboard; a finger scrolls the text without the focus)
  if (matchMedia("(pointer: fine)").matches) {
    $("editor").focus({ preventScroll: true });
  }
  crumb(`memory ${await host.memoryNote()}`);
  const r = e.payload;
  const tps = r.decodeMs > 0 ? (r.outTokens / (r.decodeMs / 1000)).toFixed(0) : "–";
  const end = r.cancelled ? t(stoppedHidden ? "done.stoppedAway" : "done.stopped") : r.stoppedOnEog ? "" : t("done.limit");
  const total = (r.loadMs ?? 0) + (r.encodeMs ?? 0) + r.prefillMs + r.decodeMs;
  // tokenChoice (the wgpu engine): "gpu" = greedy, chosen on the GPU and run ahead; "cpu" = sampled (llama.cpp: absent)
  const choice = r.tokenChoice ? [t("done.choice", { c: r.tokenChoice })] : [];
  const lines = r.cancelled || !r.stoppedOnEog ? "" : lineSuggest(S.streamed, S.readInputW);
  status(`${[...phaseTimes(r), t("done.tokens", { n: r.outTokens.toLocaleString(getLang()) }), t("done.tps", { n: tps }), ...choice, t("done.total", { t: secs(total) })].join(" · ")}${end}${lines}`,
    !r.stoppedOnEog && !r.cancelled ? "err" : "ok");
  applyState();
});
listen("ocr:error", (e) => {
  // the read ended by the engine's release in the background: no error — it starts again on return; a Stop before the
  // text (the engine released at once, host cancel): stopped, no text
  const released = restartOnReturn && away;
  const stopped = S.stopping && String(e.payload) === "stopped";
  setReading(false);
  if (released) {
    status(t("st.stoppedAway"), "ok");
  } else if (stopped) {
    status(t("st.stoppedEarly"), "ok");
  } else {
    status(String(e.payload), "err");
  }
});

// ── copy / save ───────────────────────────────────────────────────────
$("copyBtn").onclick = () => navigator.clipboard.writeText(currentMarkup()).then(() => status(t("st.copied"), "ok"));
$("saveBtn").onclick = (e) => { e.stopPropagation(); $("saveMenu").hidden = !$("saveMenu").hidden; };
document.addEventListener("click", () => { $("saveMenu").hidden = true; });
for (const b of $("saveMenu").querySelectorAll("button")) {
  b.onclick = async () => {
    $("saveMenu").hidden = true;
    const ext = b.dataset.save;
    if (ext === "png" && !S.prev) { status(t("why.openFirst"), "err"); return; }
    let p;
    try {
      p = await host.saveAs(ext, { stem: S.stem, ops: S.ops, factor: S.size.factor, text: currentMarkup() });
    } catch (e) {
      status(String(e), "err");
      return;
    }
    if (p) status(t("st.saved", { p }), "ok");
  };
}

// ── settings + presets ────────────────────────────────────────────────
// expert mode (Settings, kept in this browser, applied at once): the presets (the list beside Read, their editor) and
// the technical details (.expert-only, style.css). Off: every read uses the default preset — a preset chosen in expert
// mode is kept for when it is on again
const EXPERT_KEY = "ocrlite.expert";
let expert = false;
try { expert = localStorage.getItem(EXPERT_KEY) === "1"; } catch {}
document.body.classList.toggle("expert", expert);
$("expertChk").checked = expert;
const readPreset = () => (expert ? $("presetSel").value : S.presets.defaultName);
$("expertChk").onchange = () => {
  expert = $("expertChk").checked;
  try { localStorage.setItem(EXPERT_KEY, expert ? "1" : "0"); } catch {}
  document.body.classList.toggle("expert", expert);
  crumb(`expert ${expert ? "on" : "off"}`);
  applyState(); // Read's tip names the preset
};
// the preset editor: the decoder settings' names are technical terms — English in every language; the rest is translated
const FIELDS = [
  ["name", "text", "Name"], ["temperature", "number", "Temp"], ["topP", "number", "Top p"], ["repetitionPenalty", "number", "Rep. pen."],
  ["repetitionWindow", "number", "Window"], ["frequencyPenalty", "number", "Freq. pen."], ["maxNewTokens", "number", "Max tokens"],
  ["gtTokenCapFactor", "number", "GT cap ×"],
];
let draftUser = [];
// the presets differ from the saved ones: Save presets is offered
let presetsDirty = false;
function markPresets(dirty) {
  presetsDirty = dirty;
  $("presetSave").disabled = !dirty;
}
function fillPresetSel() {
  const sel = $("presetSel");
  sel.innerHTML = "";
  for (const p of [...S.presets.shipped, ...S.presets.user]) {
    const o = document.createElement("option");
    o.value = o.textContent = p.name;
    sel.append(o);
  }
  sel.value = S.settings.preset;
}
$("presetSel").onchange = async () => {
  const next = { ...S.settings, preset: $("presetSel").value };
  try { await call("save_settings", { settings: next }); S.settings = next; } catch { $("presetSel").value = S.settings.preset; }
  applyState(); // Read's tip names the preset
};
function presetRow(p, editable, idx) {
  const tr = document.createElement("tr");
  if (!editable) tr.className = "shipped";
  for (const [k, type, label] of FIELDS) {
    const td = document.createElement("td"), inp = document.createElement("input");
    // the column's label on the cell: a narrow screen shows each preset as a card (style.css), label beside value
    td.dataset.label = label;
    td.className = `f-${k}`;
    inp.type = type;
    if (type === "number") inp.step = "any";
    inp.value = p[k];
    inp.readOnly = !editable;
    if (!editable) setTip(inp, t("preset.readonly"));
    if (k === "name") inp.style.width = `${nameCh}ch`;
    if (editable) {
      inp.oninput = () => {
        draftUser[idx][k] = type === "number" ? Number(inp.value) : inp.value;
        markPresets(true);
      };
    }
    td.append(inp);
    tr.append(td);
  }
  const td = document.createElement("td"), b = document.createElement("button");
  td.className = "act";
  b.type = "button";
  b.className = "btn small";
  if (editable) {
    b.textContent = t("preset.delete");
    b.onclick = () => {
      draftUser.splice(idx, 1);
      markPresets(true);
      drawPresets();
    };
  } else {
    b.textContent = t("preset.duplicate");
    b.onclick = () => {
      const names = new Set([...S.presets.shipped, ...draftUser].map((x) => x.name));
      let n = `${p.name}-mine`, k = 2;
      while (names.has(n)) n = `${p.name}-mine${k++}`;
      draftUser.push({ ...p, name: n });
      markPresets(true);
      drawPresets();
    };
  }
  td.append(b);
  tr.append(td);
  return tr;
}
// the name column fits the longest preset name (else names are cut off); numbers stay narrow
let nameCh = 12;
function drawPresets() {
  const table = $("presetTable");
  nameCh = Math.max(12, ...[...S.presets.shipped, ...draftUser].map((p) => p.name.length)) + 3;
  const head = document.createElement("tr");
  for (const [, , label] of FIELDS) {
    head.append(Object.assign(document.createElement("th"), { textContent: label }));
  }
  head.append(document.createElement("th"));
  table.replaceChildren(head);
  S.presets.shipped.forEach((p, i) => table.append(presetRow(p, false, i)));
  draftUser.forEach((p, i) => table.append(presetRow(p, true, i)));
}
$("settingsBtn").onclick = () => {
  host.settingsOpen(S.settings);
  draftUser = S.presets.user.map((p) => ({ ...p }));
  markPresets(false);
  drawPresets();
  $("settingsErr").textContent = "";
  $("langSel").value = getChoice();
  $("expertChk").checked = expert;
  $("settingsDlg").showModal();
};
$("settingsCancel").onclick = () => $("settingsDlg").close();
// the cross or Escape: the other settings (the model URL) are kept; unsaved preset edits are dropped
$("settingsDlg").addEventListener("close", async () => {
  const next = host.settingsCollect(S.settings);
  if (JSON.stringify(next) === JSON.stringify(S.settings)) {
    return;
  }
  try {
    await invoke("save_settings", { settings: next });
    S.settings = next;
    $("backendInfo").textContent = await host.engineLabel(S.backend, S.settings);
  } catch (e) {
    status(String(e), "err");
  }
});
$("presetSave").onclick = async () => {
  try {
    S.presets = await invoke("save_user_presets", { user: draftUser });
    fillPresetSel();
    markPresets(false);
    $("settingsErr").textContent = "";
  } catch (e) {
    $("settingsErr").textContent = String(e);
  }
};

// ── about: version / commit / build time (which build is running) ──
async function showAbout() {
  const b = await host.buildInfo();
  $("aboutVersion").textContent = b.version;
  $("aboutGit").textContent = b.git.endsWith("-dirty") ? t("about.dirty", { git: b.git.slice(0, -6) }) : b.git;
  $("aboutBuilt").textContent = b.built ? new Date(b.built * 1000).toLocaleString() : "—";
  $("aboutEngine").textContent = $("backendInfo").textContent;
  $("aboutDlg").showModal();
}
$("aboutBtn").onclick = showAbout;

// ── first-run setup + help ────────────────────────────────────────────
const setup = setupDialog(host.setup, { pickImage, openHelp: () => $("helpDlg").showModal(), paintIcons });
// Help: how to install, per platform — the setup's sheets, all of them, unfolded
function drawHelpInstall() {
  $("helpInstall").replaceChildren(...INSTALL_PLATFORMS.map((id) => installSheet(id, { help: true })));
  paintIcons($("helpInstall"));
}
drawHelpInstall();
window.addEventListener("ocrlite:lang", drawHelpInstall);
const SETUP_LINK = { get label() { return t("st.setupLink"); }, run: () => { setup.open(); return null; } };
$("helpBtn").onclick = () => $("helpDlg").showModal();
// How to: the steps of the empty pane (cloned at start) + tips; its button opens the full Help
$("howBtn").onclick = () => $("howDlg").showModal();
$("howHelpBtn").onclick = () => {
  $("howDlg").close();
  $("helpDlg").showModal();
};
$("helpSetupBtn").onclick = () => {
  $("helpDlg").close();
  setup.open();
};
$("helpCheckBtn").onclick = () => {
  $("helpDlg").close();
  setup.check();
};
// the licence links: a new tab
for (const a of $("aboutDlg").querySelectorAll("a.ext")) {
  Object.assign(a, { target: "_blank", rel: "noopener" });
}

// ── control states: every control's state from one table, so it reflects what can be done when ──────────────────
// ONE place: applyState() sets every control's enabled / disabled (+ the why in its tip, the Read button's label)
// from the state; every state change calls it — no control is toggled anywhere else.
//
// States: no image · image (S.prev) · crop mode (Draw crop) · pending (a crop rect waiting for Crop / Cancel — drawn
// or found) · finding (Find crop runs) · busy = starting (Read's checks: model, its source) | reading | stopping (Stop
// pressed, the engine not done yet) · no model (browser: not in this browser's storage, no WebGPU; "checking" until
// the start's check answered) · no text / text · markup view · dialogs: modal (showModal) — the page behind is inert,
// nothing changes here.
//
//   control                        enabled when                                   disabled → tip says
//   Open (+ drop, the empty pane)  not busy                                       busy: "Wait for the read to end"
//   rotate ×3, contrast, pencil    image · no crop mode / pending (before a crop  "Open an image first" · "Finish the
//                                  is applied: also with Auto's rectangle up —    crop first (✓ / ✕)" · busy ·
//                                  Auto runs again after them) · not finding ·    "Finding the crop…"
//                                  not busy
//   Auto, Auto's ▾ (side)          image · not finding · not busy                 "Open an image first" · busy ·
//                                                                                 "Finding the crop…"
//   ✕ cancel crop                  a rectangle shows (or is being drawn)          "Nothing to cancel …"
//   ✓ apply crop                   a rectangle shows                              "Nothing to crop …"
//   image undo, reset              a step to undo · as the steps                  as the steps; no step: its own tip
//   preset select (expert mode)    not busy                                       busy (the running read keeps its own)
//   Help, How to                   always
//   About, Fit                     not busy (their re-layout during a read ended in crashes on the iPhone)
//   Settings                       not busy (the model store must not change      busy
//                                  under a read)
//   Read                           image · model ready · no crop mode / pending   no model: its why (e.g. "Set up the
//                                  · not finding · not starting · a crop applied  model first") · "Open an image first" ·
//                                  (Auto's rectangle comes up when an image       "Apply or cancel the crop first" ·
//                                  opens: read the crop, not the whole scan)      "Finding the crop…" · "Crop first…" ·
//                                                                                 starting: "Checking before the read…"
//     reading → "Stop" (enabled) · stopping → "Stopping…" (disabled)
//   zoom out, fit                  image · zoom > 100 % · not busy                busy (also no pinch, pan, wheel zoom)
//   zoom in                        image · zoom < 800 % · not busy                busy
//   format, ▨, table cell          the rendered view · not busy                   busy · (hidden in the markup view)
//   undo, redo                     the rendered view · not busy · a step there    (hidden in the markup view)
//   Markup                         not busy                                       busy
//   Copy                           text · not busy                                "Nothing to copy yet" · busy
//   Save ▸                         text or image · not busy
//     .txt, .html                  text · not busy                                "Nothing to save yet — read an image"
//     .png (model input)           image · not busy                               "Open an image first"
const busyWhy = () => t("why.busy");
const isBusy = () => S.starting || S.reading || S.stopping;
// sets a control's state; the why goes into its tip while disabled (tip.js), its own tip (data-tip-own) comes back when
// enabled
function enable(el, on, why = "") {
  if (el.dataset.tipOwn === undefined) {
    el.dataset.tipOwn = el.dataset.tip === undefined ? "" : el.dataset.tip;
  }
  el.disabled = !on;
  el.dataset.tip = !on && why ? why : el.dataset.tipOwn;
}
function applyState() {
  const busy = isBusy();
  const img = S.prev !== null;
  const cropping = cropMode || pending !== null || drag !== null;
  const cropped = S.ops.some((o) => o.op === "crop");
  const editing = S.editor !== null && !markupView;
  const text = S.editor !== null && currentMarkup().trim() !== "";
  const noImg = t("why.openFirst");
  const cropWhy = t("why.cropFirst");
  // the title bar
  enable($("openBtn"), !busy, busyWhy());
  // the turns and filters: also while Auto's rectangle waits as long as no crop is applied (they change what Auto sees;
  // it runs again after them — pushOp)
  const steps = img && (!cropping || !cropped) && !S.finding && !busy;
  const stepsWhy = !img ? noImg : busy ? busyWhy() : cropping ? cropWhy : t("why.finding");
  for (const b of document.querySelectorAll("[data-op]")) {
    enable(b, steps, stepsWhy);
    if (FILTERS.has(b.dataset.op)) {
      const on = S.ops.some((o) => o.op === b.dataset.op);
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    }
  }
  // Auto: with an image, also on a cropped one (it starts from the uncropped image) and over a rectangle
  const crop = img && !S.finding && !busy;
  const cropWhy2 = !img ? noImg : busy ? busyWhy() : t("why.finding");
  enable($("textCropBtn"), crop, cropWhy2);
  enable($("sideBtn"), crop, cropWhy2);
  enable($("imgUndoBtn"), steps && S.ops.length > 0, steps ? "" : stepsWhy);
  enable($("resetBtn"), steps && S.ops.length > 0, steps ? "" : stepsWhy);
  // Cancel / Apply: while a rectangle shows (Apply once it is drawn)
  enable($("cropCancelBtn"), img && cropping && !busy, !img ? noImg : t("why.noCancel"));
  enable($("cropApplyBtn"), img && pending !== null && !busy, !img ? noImg : t("why.noApply"));
  enable($("presetSel"), !busy, busyWhy());
  // the wand: its menu after a read on the crop; lit when the characters are off the training's width, on while a step
  // other than ×1 is in use
  const usable = img && !busy && !cropping && cropped && S.size.cw !== undefined;
  const lit = usable && scaleOff(), scaled = usable && S.size.factor !== 1;
  $("scaleAutoBtn").classList.toggle("suggest", lit);
  $("scaleAutoBtn").classList.toggle("on", scaled && !lit);
  enable($("scaleAutoBtn"), usable, busy ? busyWhy() : t("scale.auto.title"));
  enable($("scaleMenuBtn"), usable, busy ? busyWhy() : t("scale.auto.title"));
  if (!usable) {
    $("scaleMenu").hidden = true;
  }
  enable($("settingsBtn"), !busy, busyWhy());
  drawReadBtn(img, cropping, cropped);
  // Auto stands out while a crop is due (✓ is the green one while its rectangle waits)
  $("textCropBtn").classList.toggle("primary", img && !cropped);
  // the image card
  // the image stays as it is while a read runs (a zoom's re-layout beside the read's GPU memory ended the page on the
  // iPhone)
  enable($("zoomOutBtn"), img && zoom > 1 && !busy, busy ? busyWhy() : "");
  enable($("zoomFitBtn"), img && zoom > 1 && !busy, busy ? busyWhy() : "");
  enable($("zoomInBtn"), img && zoom < zoomMax() - 0.005 && !busy, busy ? busyWhy() : "");
  // the text card
  for (const b of $("fmtSeg").children) {
    enable(b, editing && !busy, busyWhy());
  }
  enable($("gapBtn"), editing && !busy, busyWhy());
  enable($("tabBtn"), editing && !busy, busyWhy());
  enable($("undoBtn"), editing && !busy && S.editor.canUndo(), busy ? busyWhy() : "");
  enable($("redoBtn"), editing && !busy && S.editor.canRedo(), busy ? busyWhy() : "");
  enable($("markupBtn"), S.editor !== null && !busy, busyWhy());
  // Fit and About stay off while a read runs (their re-layout during a read ended in crashes on the iPhone)
  enable($("fitBtn"), S.editor !== null && !markupView && !busy, busy ? busyWhy() : markupView ? t("why.fitMarkup") : "");
  enable($("aboutBtn"), !busy, busyWhy());
  enable($("copyBtn"), text && !busy, busy ? busyWhy() : t("why.noCopy"));
  enable($("saveBtn"), (text || img) && !busy, busy ? busyWhy() : t("why.noSave"));
  for (const b of $("saveMenu").querySelectorAll("button")) {
    if (b.dataset.save === "png") {
      enable(b, img && !busy, busy ? busyWhy() : noImg);
    } else {
      enable(b, text && !busy, busy ? busyWhy() : t("why.noSave"));
    }
  }
}
// a tap on a disabled Read (it lets taps through, style.css): the why in the status line, the control it waits for
// pointed at — a touch screen shows no tooltip
document.addEventListener("click", (e) => {
  const btn = $("readBtn");
  const r = btn.getBoundingClientRect();
  if (!btn.disabled || e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) {
    return;
  }
  status(t("st.readWaits", { why: btn.dataset.tip }), "err");
  const at = pending !== null ? $("cropApplyBtn") : cropMode ? $("cropCancelBtn") : null;
  if (at !== null) {
    at.classList.remove("attn");
    void at.offsetWidth;
    at.classList.add("attn");
  }
});
// the Read button: Read / Stop / Stopping… (the label is rebuilt only when it changes) + enabled + the why
function drawReadBtn(img, cropping, cropped) {
  const btn = $("readBtn");
  const mode = S.stopping ? "stopping" : S.reading ? "reading" : S.starting ? "starting" : "idle";
  if (btn.dataset.state !== mode) {
    btn.dataset.state = mode;
    const label = { idle: ["play", "read"], starting: ["play", "read"], reading: ["stop", "read.stop"], stopping: ["stop", "read.stopping"] }[mode];
    btn.innerHTML = `<i data-icon="${label[0]}"></i><span data-i18n="${label[1]}">${t(label[1])}</span>`;
    paintIcons(btn);
    btn.classList.toggle("stop", mode === "reading" || mode === "stopping");
    btn.classList.toggle("primary", mode === "idle" || mode === "starting");
  }
  if (mode === "reading") {
    btn.disabled = false;
    btn.dataset.tip = t("read.stopTitle");
    return;
  }
  if (mode === "stopping") {
    btn.disabled = true;
    btn.dataset.tip = t("read.stoppingTitle");
    return;
  }
  const why = mode === "starting" ? t("why.checking")
    : !S.model.ready ? S.model.why
      : !img ? t("why.openFirst")
        : cropping ? t("why.applyOrCancel")
          : S.finding ? t("why.finding")
            : !cropped ? t("why.cropDue") : "";
  btn.disabled = why !== "";
  btn.dataset.tip = why !== "" ? why : t("read.title", { p: readPreset() });
}
// can a read start, model-wise? The model in this browser's storage (and WebGPU); asked at start, whenever the model
// in the browser changes and after a refused read.
async function checkModel() {
  if (host.setup.noWebGpu) {
    S.model = { ready: false, why: t("why.noWebGpu") };
  } else {
    try {
      S.model = (await host.setup.modelMissing())
        ? { ready: false, why: t("why.setupModel") }
        : { ready: true, why: "" };
    } catch (e) {
      S.model = { ready: false, why: t("st.storage", { e: e instanceof Error ? e.message : e }) };
    }
  }
  applyState();
}

// iOS: maximum-scale=1 (index.html) stops the zoom into a field with a font under 16 px when it gets the focus (the
// markup view, the preset list, the text under Fit); the keyboard scrolls the page to the caret and leaves it there —
// the app is one screen: back to its place once the keyboard is gone (the visible height whole again)
if (window.visualViewport) {
  visualViewport.addEventListener("resize", () => {
    if (visualViewport.height >= window.innerHeight - 1 && (window.scrollX !== 0 || window.scrollY !== 0)) {
      window.scrollTo(0, 0);
    }
  });
}
document.addEventListener("focusout", () => {
  setTimeout(() => {
    if (window.scrollX !== 0 || window.scrollY !== 0) {
      window.scrollTo(0, 0);
    }
  }, 300);
});

// ── start ─────────────────────────────────────────────────────────────
(async () => {
  // the How to dialog shows the empty pane's steps (one copy in index.html)
  const steps = $("howSteps").cloneNode(true);
  steps.removeAttribute("id");
  $("howStepsDlg").replaceWith(steps);
  applyI18n();
  paintIcons();
  const st = await call("startup");
  S.settings = st.settings;
  S.presets = st.presets;
  S.marks = st.marks;
  S.editor = window.MarkupEditor($("editor"), S.marks, renderHtml, () => applyState());
  S.backend = st.backend;
  $("backendInfo").textContent = await host.engineLabel(st.backend, st.settings);
  buildFormatBar();
  fillPresetSel();
  drawCropSide();
  applyState();
  // the host's word on a missing model (not in this browser's storage yet)
  // the crash note first: it clears the device's flag, so the model note below is the current one
  const crash = host.takeCrashNote();
  await showModelNote();
  await checkModel();
  showCrashNote(crash);
  if (closedAway && !crash) {
    crumb("closed by the system in the background");
    status(t("st.iosClosed"), "ok");
  }
  // the setup opens by itself while the model is not in this browser's storage or the browser has no WebGPU (its
  // device check says why)
  if (host.setup.noWebGpu) {
    setup.open();
    return;
  }
  // the storage refused (a private window, site data blocked): the start note says so — no setup to open, it could
  // not store the model either
  let missing;
  try {
    missing = await host.setup.modelMissing();
  } catch (e) {
    crumb(`storage: ${e}`);
    return;
  }
  if (missing) {
    setup.open();
  }
})();
// the host's word on the model, at start and whenever the model in the browser changes (download / upload / delete);
// a missing model links the setup
async function showModelNote() {
  const note = await host.startupNote();
  if (note) status(note.text, note.kind, note.setup ? SETUP_LINK : note.link);
  else status(t("st.ready"));
}
// the language (Settings): applied at once (i18n.js setChoice — the static texts); here what this file wrote itself
$("langSel").onchange = () => setChoice($("langSel").value);
window.addEventListener("ocrlite:lang", () => {
  crumb(`language ${getChoice()}`);
  $("readBtn").dataset.state = ""; // the Read button's label drawn again
  if (S.marks !== null) {
    titleFormatBar();
  }
  if ($("settingsDlg").open) {
    drawPresets();
  }
  drawBakeInfo();
  if (isBusy()) {
    applyState();
  } else {
    showModelNote();
    checkModel(); // → applyState
  }
});
window.addEventListener("ocrlite:model-changed", () => {
  showModelNote();
  checkModel();
});

