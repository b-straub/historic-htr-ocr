// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// OCR Lite UI: image + prep chain on the left, the read as an editable rendered page on the right (editor.js).
// Everything heavy happens in Rust (ocr-web's wasm); this file keeps the prep chain and wires events.
// The host is web/bridge-web.js (ocr-web wasm + a WebGPU worker) — commands, events, file pick / drop / save, the
// host's settings.
import { scanOverlay } from "./scan.js";
import { setupDialog } from "./setup.js";
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
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.4a2.5 2.5 0 1 1 3.4 2.4c-.6.3-1 .8-1 1.5v.5"/><circle cx="12" cy="16.8" r=".6" fill="currentColor"/>',
  textsize: '<path d="m10 19 5-14 5 14"/><path d="M11.8 14h6.4"/><path d="m2 19 3.3-8 3.3 8"/><path d="M3.2 16.2h4.2"/>',
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
  glyphs: null,     // the page's writing size for the Text size gauge (glyph_height), null = not measured / as is
  sharp: false,     // the shown preview is the sharp (zoomed-in) one
  stem: "",
  reading: false,
  stopping: false,  // Stop pressed, the engine's ocr:done / ocr:error not yet in
  starting: false,  // Read pressed: the checks before the read (model in storage, its source) run
  finding: false,   // Find crop runs (auto_crop)
  model: { ready: false, why: "Checking the model…" }, // can a read start? why not (checkModel)
  streamed: "",
  settings: null,
  presets: null,    // { shipped, defaultName, user }
  marks: null,
  editor: null,
  backend: "",
};
const OP_LABEL = { rotateLeft: "rotate ⟲", rotateRight: "rotate ⟳", rotate180: "rotate 180°", autoContrast: "contrast", pencil: "pencil" };

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
  if (await host.setup.modelMissing()) {
    return;
  }
  const line = $("crashCheck");
  line.classList.remove("err");
  line.textContent = "Checking the stored model…";
  try {
    const r = await host.verifyModel();
    crumb(`model check: ${r.ok ? "intact" : `damaged (${r.damaged.join(", ")})`}`);
    line.textContent = r.text;
    line.classList.toggle("err", !r.ok);
    if (!r.ok) {
      status(r.text, "err");
    }
  } catch (e) {
    line.textContent = `The model check failed: ${e instanceof Error ? e.message : e}`;
    line.classList.add("err");
  }
}
// `crash`: the run before died in a GPU step (the crash marker) or lost the GPU (host.takeCrashNote); lastCrash: the
// page before ended without unloading (its breadcrumb trail)
function showCrashNote(crash) {
  if (!crash && !lastCrash) {
    return;
  }
  $("crashWhat").textContent = crash ?? `The page before ended without unloading (a crash, or the system freed the app); last action: ${lastCrash.at(-1) ?? "none"}.`;
  $("crashCheck").textContent = "";
  $("crashCopy").hidden = !lastCrash;
  $("crashCopy").textContent = "Copy the last actions";
  $("crashCopy").onclick = () => {
    const trail = [`OCR Lite — the page before ended without unloading (${navigator.userAgent})`, ...lastCrash].join("\n");
    navigator.clipboard.writeText(trail).then(() => { $("crashCopy").textContent = "Copied"; });
  };
  $("crashDlg").showModal();
  if (crash) {
    verifyAfterCrash();
  }
}

// ── image + prep chain ────────────────────────────────────────────────
async function openPath(src) {
  // Open is disabled while busy (applyState); a drop or the setup's "open an image" can still arrive
  if (isBusy()) {
    status(BUSY_WHY, "err");
    return;
  }
  let o;
  try {
    o = await host.openImage(src);
  } catch (e) {
    status(String(e), "err");
    throw e;
  }
  S.stem = o.stem;
  S.ops = [];
  setTab("image");
  zoom = 1;
  crumb(`open ${o.stem} ${o.w}×${o.h}`);
  status(`${o.stem} · ${o.w} × ${o.h}`);
  await refresh();
  showCropBar(true); // a new image: the crop bar open (cropping to the paper mostly helps)
}
// every preview request gets a number; only the latest one's answer is shown (a sharp preview still being made when
// the chain changes is dropped)
let previewSeq = 0;
async function refresh() {
  cancelCrop();
  gaugeShown = false; // a new image / step: the gauge goes (measured again below)
  const seq = ++previewSeq;
  const old = S.prev;
  const sharp = S.sharp && zoom > 1; // zoomed in on a sharp preview: ask for the sharp one straight away
  const p = await call("preview", { ops: S.ops, sharp });
  if (seq !== previewSeq) return;
  S.prev = p;
  S.sharp = sharp;
  if (!old || old.w !== p.w || old.h !== p.h) zoom = 1; // a crop / rotation: back to fit; a filter keeps the view
  $("img").src = p.url;
  $("stage").hidden = false;
  $("drop").hidden = true;
  layout();
  drawBakeInfo();
  $("opsChips").innerHTML = S.ops.map((o) => `<span class="chip">${o.op === "crop" ? `crop ${o.w}×${o.h}` : OP_LABEL[o.op]}</span>`).join("");
  applyState();
  measureGlyphs(); // a new image / step: the gauge measured again (Text size set)
}
function pushOp(op) {
  if (!S.prev) return;
  crumb(op.op === "crop" ? `crop ${op.w}×${op.h} at ${op.x},${op.y}` : op.op);
  S.ops.push(op);
  refresh();
}
document.querySelectorAll("[data-op]").forEach((b) => b.addEventListener("click", () => pushOp({ op: b.dataset.op })));
$("imgUndoBtn").onclick = () => { if (S.ops.length) { crumb("undo"); S.ops.pop(); refresh(); } };
$("resetBtn").onclick = () => { if (S.ops.length) { crumb("reset"); S.ops = []; refresh(); } };
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
  gaugeShown = false; // an image action: the gauge goes
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

// crop: only in the crop mode (Auto crop menu → Draw crop — explicitly enabled, since a zoom gesture drew crops); there press-drag-release on the image draws the rectangle;
// Crop / Enter applies, Cancel / Esc drops it and leaves the mode. Outside the mode a drag pans the image.
let cropMode = false;
let pageSide = ""; // the page mode last tapped in the crop bar (drawSide)
// how the crop at hand came about — ONE mark in the crop bar says it: "draw" (Draw, or a rectangle drawn by hand, also
// over a found one) marks Draw; "find" (a page mode's rectangle, also while its edges are adjusted) marks that mode
let cropSource = null;
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
// where a pointer lands on the drawn rectangle (image px): its edges / corners within reach of a finger, inside = move,
// else null (a new rectangle)
const GRAB_PX = 22;
function grabOf(b, p) {
  const r = GRAB_PX / D;
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
// the tabs sit in the title bar (one row), upright in a bar at the bottom; Read beside the zoom on the phone's side
// when the image shows (the title bar keeps one row), else in the title bar
const STACKED = matchMedia("(max-width: 820px) and (orientation: portrait), (orientation: landscape) and (max-height: 500px)");
const COMPACT = matchMedia("(orientation: landscape) and (max-height: 500px)");
const readHome = { parent: $("readBtn").parentNode, next: $("readBtn").nextSibling };
const helpHome = { parent: $("helpBtn").parentNode, next: $("helpBtn").nextSibling };
const tabsHome = { parent: $("paneTabs").parentNode, next: $("paneTabs").nextSibling };
function placeRead() {
  if (COMPACT.matches && document.body.dataset.tab !== "text") {
    $("zoomSeg").after($("readBtn"));
    // Help and Settings join the image head too: the title bar keeps one row, no buttons left alone at its end
    $("scaleSel").closest("label").after($("helpBtn"), $("settingsBtn"));
  } else {
    readHome.parent.insertBefore($("readBtn"), readHome.next);
    helpHome.parent.insertBefore($("helpBtn"), helpHome.next);
    $("helpBtn").after($("settingsBtn"));
  }
}
function placeTabs() {
  document.body.classList.toggle("stacked", STACKED.matches);
  if (COMPACT.matches) {
    $("aboutBtn").after($("paneTabs"));
  } else {
    tabsHome.parent.insertBefore($("paneTabs"), tabsHome.next);
  }
  placeRead();
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
  placeRead();
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

// ── Text size gauge: the page's glyph height (a sample blob outlined) beside the chosen target, at the image's scale ──
const SIZE_NAME = { smaller: "smaller", average: "average page", larger: "larger" };
let glyphSeq = 0;
// shown when the text size changes and when a read starts with one; any image action (a touch on the image, a zoom, a
// step, a crop, a new image) or its × hides it — the measure stays for the next showing
let gaugeShown = false;
function showGauge(on) {
  gaugeShown = on;
  drawGauge();
}
$("gaugeClose").onclick = () => showGauge(false);
// a tap on the label is its own: the pane under it would take the pointer for a pan / crop
$("gaugeLabel").addEventListener("pointerdown", (e) => e.stopPropagation());
async function measureGlyphs() {
  const seq = ++glyphSeq;
  S.glyphs = null;
  drawGauge();
  drawBakeInfo();
  if (S.settings.textSize === null || !S.prev) {
    return;
  }
  let g;
  try {
    g = await call("glyph_height", { ops: S.ops });
  } catch {
    return; // call() put the error in the status line
  }
  if (seq !== glyphSeq) return;
  S.glyphs = g;
  drawGauge();
  drawBakeInfo();
}
// the model size beside the image: the preview's bake; with a Text size, once measured, that size's decision and model
// input (glyph_height's sizes — the read's ocr:input reports the same)
function drawBakeInfo() {
  const p = S.prev, g = S.glyphs, size = S.settings === null ? null : S.settings.textSize;
  if (p === null) {
    return;
  }
  const info = $("bakeInfo");
  if (size !== null && g !== null && g.none === undefined) {
    const d = g.sizes[size];
    info.textContent = `${p.w} × ${p.h} → model ${d.w} × ${d.h} · ${textSize(d)}`;
    info.title = d.summary;
    return;
  }
  const b = p.bake;
  const fw = b.scaledW + b.left + b.right, fh = b.scaledH + b.top + b.bottom;
  info.textContent = size !== null && g !== null ? `${p.w} × ${p.h} → model ${fw} × ${fh} · ${textSize({ warning: g.none })}` : `${p.w} × ${p.h} → model ${fw} × ${fh}`;
  info.title = b.scaledW === p.w ? "kept at its size, white pad to the 28 px grid" : `scaled to ${b.scaledW} × ${b.scaledH}, white pad to the 28 px grid`;
}
function drawGauge() {
  const g = S.glyphs, size = S.settings === null ? null : S.settings.textSize;
  const el = $("gauge");
  if (g === null || size === null || g.none !== undefined || !gaugeShown) {
    el.hidden = true;
    if (g !== null && g.none !== undefined) {
      status(`Text size: ${g.none} — the image is read as is`, "err");
    }
    return;
  }
  const k = D / g.scale; // CSS px per bake px
  const [x, y, w, h] = g.sample;
  const t = g.targets[size];
  const base = (y + h) * k;
  // both at their exact heights (the borders inside): the blob's and the target's compare as they are
  Object.assign($("gaugeSample").style, { left: `${x * k}px`, top: `${y * k}px`, width: `${w * k}px`, height: `${h * k}px` });
  // the factor the read applies: capped where the page reaches the model's 1540 px edge — the bar = the height the
  // read gets, the dashed rest = the target it cannot reach
  const f = Math.min(t / g.xh, g.cap);
  const got = g.xh * f;
  Object.assign($("gaugeTarget").style, { left: `${(x + w) * k + 6}px`, top: `${base - got * k}px`, height: `${got * k}px` });
  const short = t - got > 0.5;
  $("gaugeMissing").hidden = !short;
  if (short) {
    Object.assign($("gaugeMissing").style, { left: `${(x + w) * k + 6}px`, top: `${base - t * k}px`, height: `${(t - got) * k}px` });
  }
  // the label names the three marks: blue = the letters now, solid orange = what the read gets, dashed = the target the
  // read cannot reach — the page already fills the model's 1540 px input; only a tighter crop leaves room to enlarge
  const by = Math.abs(f - 1) < 0.005 ? "as is" : `×${f.toFixed(2)}`;
  const tail = !short ? "" : `<span class="dash">${SIZE_NAME[size]} ${t}: full size — crop tighter for more</span>`;
  $("gaugeText").innerHTML = `<b>now ${Math.round(g.xh)}</b><span class="sep"> · </span><i>read ${Math.round(got)} (${short ? by : `${SIZE_NAME[size]}, ${by}`})</i>${tail}`;
  el.hidden = false;
  // under the blob, kept inside the image (its × in reach)
  const label = $("gaugeLabel"), room = S.prev.w * D;
  label.style.left = "0px"; // measured at the left edge: placed further right it would wrap before it is moved back
  Object.assign(label.style, { left: `${Math.max(0, Math.min(x * k, room - label.offsetWidth - 2))}px`, top: `${base + 6}px` });
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
  drawGauge();
  applyState(); // every change of the crop rect (and of the zoom, via layout) passes here
}
function setCropMode(on) {
  cropMode = on;
  $("stage").classList.toggle("cropping", on);
  applyState();
}
// the crop bar: ONE place for the crop, over the image head (never over the image, never moving it) — open after an
// image loads, the crop button opens / closes it. Its first line: Draw (a rectangle by hand) and the page modes (each
// tap finds the paper in that mode); a rectangle, found or drawn, is adjustable alike (the crop mode) and adds the
// second line: what it is, Cancel, Crop (a touch screen has no Esc / Enter). Crop applies it and closes the bar
let cropBarOpen = false;
function showCropBar(on) {
  cropBarOpen = on;
  if (!on) {
    cancelCrop();
  }
  applyState();
}
function drawCropBar(cropping, cropped) {
  $("cropBar").hidden = !cropBarOpen || S.prev === null;
  $("autoCropBtn").classList.toggle("on", cropBarOpen);
  $("drawCropBtn").classList.toggle("on", cropSource === "draw");
  $("cropDone").hidden = !cropped;
  if (cropped) {
    const c = S.ops.filter((o) => o.op === "crop").at(-1);
    $("cropDone").textContent = `Cropped to ${c.w} × ${c.h}`;
  }
  $("cropAct").hidden = !cropping;
  $("cropApplyBtn").hidden = pending === null;
  drawSide();
  $("cropText").textContent = drag !== null ? "Drawing…"
    : pending === null ? "Drag a rectangle over the text"
    : cropSource === "find" ? `Found ${pending.w} × ${pending.h}`
    : `${pending.w} × ${pending.h}`;
}
$("cropApplyBtn").onclick = () => applyCrop();
$("cropCancelBtn").onclick = () => cancelCrop();
function cancelCrop() {
  pending = null;
  drag = null;
  cropSource = null;
  drawOverlay();
  setCropMode(false);
}
function applyCrop() {
  if (!pending) return;
  const op = { op: "crop", ...pending };
  cancelCrop();
  cropBarOpen = false;
  pushOp(op);
}

// pointers on the pane: mouse / pen / one finger draw the crop in the crop mode, else pan the image; two fingers
// pinch-zoom about their midpoint and pan (Pointer Events — iPadOS Safari, Android Chrome); a double click / double tap = fit.
// A crop and a zoom never mix: once a rectangle is under way (the finger moved PINCH_GRACE_PX or held PINCH_GRACE_MS)
// and GESTURE_QUIET_MS after it, another finger, a pinch or a double tap leave the view as it is — a second finger
// sooner is a pinch (the rectangle dropped); after a pinch a finger landing again within that time draws nothing
const GESTURE_QUIET_MS = 500, PINCH_GRACE_MS = 250, PINCH_GRACE_PX = 10;
(() => {
  const pane = $("imgPane"), stage = $("stage"), img = $("img");
  const touches = new Map(); // pointerId → client point
  let pinch = null;          // { dist0, zoom0, mid }
  let lastTap = null;        // { t, x, y } of the last short tap
  let pan = null;            // { id, x, y, left, top, t } — a drag outside the crop mode
  let cropEnd = -Infinity, pinchEnd = -Infinity; // when the last crop drag / pinch ended
  const cropUnderWay = () => drag !== null && (performance.now() - drag.t > PINCH_GRACE_MS || Math.hypot(drag.cx - drag.x, drag.cy - drag.y) > PINCH_GRACE_PX);
  const zoomLocked = () => cropUnderWay() || performance.now() - cropEnd < GESTURE_QUIET_MS;
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
    if (!S.prev || stage.hidden) return;
    if (gaugeShown) {
      showGauge(false); // a touch on the image: the gauge goes
    }
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
    const grab = pending === null ? null : grabOf(pending, p);
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
      const g = cropMode && pending !== null ? grabOf(pending, imgPos(ev)) : null;
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
    if (d.mode === "new") {
      cropSource = "draw";
    }
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
  pane.addEventListener("dblclick", (ev) => { if (!cropMode && stage.contains(ev.target)) zoomTo(1, paneCentre()); });
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
// auto crop (the built-in algorithm, ocr-core autocrop.rs): the page mode tapped → a proposed rect (dashed), in the crop
// mode like a drawn one — its edges adjust it, Crop applies it
function showPending(r) {
  drag = null;
  pending = { ...r };
  cropSource = "find";
  $("rect").classList.add("auto");
  setCropMode(true);
  drawOverlay();
}
// the page mode of the found rectangle — marked only while that rectangle shows (a mark before any find looked as if it
// had run, and the first tap seemed to hit a chosen button)
function drawSide() {
  for (const b of $("pageSideSeg").children) {
    b.classList.toggle("on", cropSource === "find" && b.dataset.side === pageSide);
  }
}
for (const b of $("pageSideSeg").children) {
  b.onclick = () => {
    pageSide = b.dataset.side;
    findCrop(); // a page mode tapped: the paper found in it at once
  };
}
$("autoCropBtn").onclick = () => showCropBar(!cropBarOpen);
// Draw / the page modes (disabled without an image — applyState); Draw again leaves the crop mode
function drawCrop() {
  if (cropSource === "draw") {
    cancelCrop(); // Draw again: out of the crop
    return;
  }
  pending = null;
  cropSource = "draw";
  drawOverlay();
  setCropMode(true);
  status(KEYS ? "Draw crop: drag on the image — Crop / Enter applies, Cancel / Esc leaves" : "Drag a rectangle over the text, then Crop", "ok");
}
async function findCrop() {
  const opts = {
    // the mode flags (keepSpread = is_spread || single_leaf)
    keepSpread: pageSide === "double" || pageSide === "leaf",
    pageSide: pageSide === "left" || pageSide === "right" ? pageSide : null,
    singleLeaf: pageSide === "leaf",
  };
  status("Finding the paper…", "busy");
  S.finding = true;
  applyState();
  let r;
  try {
    r = await call("auto_crop", { ops: S.ops, opts });
  } finally {
    S.finding = false;
    applyState();
  }
  if (r.x === 0 && r.y === 0 && r.w === S.prev.w && r.h === S.prev.h) {
    cancelCrop();
    status("Auto crop: nothing to trim", "ok");
    return;
  }
  showPending(r);
  status(`Auto crop: ${r.w} × ${r.h} at (${r.x}, ${r.y}) — Crop to apply, Cancel to drop`, "ok");
}
$("drawCropBtn").onclick = drawCrop;

document.addEventListener("keydown", (e) => {
  if ((!pending && !cropMode) || document.activeElement === $("editor") || document.querySelector("dialog[open]")) return;
  if (e.key === "Enter") { e.preventDefault(); applyCrop(); }
  if (e.key === "Escape") { e.preventDefault(); cancelCrop(); }
});

// ── text: rendered editor + markup view ───────────────────────────────
const renderHtml = (text) => invoke("render", { text });
function buildFormatBar() {
  const sample = { del: "<del>S</del>", ins: "<ins>A</ins>", u: "<u>U</u>", sup: "x<sup>2</sup>", sub: "x<sub>2</sub>" };
  const name = { del: "Strike", ins: "Insertion", u: "Underline", sup: "Superscript", sub: "Subscript" };
  for (const p of S.marks.pairs) {
    const b = document.createElement("button");
    b.className = "ib fmt";
    b.dataset.tag = p.tag;
    b.innerHTML = sample[p.tag] ?? p.tag;
    b.title = `${name[p.tag] ?? p.tag} — written ${p.open}…${p.close}`;
    b.addEventListener("mousedown", (e) => e.preventDefault()); // keep the editor's selection
    b.onclick = () => S.editor.toggle(p.tag).then(applyState);
    $("fmtSeg").append(b);
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
  alert("Tech preview: OCR Lite does not keep your edits yet — a crash or a reload loses the text on the right. Copy or save what you change.\n\nReal editing, with the work kept, is the next major step.");
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
  cropBarOpen = false; // the read's scan shows the whole image
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
  await S.editor.show("");
  $("src").value = "";
  S.starting = false; // straight into reading: Read never shows enabled in between
  // the image back to fit: a zoomed image's large layer is given back before the read takes its GPU memory
  if (zoom > 1) {
    crumb("zoom to fit for the read");
    zoomTo(1, paneCentre());
  }
  if (S.settings.textSize !== null) {
    showGauge(true); // the read's text size shown as it starts
  }
  setReading(true);
  status("Starting…", "busy");
  try {
    S.readBake = null;
    await call("read", { ops: S.ops, preset: $("presetSel").value, textSize: S.settings.textSize });
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
    status(`browser storage unavailable: ${e instanceof Error ? e.message : e}`, "err");
    return false;
  }
  if (missing) {
    status("Read needs the model — it is not in this browser yet", "err", SETUP_LINK);
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
    status(`the check before the read failed: ${e instanceof Error ? e.message : e}`, "err");
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
const secs = (ms) => `${(ms / 1000).toFixed(1)} s`;
function phaseTimes(r) {
  const t = [];
  if (r.loadMs != null) t.push(`load ${secs(r.loadMs)}`);
  if (r.encodeMs != null) t.push(`image ${secs(r.encodeMs)}`);
  if (r.prefillMs != null) t.push(`prefill ${secs(r.prefillMs)}${r.encodeMs == null ? " (with the image)" : ""}`);
  return t;
}
const PHASES = { encodingImage: "Encoding image…", loadingText: "Loading the text model…", readingPrompt: "Reading prompt…", decoding: "Reading…" };
const scan = scanOverlay($("stage"), $("img"));
listen("ocr:phase", (e) => scan.phase(e.payload, S.readBake));
/// the Text size decision in a few plain words (Rust's tools::page_scale: reason / warning) — the measure in the tooltip
function textSize(d) {
  if (d.warning !== null) {
    return "⚠ no text found: as is";
  }
  if (d.factor > 1.0001) {
    return `text ×${d.factor.toFixed(1)} larger`;
  }
  if (d.factor < 0.9999) {
    return `text ×${d.factor.toFixed(2)} smaller`;
  }
  return /capped/.test(d.reason) ? "text as is (full size)" : "text already fits";
}
// a tap on the model size line: its detail (the tooltip) in the status line — a touch screen shows no tooltip
$("bakeInfo").onclick = () => {
  if ($("bakeInfo").title !== "") {
    status($("bakeInfo").title, "ok");
  }
};
// the read's model input: its plan (the scan sweeps it), its size and the Scale decision beside the image
listen("ocr:input", (e) => {
  const { plan, w, h, scale } = e.payload;
  S.readBake = plan;
  const p = S.prev;
  const size = p ? `${p.w} × ${p.h} → model ${w} × ${h}` : `model ${w} × ${h}`;
  $("bakeInfo").textContent = scale === null ? size : `${size} · ${textSize(scale)}`;
  // the measure behind it (pitch → target) in the tooltip
  $("bakeInfo").title = scale === null ? "" : scale.warning === null ? scale.summary : `${scale.summary} — ${scale.warning}`;
  crumb(`input ${w}×${h}${scale === null ? "" : ` ${scale.summary}${scale.warning === null ? "" : ` ⚠ ${scale.warning}`}`}`);
});
listen("ocr:phase", (e) => crumb(e.payload.phase)); // read start … encodingImage = the vision weights' load
listen("ocr:phase", (e) => status([PHASES[e.payload.phase] ?? e.payload.phase, ...phaseTimes(e.payload)].join(" · "), "busy"));
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
  const end = r.cancelled ? (stoppedHidden ? " · stopped: the app went to the background" : " · stopped") : r.stoppedOnEog ? "" : " · hit the token limit (runaway?)";
  const total = (r.loadMs ?? 0) + (r.encodeMs ?? 0) + r.prefillMs + r.decodeMs;
  // tokenChoice (the wgpu engine): "gpu" = greedy, chosen on the GPU and run ahead; "cpu" = sampled (llama.cpp: absent)
  const choice = r.tokenChoice ? [`token choice: ${r.tokenChoice}`] : [];
  status(`${[...phaseTimes(r), `${r.outTokens.toLocaleString()} tokens`, `${tps} tok/s`, ...choice, `total ${secs(total)}`].join(" · ")}${end}`,
    !r.stoppedOnEog && !r.cancelled ? "err" : "ok");
});
listen("ocr:error", (e) => {
  // the read ended by the engine's release in the background: no error — it starts again on return; a Stop before the
  // text (the engine released at once, host cancel): stopped, no text
  const released = restartOnReturn && away;
  const stopped = S.stopping && String(e.payload) === "stopped";
  setReading(false);
  if (released) {
    status("Stopped: the app went to the background — the read starts again when you return", "ok");
  } else if (stopped) {
    status("Stopped before the text began", "ok");
  } else {
    status(String(e.payload), "err");
  }
});

// ── copy / save ───────────────────────────────────────────────────────
$("copyBtn").onclick = () => navigator.clipboard.writeText(currentMarkup()).then(() => status("Markup copied", "ok"));
$("saveBtn").onclick = (e) => { e.stopPropagation(); $("saveMenu").hidden = !$("saveMenu").hidden; };
document.addEventListener("click", () => { $("saveMenu").hidden = true; });
for (const b of $("saveMenu").querySelectorAll("button")) {
  b.onclick = async () => {
    $("saveMenu").hidden = true;
    const ext = b.dataset.save;
    if (ext === "png" && !S.prev) { status("Open an image first", "err"); return; }
    let p;
    try {
      p = await host.saveAs(ext, { stem: S.stem, ops: S.ops, text: currentMarkup() });
    } catch (e) {
      status(String(e), "err");
      return;
    }
    if (p) status(`Saved ${p}`, "ok");
  };
}

// ── settings + presets ────────────────────────────────────────────────
const FIELDS = [
  ["name", "text", "Name"], ["temperature", "number", "Temp"], ["topP", "number", "Top p"], ["repetitionPenalty", "number", "Rep. pen."],
  ["repetitionWindow", "number", "Window"], ["frequencyPenalty", "number", "Freq. pen."], ["maxNewTokens", "number", "Max tokens"],
  ["gtTokenCapFactor", "number", "GT cap ×"],
];
let draftUser = [];
function fillPresetSel() {
  const sel = $("presetSel");
  sel.innerHTML = "";
  for (const p of [...S.presets.shipped, ...S.presets.user]) {
    const o = document.createElement("option");
    o.value = o.textContent = p.name;
    sel.append(o);
  }
  sel.value = S.settings.preset;
  $("scaleSel").value = S.settings.textSize === null ? "" : S.settings.textSize;
}
$("scaleSel").onchange = async () => {
  const v = $("scaleSel").value;
  const next = { ...S.settings, textSize: v === "" ? null : v };
  try { await call("save_settings", { settings: next }); S.settings = next; } catch { $("scaleSel").value = S.settings.textSize === null ? "" : S.settings.textSize; }
  crumb(`text size ${v === "" ? "as is" : v}`);
  gaugeShown = v !== "";
  if (S.glyphs !== null) {
    drawGauge(); // the page measured already: only the target changes
    drawBakeInfo();
  } else {
    measureGlyphs();
  }
};
$("presetSel").onchange = async () => {
  const next = { ...S.settings, preset: $("presetSel").value };
  try { await call("save_settings", { settings: next }); S.settings = next; } catch { $("presetSel").value = S.settings.preset; }
  applyState(); // Read's title names the preset
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
    if (!editable) inp.title = "a shipped preset is read-only — Duplicate makes an editable copy";
    if (k === "name") inp.style.width = `${nameCh}ch`;
    if (editable) inp.oninput = () => { draftUser[idx][k] = type === "number" ? Number(inp.value) : inp.value; };
    td.append(inp);
    tr.append(td);
  }
  const td = document.createElement("td"), b = document.createElement("button");
  td.className = "act";
  b.type = "button";
  b.className = "btn small";
  if (editable) {
    b.textContent = "Delete";
    b.onclick = () => { draftUser.splice(idx, 1); drawPresets(); };
  } else {
    b.textContent = "Duplicate";
    b.onclick = () => {
      const names = new Set([...S.presets.shipped, ...draftUser].map((x) => x.name));
      let n = `${p.name}-mine`, k = 2;
      while (names.has(n)) n = `${p.name}-mine${k++}`;
      draftUser.push({ ...p, name: n });
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
  const t = $("presetTable");
  nameCh = Math.max(12, ...[...S.presets.shipped, ...draftUser].map((p) => p.name.length)) + 3;
  t.innerHTML = "<tr>" + FIELDS.map(([, , label]) => `<th>${label}</th>`).join("") + "<th></th></tr>";
  S.presets.shipped.forEach((p, i) => t.append(presetRow(p, false, i)));
  draftUser.forEach((p, i) => t.append(presetRow(p, true, i)));
}
$("settingsBtn").onclick = () => {
  host.settingsOpen(S.settings);
  draftUser = S.presets.user.map((p) => ({ ...p }));
  drawPresets();
  $("settingsErr").textContent = "";
  $("settingsDlg").showModal();
};
$("settingsCancel").onclick = () => $("settingsDlg").close();
$("settingsSave").onclick = async () => {
  const next = host.settingsCollect(S.settings);
  try {
    S.presets = await invoke("save_user_presets", { user: draftUser });
    await invoke("save_settings", { settings: next });
    S.settings = next;
    fillPresetSel();
    $("backendInfo").textContent = await host.engineLabel(S.backend, S.settings);
    $("settingsDlg").close();
  } catch (e) {
    $("settingsErr").textContent = String(e);
  }
};

// ── about: version / commit / build time (which build is running) ──
async function showAbout() {
  const b = await host.buildInfo();
  $("aboutVersion").textContent = b.version;
  $("aboutGit").textContent = b.git.endsWith("-dirty") ? `${b.git.slice(0, -6)} + uncommitted changes` : b.git;
  $("aboutBuilt").textContent = b.built ? new Date(b.built * 1000).toLocaleString() : "—";
  $("aboutEngine").textContent = $("backendInfo").textContent;
  $("aboutDlg").showModal();
}
$("aboutBtn").onclick = showAbout;

// ── first-run setup + help ────────────────────────────────────────────
const setup = setupDialog(host.setup, { pickImage, openHelp: () => $("helpDlg").showModal() });
const SETUP_LINK = { label: "Set up OCR Lite", run: () => { setup.open(); return null; } };
$("helpBtn").onclick = () => $("helpDlg").showModal();
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
// ONE place: applyState() sets every control's enabled / disabled (+ the why in its title, the Read button's label)
// from the state; every state change calls it — no control is toggled anywhere else.
//
// States: no image · image (S.prev) · crop mode (Draw crop) · pending (a crop rect waiting for Crop / Cancel — drawn
// or found) · finding (Find crop runs) · busy = starting (Read's checks: model, its source) | reading | stopping (Stop
// pressed, the engine not done yet) · no model (browser: not in this browser's storage, no WebGPU; "checking" until
// the start's check answered) · no text / text · markup view · dialogs: modal (showModal) — the page behind is inert,
// nothing changes here.
//
//   control                        enabled when                                   disabled → title says
//   Open (+ drop, the empty pane)  not busy                                       busy: "Wait for the read to end"
//   rotate ×3, contrast, pencil    image · no crop mode / pending · not finding   "Open an image first" · "Finish the
//                                  · not busy                                     crop first (Crop / Cancel)" · busy ·
//                                                                                 "Finding the crop…"
//   crop menu ▸                    image · not finding · not busy                 "Open an image first" · busy ·
//                                                                                 "Finding the crop…"
//     Page side                    always (a setting, kept per browser)
//     Draw crop, Find crop         as the menu button (also the crop tip's two)
//   crop tip: Use whole page,      always while the tip shows (it shows only with an image, never while busy)
//     Don't show again
//   image undo, reset              a step to undo · no crop mode / pending ·      as the steps; no step: its own title
//                                  not finding · not busy
//   preset select                  not busy                                       busy (the running read keeps its own)
//   Help                           always
//   About, Fit                     not busy (their re-layout during a read ended in crashes on the iPhone)
//   Settings                       not busy (the model store must not change      busy
//                                  under a read)
//   Read                           image · model ready · no crop mode / pending   no model: its why (e.g. "Set up the
//                                  · not finding · not starting                   model first") · "Open an image first" ·
//                                                                                 "Apply or cancel the crop first" ·
//                                                                                 "Finding the crop…" · starting:
//                                                                                 "Checking before the read…"
//     reading → "Stop" (enabled) · stopping → "Stopping…" (disabled)
//   zoom out, fit                  image · zoom > 100 %
//   zoom in                        image · zoom < 800 %
//   format, ▨, table cell          the rendered view · not busy                   busy · (hidden in the markup view)
//   undo, redo                     the rendered view · not busy · a step there    (hidden in the markup view)
//   Markup                         not busy                                       busy
//   Copy                           text · not busy                                "Nothing to copy yet" · busy
//   Save ▸                         text or image · not busy
//     .txt, .html                  text · not busy                                "Nothing to save yet — read an image"
//     .png (model input)           image · not busy                               "Open an image first"
const BUSY_WHY = "Wait for the read to end — or Stop it";
const isBusy = () => S.starting || S.reading || S.stopping;
// sets a control's state; the why goes into its title while disabled, its own title comes back when enabled
function enable(el, on, why = "") {
  if (el.dataset.title === undefined) {
    el.dataset.title = el.title;
  }
  el.disabled = !on;
  el.title = !on && why ? why : el.dataset.title;
}
function applyState() {
  const busy = isBusy();
  const img = S.prev !== null;
  const cropping = cropMode || pending !== null || drag !== null;
  const cropped = S.ops.some((o) => o.op === "crop");
  drawCropBar(cropping, cropped);
  const editing = S.editor !== null && !markupView;
  const text = S.editor !== null && currentMarkup().trim() !== "";
  const noImg = "Open an image first";
  const cropWhy = "Finish the crop first (Crop / Cancel)";
  // the title bar
  enable($("openBtn"), !busy, BUSY_WHY);
  const steps = img && !cropping && !S.finding && !busy;
  const stepsWhy = !img ? noImg : busy ? BUSY_WHY : cropping ? cropWhy : "Finding the crop…";
  for (const b of document.querySelectorAll("[data-op]")) {
    enable(b, steps, stepsWhy);
  }
  const crop = img && !S.finding && !busy;
  const cropMenuWhy = !img ? noImg : busy ? BUSY_WHY : "Finding the crop…";
  enable($("autoCropBtn"), crop, cropMenuWhy);
  // an image already cropped: the crop bar's tools off — its Reset goes back to the original
  const cropAgain = crop && !cropped;
  const cropAgainWhy = cropped && crop ? "Already cropped — Reset goes back to the original" : cropMenuWhy;
  enable($("drawCropBtn"), cropAgain, cropAgainWhy);
  for (const b of $("pageSideSeg").children) {
    enable(b, cropAgain, cropAgainWhy);
  }
  enable($("imgUndoBtn"), steps && S.ops.length > 0, steps ? "" : stepsWhy);
  enable($("resetBtn"), steps && S.ops.length > 0, steps ? "" : stepsWhy);
  enable($("presetSel"), !busy, BUSY_WHY);
  enable($("scaleSel"), !busy, BUSY_WHY);
  enable($("settingsBtn"), !busy, BUSY_WHY);
  drawReadBtn(img, cropping);
  // the image card
  enable($("zoomOutBtn"), img && zoom > 1);
  enable($("zoomFitBtn"), img && zoom > 1);
  enable($("zoomInBtn"), img && zoom < zoomMax() - 0.005);
  // the text card
  for (const b of $("fmtSeg").children) {
    enable(b, editing && !busy, BUSY_WHY);
  }
  enable($("gapBtn"), editing && !busy, BUSY_WHY);
  enable($("tabBtn"), editing && !busy, BUSY_WHY);
  enable($("undoBtn"), editing && !busy && S.editor.canUndo(), busy ? BUSY_WHY : "");
  enable($("redoBtn"), editing && !busy && S.editor.canRedo(), busy ? BUSY_WHY : "");
  enable($("markupBtn"), S.editor !== null && !busy, BUSY_WHY);
  // Fit and About stay off while a read runs (their re-layout during a read ended in crashes on the iPhone)
  enable($("fitBtn"), S.editor !== null && !markupView && !busy, busy ? BUSY_WHY : markupView ? "Fit applies to the rendered text" : "");
  enable($("aboutBtn"), !busy, BUSY_WHY);
  enable($("copyBtn"), text && !busy, busy ? BUSY_WHY : "Nothing to copy yet");
  enable($("saveBtn"), (text || img) && !busy, busy ? BUSY_WHY : "Nothing to save yet — read an image");
  for (const b of $("saveMenu").querySelectorAll("button")) {
    if (b.dataset.save === "png") {
      enable(b, img && !busy, busy ? BUSY_WHY : noImg);
    } else {
      enable(b, text && !busy, busy ? BUSY_WHY : "Nothing to save yet — read an image");
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
  status(`Read waits: ${btn.title}`, "err");
  const at = pending !== null ? $("cropApplyBtn") : cropMode ? $("cropCancelBtn") : null;
  if (at !== null) {
    at.classList.remove("attn");
    void at.offsetWidth;
    at.classList.add("attn");
  }
});
// the Read button: Read / Stop / Stopping… (the label is rebuilt only when it changes) + enabled + the why
function drawReadBtn(img, cropping) {
  const btn = $("readBtn");
  const mode = S.stopping ? "stopping" : S.reading ? "reading" : S.starting ? "starting" : "idle";
  if (btn.dataset.state !== mode) {
    btn.dataset.state = mode;
    const label = { idle: ["play", "Read"], starting: ["play", "Read"], reading: ["stop", "Stop"], stopping: ["stop", "Stopping…"] }[mode];
    btn.innerHTML = `<i data-icon="${label[0]}"></i><span>${label[1]}</span>`;
    paintIcons(btn);
    btn.classList.toggle("stop", mode === "reading" || mode === "stopping");
    btn.classList.toggle("primary", mode === "idle" || mode === "starting");
  }
  if (mode === "reading") {
    btn.disabled = false;
    btn.title = "Stop the read";
    return;
  }
  if (mode === "stopping") {
    btn.disabled = true;
    btn.title = "Stopping the read…";
    return;
  }
  const why = mode === "starting" ? "Checking before the read…"
    : !S.model.ready ? S.model.why
      : !img ? "Open an image first"
        : cropping ? "Apply or cancel the crop first (Crop / Cancel)"
          : S.finding ? "Finding the crop…" : "";
  btn.disabled = why !== "";
  btn.title = why !== "" ? why : `Read the image (preset ${$("presetSel").value})`;
}
// can a read start, model-wise? The model in this browser's storage (and WebGPU); asked at start, whenever the model
// in the browser changes and after a refused read.
async function checkModel() {
  if (host.setup.noWebGpu) {
    S.model = { ready: false, why: "Reading needs WebGPU — this browser has none (see Help)" };
  } else {
    try {
      S.model = (await host.setup.modelMissing())
        ? { ready: false, why: "Set up the model first — it is not in this browser yet" }
        : { ready: true, why: "" };
    } catch (e) {
      S.model = { ready: false, why: `browser storage unavailable: ${e instanceof Error ? e.message : e}` };
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
  applyState();
  // the host's word on a missing model (not in this browser's storage yet)
  // the crash note first: it clears the device's flag, so the model note below is the current one
  const crash = host.takeCrashNote();
  await showModelNote();
  await checkModel();
  showCrashNote(crash);
  if (closedAway && !crash) {
    crumb("closed by the system in the background");
    status("iOS closed OCR Lite while it was in the background — the page was loaded again", "ok");
  }
  // the setup opens by itself while the model is not in this browser's storage or the browser has no WebGPU (its
  // device check says why)
  if (host.setup.noWebGpu || await host.setup.modelMissing()) {
    setup.open();
  }
})();
// the host's word on the model, at start and whenever the model in the browser changes (download / upload / delete);
// a missing model links the setup
async function showModelNote() {
  const note = await host.startupNote();
  if (note) status(note.text, note.kind, note.setup ? SETUP_LINK : note.link);
  else status("Ready");
}
window.addEventListener("ocrlite:model-changed", () => {
  showModelNote();
  checkModel();
});

