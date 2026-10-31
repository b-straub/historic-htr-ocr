// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// OCR Lite's own tooltips: in an installed app window (Chrome on macOS) the browser's native title tooltips almost
// never appear, so no control carries a title — its tip is data-tip, shown here:
//   setTip(el, text)  a control's own tip (app.js enable() puts a why in data-tip while the control is off and its own
//                     one back from data-tip-own); a control without visible text gets it as its aria-label too
//   the tip           one floating element (#tip, a manual popover: above an open dialog too) after DELAY_MS of a
//                     mouse / pen resting on a control — below it, above when there is no room, inside the viewport;
//                     gone on a press, a scroll, a key (Escape), the window losing focus, the pointer leaving. Never on
//                     touch.
// The control under the pointer is found by hit-testing (document.elementFromPoint), not by its own events: a disabled
// button gets no pointer events in every browser, and a disabled Read lets the pointer through (pointer-events: none,
// style.css) — the bar's controls are then found by their boxes.

const DELAY_MS = 500;
const GAP_PX = 6;

export function setTip(el, text) {
  if (el.dataset.tipOwn !== undefined) {
    el.dataset.tipOwn = text;
  }
  el.dataset.tip = text;
  if (el.textContent.trim() === "" && !el.hasAttribute("data-i18n-aria")) {
    el.setAttribute("aria-label", text);
  }
}

/// the control with a tip at the client point (x, y), or null: the element there (or its nearest ancestor with a tip);
/// none — a control that lets the pointer through (a disabled Read): the smallest tipped control of the bar whose box
/// holds the point
function tipAt(x, y) {
  const at = document.elementFromPoint(x, y);
  const hit = at === null ? null : at.closest("[data-tip]");
  if (hit !== null) {
    return hit;
  }
  let best = null;
  let area = Infinity;
  for (const el of document.querySelectorAll(".titlebar [data-tip]")) {
    const r = el.getBoundingClientRect();
    if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom && r.width * r.height < area) {
      best = el;
      area = r.width * r.height;
    }
  }
  return best;
}

// on the page only: the module is also imported where there is no document (the Node unit tests, through i18n.js)
const ON_PAGE = typeof document !== "undefined";
const tip = ON_PAGE ? Object.assign(document.createElement("div"), { id: "tip", role: "tooltip" }) : null;
if (ON_PAGE) {
  tip.setAttribute("popover", "manual");
  document.body.append(tip);
}
let target = null; // the control the pointer rests on
let shown = false;
let timer = 0;

function hide() {
  clearTimeout(timer);
  if (shown) {
    tip.hidePopover();
    shown = false;
  }
}

function show(el) {
  const text = el.dataset.tip;
  if (text === undefined || text === "" || !el.isConnected) {
    return;
  }
  tip.textContent = text;
  // shown again: on top of a dialog opened since
  if (shown) {
    tip.hidePopover();
  }
  tip.showPopover();
  shown = true;
  const r = el.getBoundingClientRect();
  const w = tip.offsetWidth, h = tip.offsetHeight;
  const below = r.bottom + GAP_PX;
  const top = below + h <= innerHeight - 4 ? below : Math.max(4, r.top - GAP_PX - h);
  const left = Math.min(Math.max(4, r.left + r.width / 2 - w / 2), innerWidth - w - 4);
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
}

if (ON_PAGE) {
  document.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch") {
      return;
    }
    const el = tipAt(e.clientX, e.clientY);
    if (el === target) {
      return;
    }
    hide();
    target = el;
    if (el !== null) {
      timer = setTimeout(() => show(el), DELAY_MS);
    }
  });
  // a press, a scroll, a key, the window or the page losing the pointer: the tip goes (it comes again on the next
  // control)
  document.addEventListener("pointerdown", hide, true);
  document.addEventListener("scroll", hide, true);
  document.addEventListener("keydown", hide, true);
  window.addEventListener("blur", hide);
  document.documentElement.addEventListener("pointerleave", () => {
    hide();
    target = null;
  });
}
