// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// OCR Lite's languages: English and German. The texts live in resource files, this module only loads and applies them:
//   lang/<l>.json         the short texts, flat { key: text } (the same keys in the same order in every language;
//                         app/tools/i18n_table.py shows them side by side)
//   lang/help.<l>.html    the longer ones (Help, the tips, the licences): one <template data-slot> per [data-slot]
//                         place of index.html
// Both are fetched at load (a module-level await: nothing that imports this module runs before the texts are here; a
// failed fetch is an error). The language follows the browser (the first of navigator.languages that OCR Lite speaks;
// none → English) unless Settings chose one (kept in this browser, "ocrlite.lang").
//   t(key, vars)      the text of `key` in the current language, {name} replaced from vars — a missing key or var throws;
//                     fixed / upTo: numbers for display in its way (translatorFrom below)
//   applyI18n(root)   the page's static texts: data-i18n (textContent), data-i18n-tip (the tip, tip.js), data-i18n-aria
//                     (aria-label),
//                     data-i18n-placeholder; [data-slot] from the language's prose; <html lang>
//   setChoice(c)      "browser" | "en" | "de": kept, applied at once, then "ocrlite:lang" on window (the other modules
//                     redraw what they wrote themselves)
// Every language must have the same keys and slots: checked at load, a difference stops the app.
import { setTip } from "./tip.js";

// a language's files (web/dist.py rewrites each path to its content-hashed name)
const FILES = {
  en: { texts: "./lang/en.json", prose: "./lang/help.en.html" },
  de: { texts: "./lang/de.json", prose: "./lang/help.de.html" },
};
export const LANGS = Object.keys(FILES);
export const CHOICES = ["browser", ...LANGS];
const KEY = "ocrlite.lang";

/// the same keys in every language ({ lang: { key: … } }), or an error naming the difference
export function checkKeys(byLang, what) {
  const [first, ...rest] = Object.keys(byLang);
  for (const l of rest) {
    const miss = Object.keys(byLang[first]).filter((k) => !(k in byLang[l]));
    const extra = Object.keys(byLang[l]).filter((k) => !(k in byLang[first]));
    if (miss.length > 0 || extra.length > 0) {
      throw new Error(`i18n: the ${what} of "${l}" differ from "${first}" — missing ${miss.join(", ") || "none"}; extra ${extra.join(", ") || "none"}`);
    }
  }
}

/// a translator for the texts `dict` of language `l`, pure (no page state — the device check's test in Node,
/// tests/gpu_check.mjs, builds one from lang/en.json):
///   tr(key, vars)   as t
///   tr.fixed(n, d)  a number shown with d decimals in l's way (0.51 → "0,51" in German) — for display only, never
///                   for parsing or inputs
///   tr.upTo(n, d)   at most d decimals, trailing zeros dropped
export function translatorFrom(dict, l) {
  const tr = (key, vars = {}) => {
    const s = dict[key];
    if (s === undefined) {
      throw new Error(`i18n: no text "${key}"`);
    }
    return s.replace(/\{(\w+)\}/g, (_, name) => {
      if (!(name in vars)) {
        throw new Error(`i18n: "${key}" needs {${name}}`);
      }
      return String(vars[name]);
    });
  };
  tr.lang = l;
  tr.fixed = (n, d) => n.toLocaleString(l, { minimumFractionDigits: d, maximumFractionDigits: d, useGrouping: false });
  tr.upTo = (n, d) => n.toLocaleString(l, { minimumFractionDigits: 0, maximumFractionDigits: d, useGrouping: false });
  return tr;
}

async function fetchText(path) {
  const r = await fetch(new URL(path, import.meta.url));
  if (!r.ok) {
    throw new Error(`i18n: ${path}: HTTP ${r.status}`);
  }
  return r.text();
}

/// every language's texts and prose ({ slot: html }), checked against each other
async function loadAll() {
  const texts = {};
  const prose = {};
  await Promise.all(LANGS.map(async (l) => {
    const [json, html] = await Promise.all([fetchText(FILES[l].texts), fetchText(FILES[l].prose)]);
    texts[l] = JSON.parse(json);
    const doc = new DOMParser().parseFromString(html, "text/html");
    prose[l] = Object.fromEntries([...doc.querySelectorAll("template[data-slot]")].map((el) => [el.dataset.slot, el.innerHTML]));
  }));
  checkKeys(texts, "texts");
  checkKeys(prose, "prose slots");
  return { texts, prose };
}

/// the browser's language: the first of navigator.languages OCR Lite speaks ("de-AT" → "de"); none → English
function browserLang() {
  for (const tag of navigator.languages) {
    const l = tag.toLowerCase().split("-")[0];
    if (LANGS.includes(l)) {
      return l;
    }
  }
  return "en";
}

/// the choice kept in this browser ("browser" when none is kept or storage is off)
function storedChoice() {
  let c = null;
  try {
    c = localStorage.getItem(KEY);
  } catch {}
  return c !== null && CHOICES.includes(c) ? c : "browser";
}

// the page: its texts loaded before anything that imports this module runs (Node — the gpu-check test — loads none)
const ON_PAGE = typeof document !== "undefined";
const ALL = ON_PAGE ? await loadAll() : null;
let current = "browser";
let tr = null;
function use(c) {
  current = c;
  const l = c === "browser" ? browserLang() : c;
  tr = translatorFrom(ALL.texts[l], l);
}

export const getLang = () => tr.lang;
export const getChoice = () => current;
/// the current translator (passed to web/gpu-check.js, which has no page language of its own)
export const getT = () => tr;
export const t = (key, vars = {}) => tr(key, vars);
export const fixed = (n, d) => tr.fixed(n, d);
export const upTo = (n, d) => tr.upTo(n, d);

export function applyI18n(root = document) {
  document.documentElement.lang = tr.lang;
  // the page shows once its texts are in (style.css: no English flashing up before German)
  document.documentElement.classList.add("i18n");
  for (const el of root.querySelectorAll("[data-i18n]")) {
    el.textContent = t(el.dataset.i18n);
  }
  for (const el of root.querySelectorAll("[data-i18n-tip]")) {
    setTip(el, t(el.dataset.i18nTip));
  }
  for (const el of root.querySelectorAll("[data-i18n-aria]")) {
    el.setAttribute("aria-label", t(el.dataset.i18nAria));
  }
  for (const el of root.querySelectorAll("[data-i18n-placeholder]")) {
    el.setAttribute(el.matches("input, textarea") ? "placeholder" : "data-placeholder", t(el.dataset.i18nPlaceholder));
  }
  const prose = ALL.prose[tr.lang];
  for (const el of root.querySelectorAll("[data-slot]")) {
    const html = prose[el.dataset.slot];
    if (html === undefined) {
      throw new Error(`i18n: no prose "${el.dataset.slot}"`);
    }
    el.innerHTML = html;
  }
}

if (ON_PAGE) {
  use(storedChoice());
  applyI18n();
}

export function setChoice(c) {
  if (!CHOICES.includes(c)) {
    throw new Error(`i18n: no language choice "${c}"`);
  }
  use(c);
  try {
    if (c === "browser") {
      localStorage.removeItem(KEY);
    } else {
      localStorage.setItem(KEY, c);
    }
  } catch {}
  applyI18n();
  window.dispatchEvent(new Event("ocrlite:lang"));
}
