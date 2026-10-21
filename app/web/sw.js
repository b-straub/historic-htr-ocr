// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// OCR Lite's service worker (generated into dist by web/dist.py: VERSION + SHELL are filled in there).
//   app shell     precached at install: the content-hashed JS / CSS / WASM / icons + index.html
//   hashed files  cache-first (their names change with their content, so a cached one is never stale)
//   *.html, /     network-first, the precached copy offline
//   model         NEVER cached here: model/* and any .gguf, and every Range request, go straight to the network — the
//                 model lives in OPFS (or streams by ranges); 1.4 GB in the Cache API would double the storage
// A new deploy installs in the background and WAITS: the page shows "New version available — Reload" (pwa.js) and
// only the click sends "skipWaiting" — a running read is never interrupted.
const VERSION = "__VERSION__";
const SHELL = __SHELL__;
const CACHE = `ocrlite-${VERSION}`;
const HASHED = /\.[0-9a-f]{12}\.(js|css|wasm|png)$/;

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL.map((p) => new Request(p, { cache: "reload" })))));
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) {
      if (k.startsWith("ocrlite-") && k !== CACHE) await caches.delete(k);
    }
    await self.clients.claim();
  })());
});

self.addEventListener("message", (e) => {
  if (e.data === "skipWaiting") self.skipWaiting();
});

const scope = new URL(self.registration.scope);

async function cacheFirst(req) {
  const hit = await caches.match(req);
  if (hit) return hit;
  const r = await fetch(req);
  if (r.ok && r.status === 200) (await caches.open(CACHE)).put(req, r.clone());
  return r;
}

async function networkFirst(req, fallback) {
  try {
    return await fetch(req);
  } catch (err) {
    const hit = (await caches.match(req, { ignoreSearch: true })) ?? (fallback ? await caches.match(fallback) : undefined);
    if (hit) return hit;
    throw err;
  }
}

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || req.headers.has("range")) return;
  const url = new URL(req.url);
  if (url.origin !== scope.origin || !url.pathname.startsWith(scope.pathname)) return;
  const rel = url.pathname.slice(scope.pathname.length);
  if (rel.startsWith("model/") || rel.endsWith(".gguf")) return; // the model: network only, never cached
  if (HASHED.test(rel)) { e.respondWith(cacheFirst(req)); return; }
  // a page of the shell offline: itself, anything else: the app
  if (req.mode === "navigate") { e.respondWith(networkFirst(req, new URL(rel.endsWith(".html") && SHELL.includes(rel) ? rel : "index.html", scope).href)); return; }
  e.respondWith(networkFirst(req, null));
});
