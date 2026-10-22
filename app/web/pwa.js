// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The deployed browser build's service worker hook-up (dist only — web/dist.py links it into index.html):
// registers sw.js, and when a new deploy has been installed in the background, shows a small banner
// "New version available — Reload". Nothing is reloaded on its own: a running read is never interrupted; the click
// activates the worker waiting at that moment (postMessage skipWaiting) and reloads once it has taken over. The worker
// is looked up at the click, not when the banner appeared: a deploy that lands while the banner shows replaces the
// waiting worker, and the earlier one turns redundant (a message to it does nothing).
const $ = (id) => document.getElementById(id);

function reading() {
  return $("readBtn").classList.contains("stop");
}

// the reload of an update: the page's engine goes first (ocrlite:release → bridge-web.js ends the engine worker, its
// wasm heap and GPU device with it) — iOS counted the old page's memory on top of the new one's (the reload keeps the
// page's process)
function reloadNow() {
  window.dispatchEvent(new Event("ocrlite:release"));
  location.reload();
}

let wanted = false;
function offer(reg) {
  if ($("updBar")) return;
  const bar = Object.assign(document.createElement("div"), { id: "updBar", className: "updbar" });
  const text = Object.assign(document.createElement("span"), { textContent: "New version available" });
  const btn = Object.assign(document.createElement("button"), { type: "button", className: "btn small primary", textContent: "Reload" });
  const later = Object.assign(document.createElement("button"), { type: "button", className: "btn small", textContent: "Later" });
  btn.onclick = () => {
    if (reading()) { text.textContent = "A read is running — reload when it is done"; return; }
    if ($("editor").textContent.trim() && !confirm("Reload now? The text on the right is not saved.")) return;
    if (reg.waiting === null) { reloadNow(); return; } // already taken over (e.g. by another window)
    wanted = true;
    reg.waiting.postMessage("skipWaiting");
  };
  later.onclick = () => bar.remove();
  bar.append(text, btn, later);
  document.body.append(bar);
}

// "Refresh the app" (Settings, the crash dialog): the service worker reloads every file of this version from the
// server into a fresh cache (sw.js refresh), then the page reloads — shown only where a service worker is in charge
async function refreshApp(say) {
  if (reading()) { say("A read is running — refresh when it is done"); return; }
  if ($("editor").textContent.trim() && !confirm("Refresh the app now? The text on the right is not saved.")) return;
  const sw = navigator.serviceWorker.controller;
  say("loading the app's files anew…");
  const r = await new Promise((resolve) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = (e) => resolve(e.data);
    sw.postMessage("refresh", [ch.port2]);
  });
  if (!r.ok) {
    say(`The refresh failed: ${r.error} — a newer version may be on the server (reload when the banner shows)`);
    return;
  }
  reloadNow();
}

if ("serviceWorker" in navigator) {
  if (navigator.serviceWorker.controller) {
    $("appGroup").hidden = false;
    $("crashRefresh").hidden = false;
    $("appRefresh").onclick = () => refreshApp((t) => { $("appRefreshText").textContent = t; });
    $("crashRefresh").onclick = () => refreshApp((t) => { $("crashCheck").textContent = t; });
  }
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (wanted) reloadNow(); // only after the click (the first install also takes control: no reload then)
  });
  navigator.serviceWorker.register("sw.js").then((reg) => {
    if (reg.waiting && navigator.serviceWorker.controller) offer(reg);
    reg.addEventListener("updatefound", () => {
      const w = reg.installing;
      w.addEventListener("statechange", () => {
        if (w.state === "installed" && navigator.serviceWorker.controller) offer(reg);
      });
    });
    // a long-open tab looks for a new deploy now and then (and whenever it comes back to the front)
    setInterval(() => reg.update().catch(() => {}), 60 * 60 * 1000);
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") reg.update().catch(() => {});
    });
  }, (e) => console.warn(`service worker not registered: ${e.message}`));
}
