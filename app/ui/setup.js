// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// OCR Lite's first-run setup — a dialog shown at start while the model is not in this browser's storage, and from the
// help (Run setup again); an average user should get from a fresh visit to a first read without help. The host side
// (download, upload, progress, cancel) is bridge-web.js `setup`.
//   gpu      the device check FIRST (the host's gpuCheck = web/gpu-check.js): green → Continue; amber (slow) → the
//            hints + Continue anyway; red (no WebGPU, no adapter, software rendering, limits short) → the hints, no way
//            on (Help stays reachable). Run again each time the setup opens, and by Help → Check this device (then
//            Continue closes)
//   where    in a browser tab only: install as an app first (recommended) or continue here — an installed web app has
//            its OWN storage, so the model must be downloaded inside the app that will be used
//   install  the detected platform's install hint (the others folded); Chrome / Edge: a real Install button
//            (beforeinstallprompt)
//   model    download from the model's Hugging Face repo (web/config.js) — on its own after the install step, the free
//            storage checked first — with progress and cancel; the user's own two files ("I have the files") only
//            where the host allows them (api.canUpload — web/config.js MODEL_SOURCES; switched off)
//   done     open an image and press Read
// Nothing is remembered: closed without a model, the setup is due again, and Read refuses with a link back here.
import { t, fixed } from "./i18n.js";

/// the platform for the install hint, from the user agent + touch points (iPadOS Safari reports a Mac user agent: a
/// "Mac" with a touch screen is an iPad) → "ios" | "mac-safari" | "chrome" | "edge" | "firefox" | "other"
export function platformOf(ua, maxTouchPoints) {
  if (/iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && maxTouchPoints > 1)) {
    return "ios";
  }
  if (/Firefox\//.test(ua)) {
    return "firefox";
  }
  if (/Edg\//.test(ua)) {
    return "edge";
  }
  if (/OPR\/|SamsungBrowser\//.test(ua)) {
    return "other";
  }
  if (/Chrome\//.test(ua)) {
    return "chrome";
  }
  if (/Macintosh/.test(ua) && /Version\/[\d.]+.*Safari\//.test(ua)) {
    return "mac-safari";
  }
  return "other";
}

// one short block per platform (its text: i18n.js su.hint.<id>); `none`: this browser cannot install web apps
const HINTS = {
  ios: { title: () => "iPad / iPhone" },
  "mac-safari": { title: () => "Mac · Safari" },
  chrome: { title: () => "Chrome" },
  edge: { title: () => "Edge" },
  firefox: { title: () => "Firefox", none: true },
  other: { title: () => t("su.hint.other.title"), none: true },
};
const hintText = (id) => t(`su.hint.${id}`);

// Chrome / Edge offer their install prompt once the page qualifies (manifest + service worker): kept for the Install
// button. Listened for at import — app.js imports this module before anything else runs.
let installPrompt = null;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  installPrompt = e;
  window.dispatchEvent(new Event("ocrlite:installable"));
});

/// running as an installed app (Home Screen / Dock / Chrome or Edge app), not in a browser tab
function standalone() {
  return window.matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
}

const gb = (b) => `${fixed(b / 1e9, 1)} GB`;

/// the setup dialog for the host's `setup` (bridge-web.js); pickImage = app.js's Open, openHelp = the help dialog →
/// { open(), check() }
export function setupDialog(api, { pickImage, openHelp }) {
  const $ = (id) => document.getElementById(id);
  const dlg = $("setupDlg");
  const platform = platformOf(navigator.userAgent, navigator.maxTouchPoints);
  let busy = false; // the model is being stored: closing cancels it
  let checkOnly = false; // the device check opened from Help: Continue closes instead of going on with the setup
  let checkRun = 0; // the device check running now (a newer one makes an older one's result stale)
  // the user's own model files: offered only where the host allows them (bridge-web.js MODEL_SOURCES)
  $("suFilesBtn").hidden = !api.canUpload;

  function show(step) {
    for (const s of dlg.querySelectorAll("[data-step]")) {
      s.hidden = s.dataset.step !== step;
    }
    // the step row: the ones before this one done, this one current (the install step belongs to "where")
    const order = ["gpu", "where", "model", "done"];
    const at = order.indexOf(step === "install" ? "where" : step);
    for (const li of $("suSteps").children) {
      const i = order.indexOf(li.dataset.for);
      li.className = i < at ? "done" : i === at ? "on" : "";
    }
  }

  function hintBlock(id) {
    const h = HINTS[id];
    const d = document.createElement("div");
    d.className = "suhint";
    d.append(Object.assign(document.createElement("b"), { textContent: h.title() }), Object.assign(document.createElement("p"), { textContent: hintText(id) }));
    return d;
  }

  // ── gpu: the device check ──
  const el = (tag, cls, text) => Object.assign(document.createElement(tag), { className: cls, textContent: text });
  function drawCheck(r) {
    const tone = { ok: "ok", slow: "slow", reject: "bad", pending: "" }[r.verdict];
    $("suGpuHead").className = `suverdict ${tone}`;
    $("suGpuHead").textContent = r.headline;
    $("suGpuDetected").replaceChildren(...r.detected.map((t) => el("li", "", t)));
    $("suGpuProblems").replaceChildren(...r.problems.map((t) => el("li", "", t)));
    $("suGpuNotes").replaceChildren(...r.notes.map((t) => el("p", "hint", t)));
    $("suGpuHints").replaceChildren(...r.hints.map((h) => {
      const d = el("div", "suhint", "");
      d.append(el("b", "", h.title), el("p", "", h.text));
      return d;
    }));
    const done = r.verdict !== "pending";
    $("suGpuAgain").disabled = !done;
    $("suGpuNext").hidden = r.verdict !== "ok";
    $("suGpuAnyway").hidden = r.verdict !== "slow";
    $("suGpuNext").textContent = t(checkOnly ? "close" : "su.next");
    $("suGpuAnyway").textContent = t(checkOnly ? "close" : "su.anyway");
  }
  async function gpuStep() {
    show("gpu");
    const run = ++checkRun;
    const last = api.lastGpuCheck();
    $("suGpuLast").textContent = last === null ? "" : t("su.lastCheck", { time: new Date(last.time).toLocaleString(), headline: last.headline });
    drawCheck({ verdict: "pending", headline: t("su.checking"), detected: [], problems: [], notes: [], hints: [] });
    let r;
    try {
      r = await api.gpuCheck((step) => {
        if (run === checkRun) {
          drawCheck(step);
        }
      });
    } catch (e) {
      r = { verdict: "reject", headline: t("su.checkFailed"), detected: [], problems: [e instanceof Error ? e.message : String(e)], notes: [], hints: [] };
    }
    if (run !== checkRun) {
      return;
    }
    $("suGpuLast").textContent = "";
    drawCheck(r);
    if (r.verdict === "ok") {
      $("suGpuNext").focus();
    }
  }
  function afterCheck() {
    if (checkOnly) {
      dlg.close();
      return;
    }
    if (standalone()) {
      afterWhere();
    } else {
      whereStep();
    }
  }
  // after the device check (an installed app) or the install choice: the model
  function afterWhere() {
    modelStep(true);
  }
  $("suGpuAgain").onclick = () => gpuStep();
  $("suGpuNext").onclick = afterCheck;
  $("suGpuAnyway").onclick = afterCheck;
  $("suGpuHelp").onclick = () => {
    dlg.close();
    openHelp();
  };

  // ── where / install ──
  function whereStep() {
    show("where");
    const none = HINTS[platform].none === true;
    $("suNoInstall").hidden = !none;
    $("suNoInstall").textContent = hintText(platform);
    $("suToInstall").hidden = none;
    $("suInBrowser").textContent = t(none ? "su.next" : "su.inBrowser");
    $("suInBrowser").classList.toggle("primary", none);
  }
  function installStep() {
    show("install");
    $("suHint").replaceChildren(hintBlock(platform));
    $("suOtherHints").replaceChildren(...Object.keys(HINTS).filter((k) => k !== platform).map(hintBlock));
    $("suInstalled").textContent = "";
    drawInstallBtn();
  }
  function drawInstallBtn() {
    $("suInstallBtn").hidden = installPrompt === null;
  }
  window.addEventListener("ocrlite:installable", drawInstallBtn);
  window.addEventListener("appinstalled", () => {
    installPrompt = null;
    drawInstallBtn();
    $("suInstalled").textContent = t("su.installed");
  });
  $("suToInstall").onclick = installStep;
  $("suInBrowser").onclick = () => afterWhere();
  $("suInstallBack").onclick = whereStep;
  $("suInstallSkip").onclick = () => afterWhere();
  $("suInstallBtn").onclick = async () => {
    const p = installPrompt;
    installPrompt = null;
    drawInstallBtn();
    await p.prompt();
    const choice = await p.userChoice;
    if (choice.outcome === "accepted") {
      $("suInstalled").textContent = t("su.installed");
    }
  };

  // ── model ──
  function setBusy(on) {
    busy = on;
    $("suDownload").hidden = on;
    $("suFilesBtn").classList.toggle("disabled", on);
    $("suFiles").disabled = on;
    $("suCancel").hidden = !on;
    $("suProgress").hidden = !on;
  }
  async function modelStep(auto) {
    show("model");
    $("suModelErr").textContent = "";
    $("suProgressText").textContent = "";
    $("suNext").hidden = true;
    $("suDownload").textContent = t("su.download");
    if (!(await api.modelMissing())) {
      $("suModelNote").textContent = t("su.haveModel");
      $("suDownload").textContent = t("su.downloadAgain");
      $("suNext").hidden = false;
      return;
    }
    $("suModelNote").textContent = t("su.modelNote", { name: api.modelName, from: api.modelFrom });
    const room = await api.room();
    if (room !== null && room.free < room.need) {
      $("suModelErr").textContent = t("su.noRoom", { free: gb(room.free), need: gb(room.need) });
      return;
    }
    if (auto) {
      store(() => api.download(drawProgress));
    }
  }
  function drawProgress(p) {
    $("suProgress").value = p.fraction;
    $("suProgressText").textContent = p.line;
  }
  async function store(job) {
    setBusy(true);
    $("suModelErr").textContent = "";
    $("suProgress").value = 0;
    $("suProgressText").textContent = t("su.starting");
    try {
      await job();
    } catch (e) {
      setBusy(false);
      $("suProgressText").textContent = "";
      const msg = e instanceof Error ? e.message : String(e);
      $("suModelErr").textContent = msg === "download cancelled" ? t("su.cancelled") : msg;
      return;
    }
    setBusy(false);
    if (await api.modelMissing()) {
      $("suModelErr").textContent = t("su.missing", { files: api.modelFiles.join(" + ") });
      return;
    }
    show("done");
  }
  $("suDownload").onclick = () => store(() => api.download(drawProgress));
  $("suCancel").onclick = () => api.cancel();
  $("suNext").onclick = () => show("done");
  $("suFiles").onchange = () => {
    const files = [...$("suFiles").files];
    $("suFiles").value = "";
    if (files.length) {
      store(() => api.upload(files, drawProgress));
    }
  };

  // ── done ──
  $("suOpen").onclick = () => {
    dlg.close();
    pickImage();
  };
  $("suDone").onclick = () => dlg.close();

  // closing: Esc does nothing while the model is stored; the close button then cancels the download first
  dlg.addEventListener("cancel", (e) => {
    if (busy) {
      e.preventDefault();
    }
  });
  $("suClose").onclick = () => {
    if (busy) {
      api.cancel();
    }
    dlg.close();
  };

  return {
    /// open the setup at its start — the device check, then the install choice (in a tab) and the model; a running
    /// download is shown as it is
    open() {
      if (!dlg.open) {
        dlg.showModal();
      }
      if (busy) {
        return;
      }
      checkOnly = false;
      gpuStep();
    },
    /// the device check alone (Help → Check this device): Continue closes; during a download the download is shown
    check() {
      if (!dlg.open) {
        dlg.showModal();
      }
      if (busy) {
        return;
      }
      checkOnly = true;
      gpuStep();
    },
  };
}
