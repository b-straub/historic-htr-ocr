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
//   install  the detected platform's install sheet (installSheet: numbered steps in the browser's own words, the
//            vendor's help linked; the others folded); Chrome / Edge (computer and Android): their own install prompt
//            (beforeinstallprompt) as the button, the menu steps folded under it; then: open OCR Lite from the Dock /
//            Home Screen / apps, the setup goes on there (an installed app skips this step)
//   model    download from the model's Hugging Face repo (web/config.js) — on its own after the install step, the free
//            storage checked first — with progress and cancel; the user's own two files ("I have the files") only
//            where the host allows them (api.canUpload — web/config.js MODEL_SOURCES; switched off)
//   done     open an image and press Read
// Nothing is remembered: closed without a model, the setup is due again, and Read refuses with a link back here.
import { t, fixed } from "./i18n.js";

/// the platform for the install sheet, from the user agent + touch points (iPadOS Safari reports a Mac user agent: a
/// "Mac" with a touch screen is an iPad; any iPhone / iPad browser gets Safari's sheet) → "ios" | "mac-safari" | "chrome" |
/// "edge" | "android-chrome" | "android-edge" | "firefox" | "other" (chrome / edge: on a computer)
export function platformOf(ua, maxTouchPoints) {
  if (/iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && maxTouchPoints > 1)) {
    return "ios";
  }
  if (/Firefox\//.test(ua)) {
    return "firefox";
  }
  if (/OPR\/|SamsungBrowser\//.test(ua)) {
    return "other";
  }
  if (/Android/.test(ua)) {
    if (/EdgA\//.test(ua)) {
      return "android-edge";
    }
    return /Chrome\//.test(ua) ? "android-chrome" : "other";
  }
  if (/Edg\//.test(ua)) {
    return "edge";
  }
  if (/Chrome\//.test(ua)) {
    return "chrome";
  }
  if (/Macintosh/.test(ua) && /Version\/[\d.]+.*Safari\//.test(ua)) {
    return "mac-safari";
  }
  return "other";
}

// the install sheet of each platform (its texts: lang su.inst.<id>.*) — steps: [icon or null] per step (su.inst.<id>.<n>,
// the item's icon where the text says {icon}); note: a last hint (true; or its icon); byHand: iPhone / iPad (no direct
// install there: su.inst.iosByHand); bar: Chrome / Edge on a computer — the install item in the address bar (steps
// su.inst.<id>.bar.<n>; "chip": Chrome's "Install" chip drawn as it looks), the first way; prompt: Chrome / Edge install
// themselves (beforeinstallprompt) — the menu steps for when that does not come; where: what to open OCR Lite from
// afterwards; help: the vendor's official guide. none: this browser cannot install web apps (su.inst.<id>.text).
// Room here for a short screencast per platform later.
const HELP = {
  iphone: "https://support.apple.com/guide/iphone/iphea86e5236/ios",
  mac: "https://support.apple.com/guide/safari/ibrw9e991864/mac",
  chrome: "https://support.google.com/chrome/answer/9658361",
  edge: "https://support.microsoft.com/topic/install-manage-or-uninstall-apps-in-microsoft-edge-0c156575-a94a-45e4-a54f-3a84846f6113",
};
const SHEETS = {
  ios: { steps: ["hdots", "share", null, null, null], note: "share", byHand: true, where: "home", help: HELP.iphone, vendor: "Apple" },
  "mac-safari": { steps: ["share", null, null, null], note: true, where: "dock", help: HELP.mac, vendor: "Apple" },
  chrome: { bar: ["chip", null, null], prompt: true, steps: ["dots", null, null, null], note: false, where: "apps", help: HELP.chrome, vendor: "Google" },
  edge: { bar: ["appavail", null, null], prompt: true, steps: ["hdots", null, null, null], note: false, where: "apps", help: HELP.edge, vendor: "Microsoft" },
  "android-chrome": { prompt: true, steps: ["dots", null, null, null], note: false, where: "android", help: HELP.chrome, vendor: "Google" },
  "android-edge": { prompt: true, steps: ["hdots", null, null, null], note: false, where: "android", help: HELP.edge, vendor: "Microsoft" },
  firefox: { none: true },
  other: { none: true },
};
export const INSTALL_PLATFORMS = Object.keys(SHEETS);

// a text with the {icon} of its item inline (the icon painted by the caller: app.js paintIcons)
const ICON_MARK = "\u0000";
function withIcon(node, key, icon) {
  if (icon === null || icon === true || icon === false) {
    node.textContent = t(key);
    return node;
  }
  const [before, after] = t(key, { icon: ICON_MARK }).split(ICON_MARK);
  const i = Object.assign(document.createElement("i"), { className: "inl" });
  i.dataset.icon = icon === "chip" ? "installmon" : icon;
  if (icon === "chip") {
    // Chrome's address-bar chip: a light blue pill, the monitor icon + "Install" in the browser's language
    const chip = Object.assign(document.createElement("span"), { className: "instchip" });
    chip.append(i, t("su.inst.chip"));
    node.append(before, chip, after);
  } else {
    node.append(before, i, after);
  }
  return node;
}

/// the install sheet of platform `id`: title, the numbered steps, the note, the vendor's help. Chrome / Edge: on a
/// computer the address bar's install item first, then (button: the prompt was caught) "or here:" + our Install button;
/// on Android the button first; the menu steps folded under them (open when neither is there). folded: the menu steps
/// closed; help: in the Help dialog (no setup button — the menu steps under a plain heading)
export function installSheet(id, { button = null, folded = false, help = false } = {}) {
  const s = SHEETS[id];
  const el = (tag, cls, text = "") => Object.assign(document.createElement(tag), { className: cls, textContent: text });
  const d = el("div", "instsheet");
  d.append(el("b", "insttitle", t(`su.inst.${id}.title`)));
  if (s.none) {
    d.append(el("p", "", t(`su.inst.${id}.text`)));
    return d;
  }
  const ol = el("ol", "inststeps");
  s.steps.forEach((icon, i) => ol.append(withIcon(el("li", ""), `su.inst.${id}.${i + 1}`, icon)));
  if (s.prompt) {
    const bar = s.bar === undefined ? null : el("ol", "inststeps");
    if (bar !== null) {
      s.bar.forEach((icon, i) => bar.append(withIcon(el("li", ""), `su.inst.${id}.bar.${i + 1}`, icon)));
      d.append(bar);
    }
    if (help) {
      d.append(el("p", "", t("su.inst.promptHelp")));
    }
    if (button !== null) {
      if (bar !== null) {
        d.append(el("p", "", t("su.inst.orHere")));
      }
      d.append(button);
    }
    const more = el("details", "instmore");
    more.open = !folded && bar === null && button === null;
    const summary = help ? "su.inst.byMenu" : bar !== null ? "su.inst.ifNoChip" : button !== null ? "su.inst.ifNoButton" : "su.inst.noPrompt";
    more.append(el("summary", "", t(summary)), ol);
    d.append(more);
  } else {
    d.append(ol);
  }
  if (s.note !== false) {
    d.append(withIcon(el("p", "hint"), `su.inst.${id}.note`, s.note));
  }
  if (s.byHand === true) {
    d.append(el("p", "hint", t("su.inst.iosByHand")));
  }
  const a = el("a", "insthelp", t("su.inst.official", { vendor: s.vendor }));
  Object.assign(a, { href: s.help, target: "_blank", rel: "noopener" });
  d.append(a);
  return d;
}

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
export function setupDialog(api, { pickImage, openHelp, paintIcons }) {
  const $ = (id) => document.getElementById(id);
  const dlg = $("setupDlg");
  const platform = platformOf(navigator.userAgent, navigator.maxTouchPoints);
  // the browser's own Install (Chrome / Edge): moved into this platform's sheet while offered — kept by reference
  const installBtn = $("suInstallBtn");
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
    const none = SHEETS[platform].none === true;
    $("suNoInstall").hidden = !none;
    $("suNoInstall").textContent = none ? t(`su.inst.${platform}.text`) : "";
    $("suToInstall").hidden = none;
    $("suInBrowser").textContent = t(none ? "su.next" : "su.inBrowser");
    $("suInBrowser").classList.toggle("primary", none);
  }
  function installStep() {
    show("install");
    $("suInstalled").textContent = "";
    drawSheets();
  }
  // this platform's sheet (the browser's own Install button first where it offers one), the others folded; drawn again
  // when the prompt comes late or goes
  function drawSheets() {
    const offered = installPrompt !== null;
    installBtn.hidden = !offered;
    $("suHint").replaceChildren(installSheet(platform, { button: offered ? installBtn : null, folded: offered }));
    $("suOtherHints").replaceChildren(...INSTALL_PLATFORMS.filter((k) => k !== platform).map((k) => installSheet(k, { folded: true })));
    $("suThen").textContent = t("su.inst.then", { where: t(`su.inst.where.${SHEETS[platform].where}`) });
    paintIcons(dlg);
  }
  function drawInstallBtn() {
    if (!dlg.querySelector('[data-step="install"]').hidden) {
      drawSheets();
    }
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
  installBtn.onclick = async () => {
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
    let missing;
    try {
      missing = await api.modelMissing();
    } catch (e) {
      $("suModelErr").textContent = t("st.storage", { e });
      return;
    }
    if (!missing) {
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
    let missing;
    try {
      missing = await api.modelMissing();
    } catch (e) {
      $("suModelErr").textContent = t("st.storage", { e });
      return;
    }
    if (missing) {
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
