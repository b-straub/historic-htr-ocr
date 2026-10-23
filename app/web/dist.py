#!/usr/bin/env python3
# OCR Lite — Copyright (c) 2026 Bernhard Straub
# SPDX-License-Identifier: MIT
#
# OCR Lite's deployable web build (called by web/dist.sh after web/build.sh): a flat static site (default app/dist/,
# generated, not in git) from the sources — ui/ + web/bridge-web.js, config.js, gpu-check.js, worker.js + pkg/
# (wasm-bindgen output) + web/sw.js, pwa.js + web/icons/.
#   * JS / CSS / WASM / icons get content-hashed names (name.<12 hex>.ext) and every reference to them is rewritten —
#     each reference must be found exactly once, or the build stops (a missed one would load a stale or missing file)
#   * index.html / sw.js / manifest.webmanifest keep their names
#   * GitHub Pages sends no custom headers: the Content-Security-Policy goes into index.html as a <meta>; web/htaccess
#     goes out as .htaccess (the headers for an Apache host — Pages ignores it)
#   * every built JS / CSS file carries the licence line (the sources' header; a /*! … */ banner where a file has none,
#     e.g. wasm-bindgen's glue) — checked at the end, like `node --check` on every JS file
#   web/dist.py [--out DIR]
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

WEB = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(WEB)
UI = os.path.join(APP, "ui")
PKG = os.path.join(WEB, "pkg")
ICONS = os.path.join(WEB, "icons")

LICENCE = "SPDX-License-Identifier: MIT"
BANNER = f"/*! OCR Lite — Copyright (c) 2026 Bernhard Straub · {LICENCE} */\n"
MANIFEST_DESCRIPTION = "Reads historic handwriting on this device (WebGPU) — the open model from Hugging Face, nothing leaves the machine."
# no server config on GitHub Pages: the policy as a <meta>. connect-src: this site + Hugging Face (the model download:
# huggingface.co answers with a redirect to its CDN hosts)
CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' blob: data:; connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co; "
       "font-src 'self'; manifest-src 'self'; object-src 'none'; base-uri 'self'; form-action 'none'")


def die(msg):
    print(f"dist: {msg}", file=sys.stderr)
    sys.exit(1)


def read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def replace_once(text, old, new, where):
    n = text.count(old)
    if n != 1:
        die(f"{where}: expected the reference {old!r} exactly once, found {n} — update web/dist.py with the source")
    return text.replace(old, new)


def with_licence(text):
    """a JS / CSS file with the licence line near its top (a banner prepended when it has none)"""
    return text if LICENCE in text[:400] else BANNER + text


class Site:
    def __init__(self, out):
        self.out = out
        self.hashed = {}  # logical name → published (hashed) name

    def put(self, name, data):
        path = os.path.join(self.out, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode("utf-8"))

    def put_hashed(self, logical, data):
        if not isinstance(data, bytes) and os.path.splitext(logical)[1] in (".js", ".css"):
            data = with_licence(data)
        b = data if isinstance(data, bytes) else data.encode("utf-8")
        stem, ext = os.path.splitext(logical)
        name = f"{stem}.{hashlib.sha256(b).hexdigest()[:12]}{ext}"
        self.put(name, b)
        self.hashed[logical] = name
        return name


def theme_colours(css):
    """the page background (--bg) of the light :root block and of the dark theme block in ui/style.css"""
    light = re.search(r":root\s*\{[^}]*?--bg:\s*(#[0-9a-fA-F]{6})", css)
    dark = re.search(r':root\[data-theme="dark"\]\s*\{[^}]*?--bg:\s*(#[0-9a-fA-F]{6})', css)
    if not light or not dark:
        die("ui/style.css: no --bg colour found in :root / :root[data-theme=\"dark\"]")
    return light.group(1), dark.group(1)


def git(*args):
    r = subprocess.run(["git", "-C", APP, *args], capture_output=True, text=True)
    if r.returncode != 0:
        die(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def app_version():
    """the app's version = the release tag: on GitHub the tag that started the run (GITHUB_REF_TYPE tag, also inside the
    CI called by pages.yml), locally a tag on HEAD; otherwise "untagged" """
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        return os.environ["GITHUB_REF_NAME"]
    tags = git("tag", "--points-at", "HEAD").split()
    if len(tags) > 1:
        die(f"HEAD has more than one tag: {' '.join(tags)}")
    return tags[0] if tags else "untagged"


def build_stamp(version):
    """version / commit / build time for the About box: the commit of app/, "-dirty" when it has uncommitted changes,
    unix seconds of this build"""
    commit = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "--", ".") != ""
    return {"version": version, "git": commit + ("-dirty" if dirty else ""), "built": int(time.time())}


# a test build's model folder (--model-base): the two model files on the same web space as the app — the CSP allows
# only the app's own origin and Hugging Face, and a download from there does not count on the Hub
TEST_MODEL_FROM = "this server (test build)"


def build(out, model_base=None):
    s = Site(out)
    version = app_version()
    stamp = build_stamp(version)
    if model_base is not None:
        stamp["modelBase"] = model_base

    # the engine: wasm ← its JS glue ← the worker ← the web bridge ← app.js
    wasm = s.put_hashed("ocr_web_bg.wasm", read(os.path.join(PKG, "ocr_web_bg.wasm"), "rb"))
    glue = replace_once(read(os.path.join(PKG, "ocr_web.js")), "'ocr_web_bg.wasm'", f"'{wasm}'", "pkg/ocr_web.js")
    glue = s.put_hashed("ocr_web.js", glue)
    worker = replace_once(read(os.path.join(WEB, "worker.js")), '"./pkg/ocr_web.js"', f'"./{glue}"', "web/worker.js")
    worker = s.put_hashed("worker.js", worker)
    imager = replace_once(read(os.path.join(WEB, "image-worker.js")), '"./pkg/ocr_web.js"', f'"./{glue}"', "web/image-worker.js")
    imager = s.put_hashed("image-worker.js", imager)
    gpucheck = s.put_hashed("gpu-check.js", read(os.path.join(WEB, "gpu-check.js")))
    config = read(os.path.join(WEB, "config.js"))
    if model_base is not None:
        config = replace_once(config, "export const MODEL_BASE = `${MODEL.repo}/resolve/main/`;", f'export const MODEL_BASE = "{model_base}";', "web/config.js")
        config = replace_once(config, 'export const MODEL_FROM = "Hugging Face";', f'export const MODEL_FROM = "{TEST_MODEL_FROM}";', "web/config.js")
        config = replace_once(config, "export const MODEL_TEST = false;", "export const MODEL_TEST = true;", "web/config.js")
    config = s.put_hashed("config.js", config)
    bridge = read(os.path.join(WEB, "bridge-web.js"))
    bridge = replace_once(bridge, '"./gpu-check.js"', f'"./{gpucheck}"', "web/bridge-web.js")
    bridge = replace_once(bridge, '"./config.js"', f'"./{config}"', "web/bridge-web.js")
    bridge = replace_once(bridge, '"./pkg/ocr_web.js"', f'"./{glue}"', "web/bridge-web.js")
    bridge = replace_once(bridge, '"./worker.js"', f'"./{worker}"', "web/bridge-web.js")
    bridge = replace_once(bridge, '"./image-worker.js"', f'"./{imager}"', "web/bridge-web.js")
    bridge = s.put_hashed("bridge-web.js", bridge)
    scan = s.put_hashed("scan.js", read(os.path.join(UI, "scan.js")))
    setup = s.put_hashed("setup.js", read(os.path.join(UI, "setup.js")))
    app = read(os.path.join(UI, "app.js"))
    app = replace_once(app, '"./scan.js"', f'"./{scan}"', "ui/app.js")
    app = replace_once(app, '"./setup.js"', f'"./{setup}"', "ui/app.js")
    app = replace_once(app, '"./bridge-web.js"', f'"./{bridge}"', "ui/app.js")
    app = s.put_hashed("app.js", app)
    editor = s.put_hashed("editor.js", read(os.path.join(UI, "editor.js")))
    css = read(os.path.join(UI, "style.css"))
    style = s.put_hashed("style.css", css)
    render = s.put_hashed("render.css", read(os.path.join(UI, "render.css")))
    pwa = s.put_hashed("pwa.js", read(os.path.join(WEB, "pwa.js")))

    # icons + manifest (Add to Home Screen: Chrome / Edge install, iPadOS Safari via apple-touch-icon + the meta tags)
    icons = {n: s.put_hashed(f"icon-{n}.png", read(os.path.join(ICONS, f"icon-{n}.png"), "rb")) for n in (180, 192, 512)}
    light, dark = theme_colours(css)
    manifest = {
        "name": "OCR Lite",
        "short_name": "OCR Lite",
        "description": MANIFEST_DESCRIPTION,
        "id": "./",
        "start_url": "./",
        "scope": "./",
        "display": "standalone",
        "background_color": light,
        "theme_color": light,
        "icons": [{"src": icons[n], "sizes": f"{n}x{n}", "type": "image/png", "purpose": "any"} for n in (192, 512)],
    }
    s.put("manifest.webmanifest", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    # index.html: the hashed files + the PWA head + pwa.js (registers sw.js, the "new version" banner)
    html = read(os.path.join(UI, "index.html"))
    html = replace_once(html, 'href="style.css"', f'href="{style}"', "ui/index.html")
    html = replace_once(html, 'href="render.css"', f'href="{render}"', "ui/index.html")
    html = replace_once(html, 'src="editor.js"', f'src="{editor}"', "ui/index.html")
    html = replace_once(html, 'src="app.js"', f'src="{app}"', "ui/index.html")
    head = "\n".join([
        '  <link rel="manifest" href="manifest.webmanifest">',
        f'  <link rel="icon" type="image/png" href="{icons[192]}">',
        f'  <link rel="apple-touch-icon" href="{icons[180]}">',
        f'  <meta name="theme-color" content="{light}" media="(prefers-color-scheme: light)">',
        f'  <meta name="theme-color" content="{dark}" media="(prefers-color-scheme: dark)">',
        '  <meta name="mobile-web-app-capable" content="yes">',
        '  <meta name="apple-mobile-web-app-capable" content="yes">',
        '  <meta name="apple-mobile-web-app-title" content="OCR Lite">',
        '  <meta name="apple-mobile-web-app-status-bar-style" content="default">',
        # the About box reads this (bridge-web.js buildInfo); index.html is revalidated on every visit
        f"  <meta name=\"ocrlite-build\" content='{json.dumps(stamp)}'>",
        f'  <meta http-equiv="Content-Security-Policy" content="{CSP}">',
        "</head>",
    ])
    html = replace_once(html, "</head>", head, "ui/index.html")
    html = replace_once(html, "</body>", f'  <script type="module" src="{pwa}"></script>\n</body>', "ui/index.html")
    s.put("index.html", html)

    # the service worker: the app shell to precache + a version that changes with any of it
    shell = ["index.html", "manifest.webmanifest", *sorted(s.hashed.values())]
    digest = hashlib.sha256()
    for name in shell:
        digest.update(name.encode())
        digest.update(read(os.path.join(out, name), "rb"))
    sw = read(os.path.join(WEB, "sw.js"))
    sw = replace_once(sw, '"__VERSION__"', json.dumps(f"{version}-{digest.hexdigest()[:12]}"), "web/sw.js")
    sw = replace_once(sw, "__SHELL__", json.dumps(shell), "web/sw.js")
    s.put("sw.js", with_licence(sw))
    s.put(".htaccess", read(os.path.join(WEB, "htaccess")))

    # no source name may be left referenced where a hashed one belongs
    refs = ["index.html", s.hashed["app.js"], s.hashed["bridge-web.js"], s.hashed["worker.js"], s.hashed["ocr_web.js"]]
    for name in refs:
        text = read(os.path.join(out, name))
        for logical in s.hashed:
            if re.search(r"""["'/]""" + re.escape(logical) + r"""["']""", text):
                die(f"{name} still references {logical} by its source name")
    check_js(out, [n for n in s.hashed.values() if n.endswith(".js")] + ["sw.js"])
    check_licence(out)
    return version, shell


def check_js(out, names):
    """every JS file of the build parses (node --check, as an ES module)"""
    if shutil.which("node") is None:
        die("node not found — needed to check the built JS (node --check)")
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            mjs = os.path.join(tmp, os.path.splitext(name)[0] + ".mjs")
            shutil.copyfile(os.path.join(out, name), mjs)
            r = subprocess.run(["node", "--check", mjs], capture_output=True, text=True)
            if r.returncode != 0:
                die(f"{name} does not parse:\n{r.stderr.strip()}")


def check_licence(out):
    """every built JS / CSS file carries the licence line near its top"""
    missing = [f for f in sorted(os.listdir(out)) if f.endswith((".js", ".css")) and LICENCE not in read(os.path.join(out, f))[:400]]
    if missing:
        die(f"no licence line in: {', '.join(missing)}")


def main():
    ap = argparse.ArgumentParser(description="OCR Lite: assemble the deployable web build")
    ap.add_argument("--out", default=os.path.join(APP, "dist"), help="default: app/dist/")
    ap.add_argument("--model-base", default=None,
                    help="a TEST build: the model folder on the app's own web space (relative to the app, e.g. ../model/, "
                         "or an absolute path /…/model/) instead of Hugging Face — the two .gguf files are put there by hand")
    args = ap.parse_args()
    model_base = args.model_base
    if model_base is not None:
        if "://" in model_base or model_base.startswith("//"):
            die(f"--model-base {model_base}: a path on the app's own web space (the CSP allows only its own origin and Hugging Face)")
        if not model_base.endswith("/"):
            model_base += "/"
    for f in ("ocr_web_bg.wasm", "ocr_web.js"):
        if not os.path.isfile(os.path.join(PKG, f)):
            die(f"web/pkg/{f} missing — run web/build.sh")
    out = os.path.abspath(args.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    version, shell = build(out, model_base)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
    print(f"dist: OCR Lite {version} → {out} ({len(shell)} shell files, {size / 1e6:.1f} MB; no model in it)")
    if model_base is not None:
        print(f"dist: TEST build — the model comes from {model_base} on the app's web space, not from Hugging Face")


if __name__ == "__main__":
    main()
