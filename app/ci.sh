#!/bin/sh
# OCR Lite — Copyright (c) 2026 Bernhard Straub
# SPDX-License-Identifier: MIT
#
# THE build and test steps — the CI (.github/workflows/ci.yml) runs exactly these, and so does a local check:
#   app/ci.sh [test] [build] [site] [browser]     (no argument = all four, in this order)
#   test     cargo test: ocr-decode, ocr-core, ocr-gguf; ocr-wgpu's GPU tests only where a GPU adapter exists (the
#            adapter example decides; without one only its CPU tests run and the log says SKIPPED); the tests that need
#            the model only with OCR_LITE_MODEL_DIR. Then the Node unit tests (tests/: editor round trip, device check)
#   build    the web app: web/dist.sh → app/dist/ (wasm32 + wasm-bindgen-cli, hashed files)
#   site     the Pages artifact → <repo>/_site/: site/ at /, app/dist/ at /app/, the examples generated from
#            site/examples/ by hf/convert_output.py at /examples/; then every local link of the site is checked
#   browser  the browser tests on _site/ (headless Chrome, tests/browser/*.cjs; npm ci in tests/ first) — the parts that
#            need a GPU or the real model run only with OCR_LITE_MODEL_DIR and report SKIP otherwise
# Nothing here is GitHub-specific: no network except cargo / npm / the pages test's Hugging Face download.
set -eu
APP=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$APP")
SITE="$REPO/_site"
cd "$APP"

step_test() {
  echo "── test: cargo (CPU crates)"
  cargo test --locked -p ocr-decode -p ocr-core -p ocr-gguf
  echo "── test: cargo (ocr-wgpu)"
  cargo test --locked -p ocr-wgpu --no-run
  if cargo run --locked -q -p ocr-wgpu --example adapter; then
    if [ -n "${OCR_LITE_MODEL_DIR:-}" ]; then
      cargo test --locked -p ocr-wgpu -- --include-ignored
    else
      cargo test --locked -p ocr-wgpu
      echo "ocr-wgpu: the tokenizer tests need the model — IGNORED (set OCR_LITE_MODEL_DIR to run them)"
    fi
  else
    echo "::warning title=ocr-wgpu GPU tests SKIPPED::no GPU adapter on this machine — the GPU tests of ocr-wgpu were compiled, not run; only its CPU tests ran"
    cargo test --locked -p ocr-wgpu --lib -- compute_choice workgroup_bytes
  fi
  echo "── test: node"
  cargo build --locked -q -p ocr-core --example render
  (cd tests && [ -d node_modules ] || npm ci --ignore-scripts)
  node tests/editor_roundtrip.js target/debug/examples/render ui/editor.js
  node tests/gpu_check.mjs
}

step_build() {
  echo "── build: the web app → app/dist/"
  web/dist.sh
}

step_site() {
  echo "── site: the Pages artifact → _site/"
  [ -f dist/index.html ] || { echo "app/dist/ missing — run app/ci.sh build first" >&2; exit 1; }
  rm -rf "$SITE"
  mkdir -p "$SITE/examples"
  (cd "$REPO/site" && find . -path ./examples -prune -o -type f -print | while read -r f; do
    mkdir -p "$SITE/$(dirname "$f")"
    cp "$f" "$SITE/$f"
  done)
  cp -R dist "$SITE/app"
  # the examples: <name>.txt (the model's output) + <name>.jpg (the page) → one self-contained viewer page each
  grep -v '^#' "$REPO/site/examples/examples.tsv" | while IFS="$(printf '\t')" read -r name title; do
    [ -n "$name" ] || continue
    python3 -I "$REPO/hf/convert_output.py" "$REPO/site/examples/$name.txt" --to html --status raw \
      --image "$REPO/site/examples/$name.jpg" --embed --title "$title" \
      --base-ocr "LightOnOCR-3-1B · Historic German Handwriting (https://huggingface.co/b-straub/LightOnOCR-3-1B-Historic-German-Handwriting)" \
      > "$SITE/examples/$name.html"
    echo "  examples/$name.html"
  done
  # every local href / src of the site's HTML resolves to a file of _site
  python3 -I - "$SITE" <<'EOF'
import html.parser, os, sys, urllib.parse
site = sys.argv[1]
bad, n = [], 0
class Refs(html.parser.HTMLParser):
    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k in ("href", "src") and v:
                self.refs.append(v)
for dp, _, fs in os.walk(site):
    for f in fs:
        if not f.endswith(".html"):
            continue
        p = Refs()
        p.refs = []
        p.feed(open(os.path.join(dp, f), encoding="utf-8").read())
        for r in p.refs:
            u = urllib.parse.urlsplit(r)
            if u.scheme or u.netloc or r.startswith("#") or r.startswith("data:"):
                continue
            target = os.path.normpath(os.path.join(dp, urllib.parse.unquote(u.path)))
            if os.path.isdir(target):
                target = os.path.join(target, "index.html")
            n += 1
            if not os.path.isfile(target):
                bad.append(f"{os.path.relpath(os.path.join(dp, f), site)} → {r}")
if bad:
    sys.exit("site: broken local links:\n  " + "\n  ".join(bad))
print(f"site: {n} local links resolve")
EOF
}

step_browser() {
  echo "── browser: tests/browser on _site/"
  (cd tests && [ -d node_modules ] || npm ci --ignore-scripts)
  rc=0
  for t in ui setup state pages; do
    node "tests/browser/$t.cjs" || rc=1
  done
  return $rc
}

[ $# -gt 0 ] || set -- test build site browser
for s in "$@"; do
  case "$s" in
    test | build | site | browser) "step_$s" ;;
    *) echo "app/ci.sh: unknown step $s (test, build, site, browser)" >&2; exit 2 ;;
  esac
done
echo "app/ci.sh: $* — done"
