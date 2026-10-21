#!/usr/bin/env python3
# OCR Lite — Copyright (c) 2026 Bernhard Straub
# SPDX-License-Identifier: MIT
#
# OCR Lite's local server for the browser build. Two modes:
#   dev (default)   the sources as they are, no copies: a path is looked up in web/ first, then in ui/; every response
#                   carries COOP / COEP and no-store (a rebuilt wasm is always picked up)
#   --dist DIR      a built site (web/dist.sh output, or the whole Pages artifact) served as GitHub Pages serves it: DIR
#                   is the only root, no extra headers, Cache-Control no-cache (the service worker caches by itself);
#                   --prefix PATH serves it under PATH (e.g. /historic-htr-ocr/, the Pages sub-path)
# In both modes, at the root:
#   /model/<text-q8_0.gguf | mmproj-bf16.gguf>   the model files from --model-dir, with HTTP Range (206) — the app reads
#                                                them with ?modelBase=/model/ (honoured on localhost only), the selftest
#                                                streams them; 404 without --model-dir
#   /data/<file>                                 files from --data-dir (test images, expected reads), if given
#   POST /log                                    one line of text printed here (the selftest's report in headless runs)
#   web/serve.py [--port 8765] [--model-dir DIR] [--data-dir DIR] [--dist DIR [--prefix PATH]]
import argparse
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTS = [HERE, os.path.join(os.path.dirname(HERE), "ui")]
MODEL_FILES = {"text-q8_0.gguf", "mmproj-bf16.gguf"}
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".wasm": "application/wasm",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".txt": "text/plain; charset=utf-8",
    ".gguf": "application/octet-stream",
    ".webmanifest": "application/manifest+json",
}
CHUNK = 1 << 20


def resolve(url_path, args):
    """the file a URL path names, or None (no '..', no hidden files)"""
    path = url_path.split("?", 1)[0]
    parts = [p for p in path.split("/") if p]
    if any(p.startswith(".") for p in parts):
        return None
    if parts[:1] == ["model"] and len(parts) == 2 and parts[1] in MODEL_FILES:
        return os.path.join(args.model_dir, parts[1]) if args.model_dir else None
    if parts[:1] == ["data"] and len(parts) == 2 and args.data_dir:
        return os.path.join(args.data_dir, parts[1])
    if args.dist and args.prefix:
        prefix = [p for p in args.prefix.split("/") if p]
        if parts[:len(prefix)] != prefix:
            return None
        parts = parts[len(prefix):]
    rel = os.path.join(*parts) if parts else "index.html"
    for root in ([args.dist] if args.dist else ROOTS):
        f = os.path.join(root, rel)
        if os.path.isdir(f):
            f = os.path.join(f, "index.html")
        if os.path.isfile(f):
            return f
    return None


class Handler(BaseHTTPRequestHandler):
    def end_headers(self):
        if self.server.args.dist:
            self.send_header("Cache-Control", "no-cache")
        else:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path != "/log":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8", "replace")
        print(body, flush=True)
        self.send_response(204)
        self.end_headers()

    def do_HEAD(self):
        self.serve(head=True)

    def do_GET(self):
        self.serve(head=False)

    def serve(self, head):
        f = resolve(self.path, self.server.args)
        if f is None or not os.path.isfile(f):
            self.send_error(404)
            return
        size = os.path.getsize(f)
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng:
            m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng.strip())
            if not m or int(m.group(1)) >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", TYPES.get(os.path.splitext(f)[1], "application/octet-stream"))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if head:
            return
        with open(f, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                b = fh.read(min(CHUNK, left))
                if not b:
                    break
                try:
                    self.wfile.write(b)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(b)

    def log_message(self, fmt, *a):
        if self.server.args.verbose:
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))


def main():
    ap = argparse.ArgumentParser(description="OCR Lite browser build: local server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model-dir", default=None, help="a folder holding text-q8_0.gguf + mmproj-bf16.gguf (/model/)")
    ap.add_argument("--data-dir", default=None, help="files served at /data/ (test images)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dist", default=None, help="serve this built site instead of web/ + ui/ (as GitHub Pages does)")
    ap.add_argument("--prefix", default=None, help="with --dist: serve it under this path (e.g. /historic-htr-ocr/)")
    args = ap.parse_args()
    if args.prefix and not args.dist:
        ap.error("--prefix goes with --dist")
    if args.model_dir:
        for m in MODEL_FILES:
            if not os.path.isfile(os.path.join(args.model_dir, m)):
                ap.error(f"{m} not in {args.model_dir}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.args = args
    if args.dist:
        args.dist = os.path.abspath(args.dist)
        print(f"OCR Lite site (as GitHub Pages serves it): http://{args.host}:{args.port}{args.prefix or '/'}", flush=True)
    else:
        print(f"OCR Lite web (dev): http://{args.host}:{args.port}/   selftest: http://{args.host}:{args.port}/selftest.html", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
