#!/usr/bin/env python3
# OCR Lite — Copyright (c) 2026 Bernhard Straub
# SPDX-License-Identifier: MIT
"""convert_output.py — the model's page transcription → TEI P5, or an HTML viewer of that TEI.

    python convert_output.py page.txt --to html --status raw > page.html
    python convert_output.py page.txt --to tei --status draft > page.xml
    cat page.txt | python convert_output.py - --to tei --status final --title "Letter 12, p. 3"
    python convert_output.py page.txt --to html --status raw --image page.jpg --embed --source-url https://… > page.html
    python convert_output.py page.txt --check      # the report as JSON on stdout, nothing converted

Image and source (both optional; without them the result names no image and no source):
  --source-url  the original scan online: TEI <facsimile> <graphic> + a <ref> in <sourceDesc>
  --image       a page image: TEI <graphic url> (the path as given, e.g. a file beside the XML), the HTML shows it
                beside the text; with --embed (HTML only) the image itself goes into the page as a data: URI — one
                self-contained file (its TEI then names only the online scan)
  --status      raw / draft / final: how far the text is corrected — in the TEI publicationStmt (required with --to)
  --layout      the page's regions (JSON {"width", "height", "regions": [{"points": [[x, y], …], "type", "lines": [n, …]}]},
                image px, reading order; "lines" = the 1-based input lines the region holds): TEI <zone> per region on the
                <surface>, each block <ab facs> to the zone holding its first line; the HTML gets a Grid view (the page image
                with its zones coloured + numbered beside the abstract grid of their blocks in the same colours)

The HTML is a self-contained, shareable viewer: it holds the TEI (an editable text area) and renders
the page from it with its own small script — toggle "TEI" to edit the TEI with a live rendering, Copy TEI, Download TEI,
Download HTML (the file saves itself with the edits). TEI that is not well-formed is flagged and the page keeps its last
good rendering. No server, no network.

The model's output format (one page):
  - one line per written line; a blank line separates blocks (paragraph, margin note, address, footer …)
  - inline marks, each with its own opening and closing character, closed at every line end and nested:
        ⟦struck⟧  ⟨inserted⟩  〔underlined〕  ⌈superscript⌉  ⌊subscript⌋
    ⌈+⌉ is the writer's insertion / reference sign; ▨ marks one lost or illegible spot
  - a table row is one line with its cells separated by TAB; ↵ is a line break inside a cell, ↵↵ a paragraph break
    inside a cell; ↵ in a line without a TAB is a line break as well (a one-column list the reader saw as a table:
    the format has no one-column table, so the model writes it as one cell)

Mapping:
    mark      TEI                                      viewer (rendered from the TEI)
    line      <lb/> before each line                   <br> between lines
    block     <ab>                                     <p>
    ⟦…⟧       <del rend="strikethrough">               <del>
    ⟨…⟩       <add>                                    <ins>
    〔…〕       <hi rend="underline">                    <u>
    〔〔…〕〕     <hi rend="double-underline">             <u class="dbl"> (a drawn double underline: two
                                                       underline pairs directly inside each other)
    ⌈…⌉       <hi rend="superscript">                  <sup>
    ⌊…⌋       <hi rend="subscript">                    <sub>
    ⌈+⌉       <metamark rend="superscript">+</metamark> <sup class="sign">+</sup>
    (layout)  region type heading → <ab type="heading">, page-number → <fw type="pageNum">, rule (a drawn horizontal
              line, no text) → <milestone unit="section" rend="line"/> before the next block; with --layout only
    ▨         <gap/>                                   <span class="gap">▨</span>
    TAB / ↵   <table><row><cell>, <lb/> in a cell      <table><tr><td>, <br> in a cell

The result is a diplomatic transcription in page order: no <choice>, <expan>, <supplied> or editorial apparatus.
Model output can be malformed (a mark never closed, a closing mark without its opening one). One case is unambiguous
and is normalised: a mark opened on one line and closed on a later line of the same block (the reader ran a struck or
inserted passage across the line break) is closed at each line end and reopened on the next line, as the format asks;
each such span is noted on stderr. Anything else is not repaired and not carried into the result: every unpaired mark
of the page is reported on stderr with its line number, nothing is written, and the exit code is 2.
"""
import argparse
import base64
import html
import json
import mimetypes
import sys

MARKS = {                     # open → (close, TEI open, TEI close)
    "⟦": ("⟧", '<del rend="strikethrough">', "</del>"),
    "⟨": ("⟩", "<add>", "</add>"),
    "〔": ("〕", '<hi rend="underline">', "</hi>"),
    "⌈": ("⌉", '<hi rend="superscript">', "</hi>"),
    "⌊": ("⌋", '<hi rend="subscript">', "</hi>"),
}
CLOSE = {v[0]: k for k, v in MARKS.items()}
SIGN = "⌈+⌉"
GAP = "▨"
CELL_BREAK = "↵"


class Report:
    """the unpaired marks of a page; any one of them stops the conversion"""

    def __init__(self):
        self.items = []
        self.notes = []                                    # spans closed and reopened per line (not errors)

    def add(self, line_no, msg):
        self.items.append((line_no, msg))


def esc(s):
    return html.escape(s, quote=False)


def attr(s):
    return html.escape(s, quote=True)


def data_uri(path):
    """the image file as a data: URI (HTML --embed)"""
    kind = mimetypes.guess_type(path)[0]
    if not kind or not kind.startswith("image/"):
        raise SystemExit(f"convert_output: --embed: {path} is not a known image type")
    with open(path, "rb") as f:
        return f"data:{kind};base64," + base64.b64encode(f.read()).decode("ascii")


def balanced(text):
    """[(open index, close index)] of the mark pairs of one line that nest properly; marks outside them stay text"""
    stack, pairs = [], []
    for i, ch in enumerate(text):
        if ch in MARKS:
            stack.append(i)
        elif ch in CLOSE:
            if stack and text[stack[-1]] == CLOSE[ch]:
                pairs.append((stack.pop(), i))
    return pairs


def inline(text, line_no, rep):
    """one line (or one cell part) → TEI inline markup"""
    pairs = balanced(text)
    opens = {a: b for a, b in pairs}
    closes = {b for _a, b in pairs}
    # 〔〔…〕〕 = ONE double underline: an underline pair whose next character opens an underline that
    # closes right before it — the inner pair's two characters are absorbed
    dbl = {a: b for a, b in pairs if text[a] == "〔" and text[a + 1:a + 2] == "〔" and opens.get(a + 1) == b - 1}
    absorbed = {a + 1 for a in dbl} | {b - 1 for b in dbl.values()}
    dbl_close = set(dbl.values())
    out, i = [], 0
    while i < len(text):
        if i in dbl:
            out.append('<hi rend="double-underline">')
            i += 1
            continue
        if i in absorbed:
            i += 1
            continue
        if i in dbl_close:
            out.append("</hi>")
            i += 1
            continue
        if text.startswith(SIGN, i) and opens.get(i) == i + 2:
            out.append('<metamark rend="superscript">+</metamark>')
            i += 3
            continue
        ch = text[i]
        if i in opens:
            out.append(MARKS[ch][1])
        elif i in closes:
            out.append(MARKS[CLOSE[ch]][2])
        elif ch in MARKS or ch in CLOSE:
            rep.add(line_no, f"unpaired mark {ch!r}: {text!r}")
        elif ch == GAP:
            out.append("<gap/>")
        else:
            out.append(esc(ch))
        i += 1
    return "".join(out)


def cell(text, line_no, rep):
    """a table cell: ↵ = a line break inside the cell (↵↵ = a paragraph break: two breaks)"""
    return "<lb/>".join(inline(p, line_no, rep) for p in text.split(CELL_BREAK))


def respan(block, notes):
    """a block whose marks run across line breaks → each mark closed at the line end and reopened on the next line;
    the block unchanged (its marks then reported as unpaired) when a closer does not match its opener or a mark is
    still open at the block's end"""
    carry, out, spans = [], [], []
    for n, ln in block:
        line = "".join(carry) + ln
        stack = []
        for ch in line:
            if ch in MARKS:
                stack.append(ch)
            elif ch in CLOSE:
                if not stack or stack[-1] != CLOSE[ch]:
                    return block
                stack.pop()
        if stack:
            spans.append((n, "".join(stack)))
        out.append((n, line + "".join(MARKS[o][0] for o in reversed(stack))))
        carry = stack
    if carry:
        return block
    for n, marks in spans:
        notes.append((n, f"{marks} runs on into line {n + 1} — closed at the line end and reopened"))
    return out


def blocks(text, notes):
    """the page → [[(line number, line)]] per block (blank-line separated), marks run across lines respanned"""
    out, cur = [], []
    for n, raw in enumerate(text.splitlines(), 1):
        if raw.strip():
            cur.append((n, raw.rstrip("\r")))
        elif cur:
            out.append(respan(cur, notes))
            cur = []
    if cur:
        out.append(respan(cur, notes))
    return out


def runs(block):
    """a block → [("lines" | "table", [(n, line)])]: consecutive lines holding a TAB form one table"""
    out = []
    for n, ln in block:
        kind = "table" if "\t" in ln else "lines"
        if out and out[-1][0] == kind:
            out[-1][1].append((n, ln))
        else:
            out.append((kind, [(n, ln)]))
    return out


def table(rows, rep, indent):
    width = max(len(ln.split("\t")) for _n, ln in rows)
    out = [f"{indent}<table>"]
    for n, ln in rows:
        cells = ln.split("\t")
        cells += [""] * (width - len(cells))
        out.append(f"{indent}  <row>" + "".join(f"<cell>{cell(c, n, rep)}</cell>" for c in cells) + "</row>")
    out.append(f"{indent}</table>")
    return out


BASE_OCR = "LightOnOCR-3-1B, fine-tuned for historic handwriting"   # --base-ocr default
STATUS = {                    # --status: how far the transcription is corrected
    "raw": "Automatic transcription (handwritten text recognition), not proofread.",
    "draft": "Automatic transcription (handwritten text recognition), partly corrected by hand; not final.",
    "final": "Transcription proofread and corrected by hand.",
}


def to_tei(text, title, rep, image=None, source_url=None, status="raw", base_ocr=BASE_OCR, layout=None):
    zone_of = {}                                   # input line → the zone id of the region holding it
    regs = (layout or {}).get("regions", [])
    for k, r in enumerate(regs):
        for n in r.get("lines", []):
            zone_of.setdefault(n, f"r{k + 1}")
    zone_type = {f"r{k + 1}": r.get("type") for k, r in enumerate(regs)}
    # a RULE region (a drawn horizontal line) holds no text: a <milestone> before the first block of
    # the next region in reading order that holds text (none after it: at the end)
    rule_before, rule_end = {}, []
    for k, r in enumerate(regs):
        if r.get("type") != "rule":
            continue
        m = f'      <milestone unit="section" rend="line" facs="#r{k + 1}"/>'
        nxt = next((min(q["lines"]) for q in regs[k + 1:] if q.get("lines")), None)
        (rule_before.setdefault(nxt, []) if nxt is not None else rule_end).append(m)
    body = []
    for b in blocks(text, rep.notes):
        z = zone_of.get(b[0][0])
        body += rule_before.pop(b[0][0], [])
        # the region's type names the block: a heading → <ab type="heading">, a page number →
        # <fw type="pageNum"> (both valid anywhere in a div); everything else <ab>
        t = zone_type.get(z) if z else None
        tag = "fw" if t == "page-number" else "ab"
        typ = ' type="pageNum"' if t == "page-number" else ' type="heading"' if t == "heading" else ""
        close = f"      </{tag}>"
        body.append(f'      <{tag}{typ} facs="#{z}">' if z else "      <ab>")
        for kind, lines in runs(b):
            if kind == "table":
                body += table(lines, rep, "        ")
            else:
                # ↵ outside a table = a line break too (a one-column table: the model writes such a list as one cell)
                body += [f"        <lb/>{inline(p, n, rep)}" for n, ln in lines for p in ln.split(CELL_BREAK)]
        body.append(close)
    for rest in rule_before.values():               # a rule before a block that did not come (re-blocked text): at the end
        rule_end += rest
    body += rule_end
    graphics = [u for u in (image, source_url) if u]
    source = f'<p>Original scan: <ref target="{attr(source_url)}">{esc(source_url)}</ref></p>' if source_url else ""
    zones = []
    if layout:
        for k, r in enumerate(layout["regions"]):
            pts = " ".join(f"{round(x)},{round(y)}" for x, y in r["points"])
            ty = f' type="{attr(r["type"])}"' if r.get("type") else ""
            zones.append(f'      <zone xml:id="r{k + 1}"{ty} points="{pts}"/>')
    size = f' ulx="0" uly="0" lrx="{round(layout["width"])}" lry="{round(layout["height"])}"' if layout else ""
    facs = [
        "  <facsimile>",
        f'    <surface xml:id="page"{size}>',
        *[f'      <graphic url="{attr(u)}"/>' for u in graphics],
        *zones,
        "    </surface>",
        "  </facsimile>",
    ] if graphics or layout else []
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">',
        "  <teiHeader>",
        "    <fileDesc>",
        f"      <titleStmt><title>{esc(title)}</title></titleStmt>",
        f"      <publicationStmt><p>{STATUS[status]}</p>"
        + (f"<p>Base OCR: {esc(base_ocr)}.</p>" if base_ocr else "")
        + f'<p><idno type="status">{status}</idno></p></publicationStmt>',
        "      <sourceDesc><p>Diplomatic transcription in page reading order; no expansions, normalisation or"
        f" editorial apparatus.</p>{source}</sourceDesc>",
        "    </fileDesc>",
        "  </teiHeader>",
        *facs,
        "  <text>",
        "    <body>",
        *(['      <pb facs="#page"/>'] if graphics or layout else []),
        "      <div>",
        *["  " + ln for ln in body],
        "      </div>",
        "    </body>",
        "  </text>",
        "</TEI>",
        "",
    ])


CSS = """body{font-family:Georgia,serif;max-width:46rem;margin:0 auto;padding:0 1rem 2rem;line-height:1.5;background:#fff;color:#1c1c1c}
body.with-image,body.edit{max-width:90rem}
p{margin:0 0 1.1em}
h1{font-size:1.3rem;margin:.6rem 0 .3rem}
del{text-decoration:line-through;color:#8a3b3b}
ins{text-decoration:none;color:#2b5d8a}
u.dbl{text-decoration-style:double}
p.head{font-weight:bold;font-size:1.08em}
p.fw{font-size:.85em;color:#666;text-align:right}
hr.rule{border:0;border-top:1.5px solid #555;margin:.6em 15%}
.sign{color:#2b5d8a}
.gap{color:#999}
table{border-collapse:collapse;margin:0 0 1.1em}
td{border:1px solid #ccc;padding:.15em .5em;vertical-align:top}
.src{font-size:.85em;color:#666;margin:0 0 .4em;overflow-wrap:anywhere}
.bar{position:sticky;top:0;z-index:1;background:#fff;display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;
  padding:.5rem 0;border-bottom:1px solid #ddd;font:13px system-ui,sans-serif}
.bar button{font:inherit;padding:.2rem .6rem}
.err{color:#b00020}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,20rem),1fr));gap:2rem;margin-top:1rem}
.page-image img{width:100%;height:auto;position:sticky;top:3rem}
#tei{display:none;width:100%;box-sizing:border-box;min-height:75vh;font:12px/1.45 ui-monospace,Menlo,monospace}
body.edit #tei{display:block}
#grid{display:none;margin-top:1rem;grid-template-columns:1fr 1fr;gap:1rem;align-items:start}
body.grid #grid{display:grid}
body.grid .cols{display:none}
@media (max-width:50rem){#grid{grid-template-columns:1fr}}
#grid .gpane{position:relative;width:100%}
#grid .gimg img{position:absolute;inset:0;width:100%;height:100%}
#grid .gimg svg{position:absolute;inset:0;width:100%;height:100%}
#grid .gimg polygon{stroke-width:2;vector-effect:non-scaling-stroke;cursor:pointer}
#grid .gimg text{font:bold 28px system-ui,sans-serif;paint-order:stroke;stroke:#fff;stroke-width:5px}
#grid .gcells{border:1px solid #e3e3e3;background:#fff}
#grid .cell{position:absolute;box-sizing:border-box;overflow:hidden;padding:.15em .3em;line-height:1.25;border:1.5px solid;
  border-radius:3px}
#grid .hot{outline:3px solid #222;z-index:2}
#grid polygon.hot{stroke-width:4;outline:none}
#grid .cell p{margin:0 0 .4em}
#grid .cell .lbl{font:bold 10px system-ui,sans-serif;float:right;margin-left:.3em}
#grid .unplaced{grid-column:1/-1;padding:.4rem .6rem;border:1px dashed #c96;background:#fffaf3}
#grid .unplaced .lbl{font:11px system-ui,sans-serif;color:#a63;margin-bottom:.3rem}"""

VIEWER_JS = r"""(function () {
  // the page rendered from its own TEI (the text area) — edit the TEI, the page follows; nothing leaves the browser
  var NS = "http://www.tei-c.org/ns/1.0";
  var ta = document.getElementById("tei"), page = document.getElementById("page"), meta = document.getElementById("meta"),
      err = document.getElementById("err"), timer = null;
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) { e.className = cls; }
    if (text != null) { e.textContent = text; }
    return e;
  }
  function first(root, name) { return root ? root.getElementsByTagNameNS(NS, name)[0] : undefined; }
  var HI = {underline: "u", superscript: "sup", subscript: "sub"};
  function kids(src, dst) { for (var i = 0; i < src.childNodes.length; i++) { node(src.childNodes[i], dst); } }
  function node(n, dst) {
    if (n.nodeType === 3) {
      if (/\n/.test(n.nodeValue) && !n.nodeValue.trim()) { return; }          // the TEI's own line layout
      dst.appendChild(document.createTextNode(n.nodeValue.replace(/\s*\n\s*/g, " ")));
      return;
    }
    if (n.nodeType !== 1) { return; }
    var e, name = n.localName, rend = n.getAttribute("rend") || "";
    if (name === "lb") { dst.appendChild(el("br")); return; }
    if (name === "gap") { dst.appendChild(el("span", "gap", "▨")); return; }
    if (name === "pb") { return; }
    if (name === "milestone") {                    // a drawn horizontal rule (rend="line")
      if (rend === "line") { dst.appendChild(el("hr", "rule")); }
      return;
    }
    if (name === "ab" || name === "fw") {          // a block; a heading / a page number (fw) styled by its type
      var ty = n.getAttribute("type") || "";
      e = el("p", ty === "heading" ? "head" : name === "fw" ? "fw" : null);
      kids(n, e);
      if (e.firstChild && e.firstChild.nodeName === "BR") { e.removeChild(e.firstChild); }   // <lb/> opens each line
      dst.appendChild(e);
      return;
    }
    e = name === "del" ? el("del") : name === "add" ? el("ins")
      : name === "hi" ? (rend === "double-underline" ? el("u", "dbl") : el(HI[rend] || "span"))
      : name === "metamark" ? el("sup", "sign") : name === "table" ? el("table") : name === "row" ? el("tr")
      : name === "cell" ? el("td") : el("div");
    kids(n, e);
    if (name === "cell" && e.firstChild && e.firstChild.nodeName === "BR") { e.removeChild(e.firstChild); }
    dst.appendChild(e);
  }
  function render() {
    var doc = new DOMParser().parseFromString(ta.value, "application/xml");
    var bad = doc.getElementsByTagName("parsererror")[0];
    if (bad) {
      err.textContent = "TEI not well-formed — the page shows the last good state: " + bad.textContent.split("\n")[0];
      err.hidden = false;
      return;
    }
    err.hidden = true;
    var title = first(doc, "title"), ps = first(doc, "publicationStmt"), ref = first(first(doc, "sourceDesc"), "ref");
    document.title = title ? title.textContent : "transcription";
    meta.textContent = "";
    meta.appendChild(el("h1", null, document.title));
    var pubs = ps ? ps.getElementsByTagNameNS(NS, "p") : [];
    for (var i = 0; i < pubs.length; i++) {                // status sentence, base OCR — not the bare status idno
      if (!first(pubs[i], "idno") && pubs[i].textContent.trim()) { meta.appendChild(el("p", "src", pubs[i].textContent.trim())); }
    }
    if (ref) {
      var p = el("p", "src", "Original scan: "), a = el("a", null, ref.getAttribute("target"));
      a.href = ref.getAttribute("target");
      p.appendChild(a);
      meta.appendChild(p);
    }
    page.textContent = "";
    var body = first(doc, "body");
    if (body) { kids(body, page); }
    grid(doc, body);
  }
  // GRID (--layout: the abstract grid of the blocks beside the coloured regions on the image, mapping one back onto the
  // other): left the page image with every zone filled in its colour + its number,
  // right the abstract grid — every zone a cell at its box (percent of the surface) in the same colour, its blocks
  // inside, the text shrunk until it fits; hovering a cell or a region marks both. Blocks no zone holds: listed below.
  var gridEl = document.getElementById("grid"), gridBtn = document.getElementById("gridmode");
  var SVG = "http://www.w3.org/2000/svg";
  function hue(i) { return Math.round((i * 137.508) % 360); }          // golden-angle hues: neighbours stay apart
  function grid(doc, body) {
    if (!gridEl) { return; }
    var surf = first(doc, "surface"), zs = surf ? surf.getElementsByTagNameNS(NS, "zone") : [];
    var W = surf ? +surf.getAttribute("lrx") : 0, H = surf ? +surf.getAttribute("lry") : 0;
    gridBtn.hidden = !(zs.length && W && H);
    gridEl.textContent = "";
    if (gridBtn.hidden) { document.body.classList.remove("grid"); return; }
    var left = el("div", "gpane gimg"), right = el("div", "gpane gcells");
    left.style.aspectRatio = right.style.aspectRatio = W + " / " + H;
    if (gridEl.dataset.img) { var im = el("img"); im.src = gridEl.dataset.img; im.alt = "page image"; left.appendChild(im); }
    var svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("preserveAspectRatio", "none");
    left.appendChild(svg);
    var abs = [], cells = [];                      // the text blocks in order: <ab> and <fw> (a page number)
    if (body) {
      var all = body.getElementsByTagNameNS(NS, "*");
      for (var q = 0; q < all.length; q++) { if (all[q].localName === "ab" || all[q].localName === "fw") { abs.push(all[q]); } }
    }
    function pair(a, b) {
      [a, b].forEach(function (x) {
        x.addEventListener("mouseenter", function () { a.classList.add("hot"); b.classList.add("hot"); });
        x.addEventListener("mouseleave", function () { a.classList.remove("hot"); b.classList.remove("hot"); });
      });
    }
    for (var i = 0; i < zs.length; i++) {
      var z = zs[i], id = z.getAttribute("xml:id") || z.getAttributeNS("http://www.w3.org/XML/1998/namespace", "id");
      var pts = (z.getAttribute("points") || "").trim().split(/\s+/).map(function (p) { return p.split(",").map(Number); });
      var xs = pts.map(function (p) { return p[0]; }), ys = pts.map(function (p) { return p[1]; });
      var x0 = Math.max(0, Math.min.apply(null, xs)), x1 = Math.min(W, Math.max.apply(null, xs));
      var y0 = Math.max(0, Math.min.apply(null, ys)), y1 = Math.min(H, Math.max.apply(null, ys));
      var h = hue(i), line = "hsl(" + h + ",65%,38%)";
      var poly = document.createElementNS(SVG, "polygon");
      poly.setAttribute("points", pts.map(function (p) { return p.join(","); }).join(" "));
      poly.setAttribute("fill", "hsla(" + h + ",75%,50%,.22)");
      poly.setAttribute("stroke", line);
      svg.appendChild(poly);
      var num = document.createElementNS(SVG, "text");
      num.setAttribute("x", x0 + 8); num.setAttribute("y", y0 + 30); num.setAttribute("fill", line);
      num.textContent = String(i + 1);
      svg.appendChild(num);
      var c = el("div", "cell");
      c.style.left = (100 * x0 / W) + "%"; c.style.top = (100 * y0 / H) + "%";
      c.style.width = (100 * (x1 - x0) / W) + "%"; c.style.height = (100 * (y1 - y0) / H) + "%";
      c.style.background = "hsla(" + h + ",75%,50%,.12)"; c.style.borderColor = line;
      var lbl = el("span", "lbl", (i + 1) + (z.getAttribute("type") && z.getAttribute("type") !== "body" ? " · " + z.getAttribute("type") : ""));
      lbl.style.color = line;
      c.appendChild(lbl);
      if (z.getAttribute("type") === "rule") {     // a drawn horizontal line: the cell shows the line, no text
        c.style.background = "transparent";
        var hr = el("hr", "rule");
        hr.style.margin = "0"; hr.style.position = "absolute"; hr.style.top = "50%"; hr.style.left = "0"; hr.style.right = "0";
        c.appendChild(hr);
      }
      for (var k = 0; k < abs.length; k++) { if (abs[k].getAttribute("facs") === "#" + id) { node(abs[k], c); } }
      right.appendChild(c);
      cells.push(c);
      pair(poly, c);
    }
    gridEl.appendChild(left);
    gridEl.appendChild(right);
    var rest = el("div", "unplaced");                 // blocks no zone holds: listed under the page, never dropped
    for (var k = 0; k < abs.length; k++) { if (!abs[k].getAttribute("facs")) { node(abs[k], rest); } }
    if (rest.childNodes.length) { rest.insertBefore(el("div", "lbl", "not placed on the page"), rest.firstChild); gridEl.appendChild(rest); }
    fit(cells);
  }
  function fit(cells) {                            // the largest font ≤ 1em (steps of 0.88, ≥ 6 px) that fits each box
    if (!document.body.classList.contains("grid")) { return; }
    var basePx = parseFloat(getComputedStyle(document.body).fontSize);
    cells.forEach(function (c) {
      var f = basePx;
      c.style.fontSize = f + "px";
      while (f > 6 && (c.scrollHeight > c.clientHeight + 1 || c.scrollWidth > c.clientWidth + 1)) {
        f = Math.max(6, f * 0.88);
        c.style.fontSize = f + "px";
      }
    });
  }
  if (gridBtn) {
    gridBtn.onclick = function () {
      var on = document.body.classList.toggle("grid");
      this.textContent = on ? "Plain" : "Grid";
      if (on) { fit(Array.prototype.slice.call(gridEl.getElementsByClassName("cell"))); }
    };
    window.addEventListener("resize", function () { fit(Array.prototype.slice.call(gridEl.getElementsByClassName("cell"))); });
  }
  function save(name, type, data) {
    var a = el("a");
    a.href = URL.createObjectURL(new Blob([data], {type: type}));
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }
  function base() { return (document.title || "transcription").replace(/[^\w.-]+/g, "_"); }
  ta.addEventListener("input", function () { clearTimeout(timer); timer = setTimeout(render, 250); });
  document.getElementById("mode").onclick = function () {
    var on = document.body.classList.toggle("edit");
    this.textContent = on ? "Page only" : "TEI";
  };
  document.getElementById("copy").onclick = function () { navigator.clipboard.writeText(ta.value); };
  document.getElementById("dl-tei").onclick = function () { save(base() + ".tei.xml", "application/xml", ta.value); };
  document.getElementById("dl-html").onclick = function () {
    ta.textContent = ta.value;                     // the edits go into the saved page
    var edit = document.body.classList.contains("edit");
    document.body.classList.remove("edit");
    var wasGrid = document.body.classList.contains("grid");
    document.body.classList.remove("grid");
    page.textContent = "";
    meta.textContent = "";
    if (gridEl) { gridEl.textContent = ""; }
    var html = "<!doctype html>\n" + document.documentElement.outerHTML;
    if (wasGrid) { document.body.classList.add("grid"); }
    if (edit) { document.body.classList.add("edit"); }
    render();
    save(base() + ".html", "text/html", html);
  };
  render();
})();"""


def to_html(tei, title, image=None, layout=False):
    """the viewer page: the TEI in an editable text area, rendered by VIEWER_JS; `image` = an img src (path, URL or
    data: URI) shown beside the text; `layout` = the TEI has zones: a Grid view (the blocks at their zones, the image
    under them)"""
    pic = f'<div class="page-image"><img src="{attr(image)}" alt="page image"></div>' if image else ""
    grid = (f'<div id="grid" data-img="{attr(image)}"></div>' if image else '<div id="grid"></div>') if layout else ""
    gbtn = ('<button id="gridmode" hidden title="the page image with its coloured regions beside the grid of their '
            'blocks">Grid</button>') if layout else ""
    return "\n".join([
        "<!doctype html>",
        '<html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{esc(title)}</title>",
        f"<style>\n{CSS}\n</style>",
        '</head><body class="with-image">' if image else "</head><body>",
        '<div class="bar"><button id="mode" title="show the TEI beside the page — edit it, the page follows">TEI</button>'
        + gbtn +
        '<button id="copy">Copy TEI</button><button id="dl-tei">Download TEI</button>'
        '<button id="dl-html" title="this page with your edits">Download HTML</button>'
        '<span id="err" class="err" hidden></span></div>',
        '<header id="meta"></header>',
        "<noscript><p>The page is rendered from its TEI with JavaScript.</p></noscript>",
        f'<div class="cols">{pic}<main id="page"></main><textarea id="tei" spellcheck="false">{esc(tei)}</textarea></div>',
        grid,
        f"<script>\n{VIEWER_JS}\n</script>",
        "</body></html>",
        "",
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help="the model's output (a text file), or - for stdin")
    ap.add_argument("--to", choices=("html", "tei"))
    ap.add_argument("--check", action="store_true",
                    help="convert nothing: the report as JSON on stdout ({errors, notes}: [{line, msg}]), exit 2 if malformed")
    ap.add_argument("--source-url", help="the original scan online")
    ap.add_argument("--image", help="a page image (path or URL) beside the text / in <facsimile>")
    ap.add_argument("--embed", action="store_true", help="HTML: the --image file itself in the page (data: URI)")
    ap.add_argument("--base-ocr", default=BASE_OCR, help=f"the reading model named in the TEI (default: {BASE_OCR!r}; "
                    "an empty string names none, e.g. for an edition's own transcription)")
    ap.add_argument("--status", choices=tuple(STATUS), help="required with --to: raw (not proofread) · draft (partly "
                    "corrected) · final (proofread) — named in the TEI publicationStmt and above the HTML text")
    ap.add_argument("--title", help="document title (default: the file name)")
    ap.add_argument("--layout", help="a JSON file with the page's regions (image px): TEI zones + the HTML Grid view")
    a = ap.parse_args()
    if not a.check and not a.to:
        ap.error("--to is required (or --check)")
    if a.to and not a.status:
        ap.error("--status raw|draft|final is required with --to (how far the transcription is corrected)")
    if a.embed and (a.to != "html" or not a.image):
        ap.error("--embed needs --to html and --image")
    text = sys.stdin.read() if a.input == "-" else open(a.input, encoding="utf-8").read()
    title = a.title or ("transcription" if a.input == "-" else a.input.rsplit("/", 1)[-1])
    rep = Report()
    if a.check:
        to_tei(text, title, rep)
        as_list = lambda xs: [{"line": n, "msg": m} for n, m in xs]
        print(json.dumps({"errors": as_list(rep.items), "notes": as_list(rep.notes)}, ensure_ascii=False))
        sys.exit(2 if rep.items else 0)
    layout = json.load(open(a.layout, encoding="utf-8")) if a.layout else None
    if a.to == "tei":
        out = to_tei(text, title, rep, a.image, a.source_url, a.status, a.base_ocr, layout)
    else:                                          # the viewer around the same TEI; an embedded image is no file to name
        tei = to_tei(text, title, rep, None if a.embed else a.image, a.source_url, a.status, a.base_ocr, layout)
        out = to_html(tei, title, data_uri(a.image) if a.embed else a.image, layout is not None)
    for n, m in rep.notes:
        print(f"convert_output: line {n}: {m}", file=sys.stderr)
    if rep.items:
        for n, m in rep.items:
            print(f"convert_output: line {n}: {m}", file=sys.stderr)
        print(f"convert_output: {len(rep.items)} unpaired mark(s) — malformed read, nothing written", file=sys.stderr)
        sys.exit(2)
    sys.stdout.write(out)


if __name__ == "__main__":
    main()
