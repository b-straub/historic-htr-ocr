#!/usr/bin/env python3
# OCR Lite — Copyright (c) 2026 Bernhard Straub
# SPDX-License-Identifier: MIT
#
# OCR Lite's texts as one table, for spotting what is missing or untranslated: every ui/lang/<lang>.json side by side
# (key · one column per language, English first) in a self-contained HTML page. Each row is flagged when
#   missing        a language has no text for the key
#   untranslated   a language's text equals the English one — skipped where the text is only symbols, numbers,
#                  placeholders or names / technical terms that stay as they are (TERMS)
#   placeholders   the {name} placeholders differ from the English text's
# On top: the counts per flag; a switch shows the flagged rows only. Standard library only.
#   app/tools/i18n_table.py [OUT.html]      (default: i18n_table.html in the current directory; prints where it wrote)
import html
import json
import os
import re
import sys

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANG = os.path.join(APP, "ui", "lang")
# words that stay the same in every language (names, and technical terms, which are never Germanized)
TERMS = {
    "ocr", "lite", "webgpu", "gpu", "auto", "markup", "text", "version", "build", "engine", "name", "ok", "hugging", "face",
    "ipad", "iphone", "safari", "chrome", "edge", "firefox", "mac", "nvidia", "prefill", "tok", "tokens", "token", "preset",
    "greedy", "cache", "pwa", "opfs", "txt", "html", "png", "gguf", "mib", "linux", "android", "windows", "control", "panel",
    "encode", "decode", "weights", "compute", "choice", "peak",
}


def load():
    langs = {}
    for name in sorted(os.listdir(LANG)):
        if name.endswith(".json"):
            with open(os.path.join(LANG, name), encoding="utf-8") as f:
                langs[name[:-5]] = json.load(f)
    if "en" not in langs:
        sys.exit(f"i18n_table: no en.json in {LANG}")
    order = ["en"] + [l for l in langs if l != "en"]
    return {l: langs[l] for l in order}


def placeholders(s):
    return set(re.findall(r"\{(\w+)\}", s))


def worth_translating(s):
    """a text with words that a translation would change (not only symbols, numbers, placeholders, TERMS)"""
    rest = re.sub(r"\{\w+\}|https?://\S+|\S+\.(?:json|js|html|gguf)", " ", s)
    words = [w for w in re.findall(r"[^\W\d_]{3,}", rest) if w.lower() not in TERMS]
    return len(words) > 0


def flags(key, langs):
    en = langs["en"].get(key)
    out = []
    for l, texts in langs.items():
        if key not in texts:
            out.append(("missing", l))
            continue
        if l == "en" or en is None:
            continue
        if texts[key] == en and worth_translating(en):
            out.append(("untranslated", l))
        if placeholders(texts[key]) != placeholders(en):
            out.append(("placeholders", l))
    return out


def page(langs):
    keys = list(langs["en"])
    for texts in langs.values():
        keys += [k for k in texts if k not in keys]
    counts = {"missing": 0, "untranslated": 0, "placeholders": 0}
    rows = []
    for k in keys:
        fl = flags(k, langs)
        for kind, _ in fl:
            counts[kind] += 1
        cells = "".join(
            f'<td class="{"miss" if k not in t else ""}">{html.escape(t[k]) if k in t else "—"}</td>' for t in langs.values()
        )
        note = ", ".join(f"{kind} ({l})" for kind, l in fl)
        rows.append(f'<tr class="{"flag" if fl else ""}"><td class="key">{html.escape(k)}</td>{cells}<td class="note">{html.escape(note)}</td></tr>')
    heads = "".join(f"<th>{html.escape(l)}</th>" for l in langs)
    summary = " · ".join(f"{n} {kind}" for kind, n in counts.items())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR Lite texts</title>
<style>
:root {{ --bg: #fff; --fg: #16181d; --muted: #6b7280; --line: #e5e7eb; --flag: #fff7ed; --bad: #c2410c; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #111317; --fg: #e8eaee; --muted: #9aa1ad; --line: #2a2e35; --flag: #2a1d12; --bad: #fb923c; }} }}
body {{ margin: 0; padding: 16px; background: var(--bg); color: var(--fg); font: 13px/1.45 system-ui, sans-serif; }}
h1 {{ font-size: 17px; margin: 0 0 4px; }}
.sum {{ color: var(--muted); margin-bottom: 10px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; vertical-align: top; padding: 5px 8px; border-bottom: 1px solid var(--line); }}
th {{ position: sticky; top: 0; background: var(--bg); }}
td.key {{ font-family: ui-monospace, Menlo, monospace; font-size: 12px; white-space: nowrap; }}
td.note, td.miss {{ color: var(--bad); }}
tr.flag {{ background: var(--flag); }}
body.only tr:not(.flag):not(:first-child) {{ display: none; }}
.wrap {{ overflow-x: auto; }}
</style></head>
<body>
<h1>OCR Lite texts</h1>
<div class="sum">{len(keys)} keys · {summary} · <label><input type="checkbox" id="only"> flagged rows only</label></div>
<div class="wrap"><table><tr><th>key</th>{heads}<th>flags</th></tr>
{chr(10).join(rows)}
</table></div>
<script>document.getElementById("only").onchange = (e) => document.body.classList.toggle("only", e.target.checked);</script>
</body></html>
"""


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "i18n_table.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(page(load()))
    print(f"i18n_table: wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
