// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// ui/editor.js round-trip + toggle / undo / insert check in jsdom (no app, no model):
//   cargo build --release -p ocr-core --example render; (dir with jsdom) node tests/editor_roundtrip.js target/release/examples/render ui/editor.js
const { JSDOM } = require("jsdom");
const { execFileSync } = require("child_process");
const fs = require("fs");
const [render, editorJs] = process.argv.slice(2);
const dom = new JSDOM("<div id=e></div>", { runScripts: "outside-only" });
const w = dom.window;
w.eval(fs.readFileSync(editorJs, "utf8"));
const marks = { pairs: [["del","⟦","⟧"],["ins","⟨","⟩"],["sub","⌊","⌋"],["sup","⌈","⌉"],["u","〔","〕"]].map(([tag,open,close])=>({tag,open,close})), cellSep: "\t", cellBreak: "↵", gap: "▨" };
const html = (t) => Promise.resolve(execFileSync(render, { input: t }).toString());
const root = w.document.getElementById("e");
const ed = w.MarkupEditor(root, marks, html, () => {});
let fail = 0;
const eq = (name, got, want) => { if (got !== want) { fail++; console.log("FAIL", name, JSON.stringify(got), "≠", JSON.stringify(want)); } else console.log("ok  ", name); };
(async () => {
  for (const m of ["Verehrteste ⟦Excel⟧lenz!", "a⟦b⌈c⌉⟧d ⟨neu⟩ 〔u〕 x⌊2⌋", "line1\n\nline3", "A\tB↵C\n\tD", "A\tB↵↵C", "x⟧ ⟨y", "<&> ▨ \"q\"", "  indented", "pre\nA\tB\npost"]) {
    await ed.load(m);
    eq("round " + JSON.stringify(m), ed.serialize(), m);
  }
  // toggle strike on "bc" of "abcd"
  await ed.load("abcd");
  root.focus(); ed.select(1, 3); await ed.toggle("del");
  eq("toggle on", ed.markup, "a⟦bc⟧d");
  ed.select(1, 3); await ed.toggle("del");
  eq("toggle off", ed.markup, "abcd");
  ed.select(0, 2); await ed.toggle("sup");
  eq("toggle sup", ed.markup, "⌈ab⌉cd");
  await ed.undo(); eq("undo", ed.markup, "abcd");
  await ed.undo(); eq("undo2", ed.markup, "a⟦bc⟧d");
  await ed.redo(); eq("redo", ed.markup, "abcd");
  // partial overlap: strike "a⟦bc⟧d" chars 2..4 → c and d struck
  await ed.load("a⟦bc⟧d"); ed.select(2, 4); await ed.toggle("del");
  eq("extend", ed.markup, "a⟦bcd⟧");
  // insert gap at caret 1
  await ed.load("abc"); ed.select(1, 1); await ed.insert("▨");
  eq("insert", ed.markup, "a▨bc");
  ed.select(4, 4);
  await ed.insert("\t"); eq("tab", ed.markup, "a▨bc\t");
  // two lines toggling across the newline
  await ed.load("ab\ncd"); ed.select(1, 4); await ed.toggle("u");
  eq("across lines", ed.markup, "a〔b〕\n〔c〕d");
  console.log(fail ? `${fail} FAILED` : "ALL OK");
})();
