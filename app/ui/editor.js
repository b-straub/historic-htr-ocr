// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// WYSIWYG editing of the tagged markup: the marks are shown as what they do, not as raw brackets, with undo / redo.
// The markup string stays the one truth; the editor shows its rendering (Rust
// render → contenteditable) and maps edits back:
//   flatten(DOM) → items: one per visible character { ch, tags } + separators ('\n' line · '\t' cell · '↵' in-cell break)
//   toMarkup(items) → the markup (marks opened / closed where the tag set changes; all closed at every separator —
//   the renderer pairs per line / cell piece)
// A format button toggles a tag on the selected items and re-renders; the caret / selection is kept as an item index.
// Undo / redo = the editor's own history of markup snapshots (typing coalesced); the browser's native undo is blocked.
"use strict";

window.MarkupEditor = function MarkupEditor(root, marks, renderHtml, onChange) {
  const STYLE = new Set(["DEL", "INS", "U", "SUP", "SUB"]);
  const openOf = Object.fromEntries(marks.pairs.map((p) => [p.tag, p.open]));
  const closeOf = Object.fromEntries(marks.pairs.map((p) => [p.tag, p.close]));
  let markup = "";
  let hist = [{ m: "", a: 0, b: 0 }];
  let hpos = 0;
  let typingTimer = null;

  // ── DOM → items ─────────────────────────────────────────────────────
  function trailingBr(br, block) {
    for (let n = br; n && n !== block; n = n.parentNode) {
      for (let s = n.nextSibling; s; s = s.nextSibling) {
        if (s.nodeType === 1 || (s.nodeType === 3 && s.data.length)) return false;
      }
    }
    return true;
  }

  function flatten(container) {
    const items = [];
    const blocks = [];
    let first = true;
    const newBlock = (el) => {
      if (!first) items.push({ sep: "\n" });
      first = false;
      blocks.push({ index: items.length, el });
    };
    const inline = (node, tags, inCell, block) => {
      let had = false;
      for (const c of node.childNodes) {
        if (c.nodeType === 3) {
          for (let k = 0; k < c.data.length; k++) items.push({ ch: c.data[k], tags, node: c, off: k });
          had = had || c.data.length > 0;
        } else if (c.nodeType === 1) {
          const n = c.nodeName;
          if (n === "BR") {
            if (!trailingBr(c, block)) items.push({ sep: inCell ? "↵" : "\n" });
            had = true;
          } else if (n === "DIV" || n === "P") {
            // a block the browser made inside a line / cell (Enter): a break before it
            if (had) items.push({ sep: inCell ? "↵" : "\n" });
            blocks.push({ index: items.length, el: c });
            inline(c, tags, inCell, c);
            had = true;
          } else if (STYLE.has(n)) {
            const t = n.toLowerCase();
            inline(c, tags.includes(t) ? tags : [...tags, t], inCell, block);
            had = true;
          } else {
            inline(c, tags, inCell, block);
            had = true;
          }
        }
      }
    };
    for (const c of container.childNodes) {
      if (c.nodeName === "TABLE") {
        for (const tr of c.rows) {
          newBlock(tr.cells[0] || tr);
          [...tr.cells].forEach((td, j) => {
            if (j > 0) {
              items.push({ sep: "\t" });
              blocks.push({ index: items.length, el: td });
            }
            inline(td, [], true, td);
          });
        }
      } else if (c.nodeType === 1) {
        newBlock(c);
        inline(c, [], false, c);
      } else if (c.nodeType === 3 && c.data.trim()) {
        // text typed straight into the editor root: a line of its own
        newBlock(c);
        for (let k = 0; k < c.data.length; k++) items.push({ ch: c.data[k], tags: [], node: c, off: k });
      }
    }
    return { items, blocks };
  }

  function toMarkup(items) {
    let out = "";
    const stack = [];
    const closeAll = () => {
      while (stack.length) out += closeOf[stack.pop()];
    };
    for (const it of items) {
      if (it.sep) {
        closeAll();
        out += it.sep === "\n" ? "\n" : it.sep === "\t" ? marks.cellSep : marks.cellBreak;
        continue;
      }
      let k = 0;
      while (k < stack.length && it.tags.includes(stack[k])) k++;
      while (stack.length > k) out += closeOf[stack.pop()];
      for (const t of it.tags) {
        if (!stack.includes(t)) {
          stack.push(t);
          out += openOf[t];
        }
      }
      out += it.ch;
    }
    closeAll();
    return out;
  }

  // ── caret ↔ item index ──────────────────────────────────────────────
  function indexOf(node, off) {
    const r = document.createRange();
    r.setStart(root, 0);
    r.setEnd(node, off);
    const tmp = document.createElement("div");
    tmp.append(r.cloneContents());
    return flatten(tmp).items.length;
  }

  function selection() {
    const s = window.getSelection();
    if (!s.rangeCount || !root.contains(s.anchorNode)) return null;
    const r = s.getRangeAt(0);
    return { a: indexOf(r.startContainer, r.startOffset), b: indexOf(r.endContainer, r.endOffset) };
  }

  function point(flat, i) {
    const { items, blocks } = flat;
    if (i < items.length && !items[i].sep) return [items[i].node, items[i].off];
    if (i > 0 && i <= items.length && !items[i - 1].sep) return [items[i - 1].node, items[i - 1].off + 1];
    const b = blocks.find((x) => x.index === i);
    if (b) return [b.el, 0];
    return [root, root.childNodes.length];
  }

  function select(a, b) {
    const flat = flatten(root);
    const n = flat.items.length;
    const r = document.createRange();
    r.setStart(...point(flat, Math.min(a, n)));
    r.setEnd(...point(flat, Math.min(b, n)));
    const s = window.getSelection();
    s.removeAllRanges();
    s.addRange(r);
  }

  // ── render ──────────────────────────────────────────────────────────
  async function paint(m) {
    if (m === "") {
      root.innerHTML = ""; // :empty → the placeholder shows
      return;
    }
    const html = (await renderHtml(m)).replace(/>\n/g, ">");
    root.innerHTML = html;
    for (const td of root.querySelectorAll("td")) {
      if (!td.childNodes.length) td.append(document.createElement("br"));
    }
  }

  function changed() {
    onChange(markup);
  }

  // ── history ─────────────────────────────────────────────────────────
  function record(sel) {
    if (hist[hpos].m === markup) return;
    hist.splice(hpos + 1);
    hist.push({ m: markup, a: sel ? sel.a : 0, b: sel ? sel.b : 0 });
    hpos = hist.length - 1;
  }
  function flushTyping() {
    if (typingTimer) {
      clearTimeout(typingTimer);
      typingTimer = null;
      record(selection());
    }
  }
  async function jump(to) {
    flushTyping();
    if (to < 0 || to >= hist.length) return;
    hpos = to;
    markup = hist[hpos].m;
    await paint(markup);
    root.focus();
    select(hist[hpos].a, hist[hpos].b);
    changed();
  }

  /// replace the whole markup (new edits from the items model), re-render, select [a, b)
  async function commit(m, a, b) {
    flushTyping();
    markup = m;
    await paint(markup);
    root.focus();
    select(a, b);
    record({ a, b });
    changed();
  }

  // ── events ──────────────────────────────────────────────────────────
  root.addEventListener("input", () => {
    markup = toMarkup(flatten(root).items);
    changed();
    clearTimeout(typingTimer);
    typingTimer = setTimeout(() => {
      typingTimer = null;
      record(selection());
    }, 450);
  });
  root.addEventListener("beforeinput", (e) => {
    if (e.inputType === "historyUndo") {
      e.preventDefault();
      api.undo();
    } else if (e.inputType === "historyRedo") {
      e.preventDefault();
      api.redo();
    }
  });
  root.addEventListener("keydown", (e) => {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === "z") {
      e.preventDefault();
      e.shiftKey ? api.redo() : api.undo();
    } else if (mod && e.key.toLowerCase() === "y") {
      e.preventDefault();
      api.redo();
    } else if (e.key === "Tab" && !mod && !e.altKey) {
      e.preventDefault();
      api.insert("\t");
    } else if (e.key === "Enter" && !mod) {
      const n = window.getSelection().anchorNode;
      const el = n && (n.nodeType === 1 ? n : n.parentElement);
      if (el && el.closest("td")) {
        e.preventDefault();
        document.execCommand("insertLineBreak");
      }
    }
  });
  // copy / cut the MARKUP of the selection (the flags travel with the text)
  const copySel = (e) => {
    const sel = selection();
    if (!sel || sel.a === sel.b) return false;
    e.clipboardData.setData("text/plain", toMarkup(flatten(root).items.slice(sel.a, sel.b)));
    e.preventDefault();
    return true;
  };
  root.addEventListener("copy", copySel);
  root.addEventListener("cut", (e) => {
    if (copySel(e)) document.execCommand("delete");
  });
  // paste = plain text; marks / TABs in it are re-rendered as flags / cells
  root.addEventListener("paste", (e) => {
    e.preventDefault();
    api.insert(e.clipboardData.getData("text/plain"));
  });

  const api = {
    get markup() {
      return markup;
    },
    /// the markup the current DOM stands for (the round-trip check; equals `markup` after every edit)
    serialize: () => toMarkup(flatten(root).items),
    /// select visible items [a, b) (tests)
    select,
    /// show a markup without history (streaming read)
    async show(m) {
      markup = m;
      await paint(m);
    },
    /// load a markup as the new base of the history
    async load(m) {
      clearTimeout(typingTimer);
      typingTimer = null;
      markup = m;
      hist = [{ m, a: 0, b: 0 }];
      hpos = 0;
      await paint(m);
      changed();
    },
    setEditable(on) {
      root.contentEditable = on ? "true" : "false";
    },
    canUndo: () => hpos > 0 || !!typingTimer,
    canRedo: () => hpos < hist.length - 1,
    undo() {
      if (typingTimer) {
        flushTyping();
      }
      return jump(hpos - 1);
    },
    redo() {
      return jump(hpos + 1);
    },
    /// toggle a style tag on the selection (all chars have it → off, else on)
    async toggle(tag) {
      const sel = selection();
      if (!sel || sel.a === sel.b) return;
      const items = flatten(root).items;
      const chars = items.slice(sel.a, sel.b).filter((i) => !i.sep);
      const all = chars.length > 0 && chars.every((i) => i.tags.includes(tag));
      for (let i = sel.a; i < sel.b; i++) {
        const it = items[i];
        if (it.sep) continue;
        it.tags = all ? it.tags.filter((t) => t !== tag) : it.tags.includes(tag) ? it.tags : [...it.tags, tag];
      }
      await commit(toMarkup(items), sel.a, sel.b);
    },
    /// the tags in force at the caret (toolbar state)
    tagsAtCaret() {
      const sel = selection();
      if (!sel) return [];
      const items = flatten(root).items;
      const it = items[sel.a] && !items[sel.a].sep ? items[sel.a] : items[sel.a - 1];
      return it && !it.sep ? it.tags : [];
    },
    /// insert text at the caret (replacing the selection); marks / TABs / newlines in it take effect
    async insert(text) {
      const sel = selection() || { a: 0, b: 0 };
      const items = flatten(root).items;
      const before = toMarkup(items.slice(0, sel.a));
      const after = toMarkup(items.slice(sel.b));
      const m = before + text + after;
      // the caret goes after the inserted text: count its visible items after the re-render
      await paint(m);
      const total = flatten(root).items.length;
      const tail = items.length - sel.b;
      await commit(m, total - tail, total - tail);
    },
  };
  return api;
};
