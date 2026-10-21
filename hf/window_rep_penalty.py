# OCR Lite — Copyright (c) 2026 Bernhard Straub
# SPDX-License-Identifier: MIT

"""window_rep_penalty.py — a repetition penalty over a WINDOW of recent tokens, as a vLLM custom logits processor: the
`greedy_rp1.1` decoding of the model card (rp 1.1 over the last 20 tokens), as llama.cpp and OCR Lite apply it.

WHY: vLLM's native `repetition_penalty` penalises EVERY token of prompt + output seen so far (whole context, no window).
On long pages that taxes the whole page vocabulary (marks such as ⟨ ⟩ ⌈, line breaks …) and makes blocks loop; a
penalty over only the last N tokens does not.

SEMANTICS: window = the last N tokens of (prompt tokens + generated tokens); each UNIQUE token in it: logit < 0 →
logit × rp, else logit / rp. Flat, no counts.

USE: vllm serve … --logits-processors window_rep_penalty:WindowRepetitionPenalty   (with this file on PYTHONPATH)
  per request (OpenAI body, vLLM `vllm_xargs`): {"rep_penalty": …, "rep_window": …} — REQUIRED, no hidden default: a
  request without them is rejected.
  The NATIVE `repetition_penalty` must stay 1.0 — a request with another value is REJECTED (HTTP 400): two penalties
  stacking silently is exactly the mix-up this module removes. rep_penalty 1.0 = off.
"""
from __future__ import annotations

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates



def _params(sp: SamplingParams) -> tuple[float, int]:
    xa = sp.extra_args or {}
    if "rep_penalty" not in xa or "rep_window" not in xa:
        raise ValueError("vllm_xargs must carry rep_penalty and rep_window (the decoding preset) — this server has no default")
    return float(xa["rep_penalty"]), int(xa["rep_window"])


class WindowRepetitionPenalty(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        if sampling_params.repetition_penalty != 1.0:
            raise ValueError(
                f"repetition_penalty={sampling_params.repetition_penalty}: this server applies the windowed penalty "
                "(window_rep_penalty) — send repetition_penalty=1.0 and vllm_xargs {'rep_penalty': …, 'rep_window': …}")
        pen, win = _params(sampling_params)
        if pen <= 0 or win < 1:
            raise ValueError(f"rep_penalty must be > 0 and rep_window ≥ 1 (got {pen}, {win})")

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
        self.device = device
        # persistent-batch row → (penalty, window, prompt tail, live output-token list)
        self.reqs: dict[int, tuple[float, int, list[int], list[int]]] = {}

    def is_argmax_invariant(self) -> bool:
        return False                      # the penalty changes the greedy argmax — never skip it

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        def new_state(params, prompt_ids, output_ids):
            pen, win = _params(params)
            if pen == 1.0:
                return None
            tail = list(prompt_ids[-win:]) if prompt_ids is not None else []
            return pen, win, tail, output_ids
        process_dict_updates(self.reqs, batch_update, new_state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.reqs:
            return logits
        rows: list[int] = []
        cols: list[int] = []
        pens: list[float] = []
        for row, (pen, win, tail, out) in self.reqs.items():
            # out is long late in a page: slice it, never concatenate it (O(n) per step); short → prompt tail + out
            window = out[-win:] if len(out) >= win else (tail + list(out))[-win:]
            # async scheduling can leave -1 placeholders for a not-yet-sampled token: never index with them
            ids = {t for t in window if 0 <= t < logits.shape[1]}
            rows.extend([row] * len(ids))
            cols.extend(ids)
            pens.extend([pen] * len(ids))
        if not rows:
            return logits
        r = torch.tensor(rows, dtype=torch.long, device=logits.device)
        c = torch.tensor(cols, dtype=torch.long, device=logits.device)
        p = torch.tensor(pens, dtype=logits.dtype, device=logits.device)
        sel = logits[r, c]
        logits[r, c] = torch.where(sel < 0, sel * p, sel / p)
        return logits
