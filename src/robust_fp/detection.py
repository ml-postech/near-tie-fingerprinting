# Copyright (c) 2026 the authors.
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Fingerprint detection: the pooled green-token statistic.

A fingerprinted model is identified from ``N`` of its responses. Every completion token whose
preceding ``k`` tokens are known carries a green/red label under the secret green list; the
detector counts green labels over the responses and reports

    z = (G - gamma * T) / sqrt(gamma * (1 - gamma) * T)

where ``T`` is the number of distinct ``(k+1)``-grams over all responses (each distinct n-gram is
counted once, across responses) and ``G`` the number of green ones. Under the null hypothesis of an
unwatermarked model, ``z`` is approximately standard normal.

Every watermark is read with its own detector: a teacher that biases only some positions is scored
on those positions alone, as decided by :meth:`Watermark.eligibility` from the base model's
next-token distribution (the near-tie gap for KGW-NT and the near-tie wrapper, the entropy for
SWEET, both for SWEET+near-tie). That restricted count is ``elig_z``; the count over every position
is ``all_z``. Teachers that bias every position (KGW, KGW-TK, MorphMark) are read with ``all_z``.
The teacher's name and the key under which a row is read are recorded with it as ``teacher`` and ``z_key``.

Eligibility is decided by the unwatermarked base model, so the detector needs the base model, the
green list and the teacher's eligibility rule, and never the fingerprinted model itself.
"""
import math
from typing import Iterable, List, Tuple

import torch

from robust_fp.watermarks import Watermark

#: Tokens of the prompt kept as context for the base-model forward pass (the tail of the prompt).
PROMPT_CONTEXT = 200
#: Maximum length of prompt context plus completion scored per response.
MAX_SCORED_TOKENS = 1280


def z_score(green: int, total: int, gamma: float) -> float:
    if total <= 0:
        return float("nan")
    return (green - gamma * total) / math.sqrt(gamma * (1 - gamma) * total)


class Detector:
    """Green/eligibility labelling of responses under one watermark.

    Args:
        watermark: the teacher built from the cell's configuration (green list and eligibility rule).
        tokenizer: the base model's tokenizer.
        base_model: the unwatermarked base model (``AutoModelForCausalLM``) on the green list's
            device; needed when the watermark restricts the scored positions.
    """

    def __init__(self, watermark: Watermark, tokenizer, base_model=None):
        self.wm = watermark
        self.greenlist = watermark.greenlist
        self.k = self.greenlist.k
        self.gamma = self.greenlist.gamma
        self.tok = tokenizer
        self.base = base_model
        self.device = self.greenlist.device
        if watermark.restricts_positions and base_model is None:
            raise ValueError(f"the {watermark.name} detector needs the base model to decide eligibility")

    @property
    def z_key(self) -> str:
        return "elig_z" if self.wm.restricts_positions else "all_z"

    # ------------------------------------------------------------------ per response
    def _token_ids(self, prompt: str, completion: str):
        """Token ids of the scored window and the index where the completion starts."""
        prefix = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                              add_generation_prompt=True, tokenize=True)
        full = self.tok.apply_chat_template([{"role": "user", "content": prompt},
                                             {"role": "assistant", "content": completion}], tokenize=True)
        start = len(prefix)
        if start > PROMPT_CONTEXT:
            drop = start - PROMPT_CONTEXT
            full = full[:1] + full[1 + drop:]       # keep the first token (BOS), drop prompt tokens after it
            start -= drop
        return full[:MAX_SCORED_TOKENS], start

    @torch.no_grad()
    def eligibility(self, ids: torch.LongTensor, start: int) -> torch.Tensor:
        """Eligibility of completion positions ``start .. L-1`` from the base model's prediction of each."""
        n = ids.shape[1] - start
        if not self.wm.restricts_positions:
            return torch.ones(n, dtype=torch.bool, device=ids.device)
        logp = torch.log_softmax(self.base(ids).logits[0][start - 1:-1].float(), dim=-1)
        eligible = self.wm.eligibility(logp)
        if eligible is None:
            raise ValueError(f"{self.wm.name} restricts the scored positions but its eligibility() returns None")
        return eligible

    @torch.no_grad()
    def response_flags(self, prompt: str, completion: str) -> List[Tuple[tuple, bool, bool]]:
        """``[(ngram, green, eligible)]`` for every scored completion position.

        ``ngram`` is the ``(k+1)``-gram ending at the position, the key under which it is counted
        once across responses.
        """
        if not completion or not completion.strip():
            return []
        full, start = self._token_ids(prompt, completion)
        if len(full) <= start + self.k:
            return []
        ids = torch.tensor([full], device=self.device)
        comp = ids[0, start:]
        eligible = self.eligibility(ids, start)[self.k:].tolist()
        hashes = self.greenlist.hashes(comp)
        green = self.greenlist.is_green(hashes[:-1], comp[self.k:]).tolist()
        tokens = comp.tolist()
        return [(tuple(tokens[j:j + self.k + 1]), g, e) for j, (g, e) in enumerate(zip(green, eligible))]


class PooledCounts:
    """Running green counts over responses, with each distinct n-gram counted once."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.seen_all, self.seen_elig = set(), set()
        self.G_all = self.T_all = self.G_elig = self.T_elig = 0
        self.n_scored = 0

    def add(self, flags: Iterable[Tuple[tuple, bool, bool]]) -> None:
        added = False
        for ngram, green, eligible in flags:
            added = True
            if ngram not in self.seen_all:
                self.seen_all.add(ngram)
                self.T_all += 1
                self.G_all += int(green)
            if eligible and ngram not in self.seen_elig:
                self.seen_elig.add(ngram)
                self.T_elig += 1
                self.G_elig += int(green)
        self.n_scored += int(added)

    def counts(self, z_key: str) -> Tuple[int, int]:
        """``(G, T)`` under ``z_key``: the eligible positions for ``elig_z``, every position for ``all_z``."""
        return (self.G_elig, self.T_elig) if z_key == "elig_z" else (self.G_all, self.T_all)

    def z(self, z_key: str) -> float:
        return z_score(*self.counts(z_key), self.gamma)

    def summary(self) -> dict:
        g, Te, Ta = self.gamma, self.T_elig, self.T_all
        return dict(
            n_scored=self.n_scored,
            elig_green=round(self.G_elig / Te, 4) if Te else None,
            elig_z=round(z_score(self.G_elig, Te, g), 2) if Te else None,
            elig_frac=round(Te / Ta, 3) if Ta else None,
            all_green=round(self.G_all / Ta, 4) if Ta else None,
            all_z=round(z_score(self.G_all, Ta, g), 2) if Ta else None,
            all_n=Ta,
        )
