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

Every watermark is read with its own detector. A watermark that biases only some positions is
scored on those positions alone: for KGW-NT (and the near-tie wrapper) the positions where the base
model's top-1/top-2 logit gap is below ``tau_gap``; for SWEET the positions whose base-model
next-token entropy exceeds ``entropy_threshold``; for SWEET+near-tie both; for MorphMark+near-tie the
near-tie positions. That restricted count is
``elig_z``; the count over every position is ``all_z``. KGW, KGW-TK and MorphMark bias every
position and are read with ``all_z``.

Eligibility is decided by the unwatermarked base model, so the detector needs the base model, the
green list and the gate thresholds, and never the fingerprinted model itself.
"""
import math
from typing import Iterable, List, Optional, Tuple

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


def uses_eligibility(gate_name: str) -> bool:
    """Whether a watermark is read on a restricted set of positions (``elig_z``) or on every
    position (``all_z``)."""
    return "neartie" in gate_name or "sweet" in gate_name


def z_key(gate_name: str) -> str:
    return "elig_z" if uses_eligibility(gate_name) else "all_z"


class Detector:
    """Green/eligibility labelling of responses under one watermark.

    Args:
        greenlist_masks: the ``(k * |V|, |V|)`` green-list mask of the watermark.
        k: context width of the green-list seeding.
        gamma: green-list fraction.
        gate: the watermark's gate name (``kgw``, ``neartie``, ``sweet``, ``sweet+neartie``, ...).
        tau_gap: near-tie window, when the gate includes ``neartie``.
        entropy_threshold: entropy threshold, when the gate includes ``sweet``.
        tokenizer: the base model's tokenizer.
        base_model: the unwatermarked base model (``AutoModelForCausalLM``) on the mask's device;
            needed only when the gate restricts positions.
    """

    def __init__(self, greenlist_masks: torch.Tensor, k: int, gamma: float, gate: str, tokenizer,
                 tau_gap: Optional[float] = None, entropy_threshold: Optional[float] = None, base_model=None):
        self.greenlist_masks = greenlist_masks
        self.k = int(k)
        self.gamma = float(gamma)
        self.gate = gate
        self.tok = tokenizer
        self.base = base_model
        self.device = greenlist_masks.device
        if "neartie" in gate and tau_gap is None:
            raise ValueError("a near-tie gate needs tau_gap")
        if "sweet" in gate and entropy_threshold is None:
            raise ValueError("a SWEET gate needs entropy_threshold")
        self.tau_gap = float(tau_gap) if "neartie" in gate else None
        self.entropy_threshold = float(entropy_threshold) if "sweet" in gate else None
        if uses_eligibility(gate) and base_model is None:
            raise ValueError(f"the {gate} detector needs the base model to decide eligibility")

    @classmethod
    def from_watermark(cls, watermark: Watermark, tokenizer, base_model=None) -> "Detector":
        """The detector of a teacher built from a cell's configuration."""
        gate = watermark.gate_name
        return cls(watermark.greenlist_masks, watermark.k, watermark.gamma, gate, tokenizer,
                   tau_gap=watermark.tau_gap if "neartie" in gate else None,
                   entropy_threshold=watermark.entropy_threshold if "sweet" in gate else None,
                   base_model=base_model)

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
        """Eligibility of completion positions ``start .. L-1``, from the base model's prediction
        of each of them."""
        n = ids.shape[1] - start
        if not uses_eligibility(self.gate):
            return torch.ones(n, dtype=torch.bool, device=ids.device)
        logp = torch.log_softmax(self.base(ids).logits[0][start - 1:-1].float(), dim=-1)
        eligible = None
        if self.entropy_threshold is not None:
            entropy = -(logp.exp() * logp).sum(-1)
            eligible = entropy > self.entropy_threshold
        if self.tau_gap is not None:
            top2 = logp.topk(2, dim=-1).values
            near_tie = (top2[:, 0] - top2[:, 1]) < self.tau_gap
            eligible = near_tie if eligible is None else (eligible & near_tie)
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
        hashes = comp.unfold(0, self.k, 1).sum(-1)
        green = self.greenlist_masks[hashes[:-1], comp[self.k:]].tolist()
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
