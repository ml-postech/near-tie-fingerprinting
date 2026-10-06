# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Near-tie restriction as a wrapper around another watermark.

The near-tie support set of a position is defined by the base model alone:

    S_t = { v : logit_v > top-1 logit - tau_gap }

:class:`NearTieWrapper` keeps only the part of the wrapped watermark's perturbation that falls
inside ``S_t``, through the wrapped watermark's :meth:`~robust_fp.watermarks.Watermark.restricted_logits_diff`:
for an additive teacher (KGW, SWEET) the bias outside the support is zeroed, which is the rule
:class:`~robust_fp.watermarks.selectors.NearTieSelector` applies inside KGW; MorphMark restricts its
probability redistribution. Detection scores the positions that satisfy the wrapped watermark's
criterion (if any) and the near-tie criterion.
"""
from .base import Watermark
from .selectors import NearTieSelector


class NearTieWrapper(Watermark):
    """Restrict ``base``'s perturbation to the near-tie support set of width ``tau_gap``."""

    def __init__(self, base: Watermark, tau_gap: float):
        self.base = base
        self.greenlist = base.greenlist
        self.near_tie = NearTieSelector(tau_gap)

    @property
    def name(self) -> str:
        return f"{self.base.name}+neartie"

    @property
    def restricts_positions(self) -> bool:
        return True

    def eligibility(self, logp):
        near_tie = self.near_tie.eligibility(logp)
        inner = self.base.eligibility(logp)
        return near_tie if inner is None else (inner & near_tie)

    def watermark_logits_diff(self, input_ids, logits):
        return self.base.restricted_logits_diff(input_ids, logits, self.near_tie.support(logits))
