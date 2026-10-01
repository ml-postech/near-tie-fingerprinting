# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""KGW: a logit bias on green tokens (Kirchenbauer et al., 2023), with a pluggable token selector."""
from typing import Optional

import torch

from .base import Watermark
from .greenlist import GreenList
from .selectors import DenseSelector, TokenSelector


class KGW(Watermark):
    """Green-token logit bias with a pluggable token selector.

    At every position with a full context, ``delta`` is added to the logits of the green tokens the
    ``selector`` chooses: all of them for KGW, the near-tie ones for KGW-NT, the top-k ones for
    KGW-TK, all of them at high-entropy positions for SWEET. The teacher's name, eligibility rule
    and ``restricts_positions`` are the selector's.
    """

    def __init__(self, greenlist: GreenList, delta: float, selector: Optional[TokenSelector] = None):
        self.greenlist = greenlist
        self.delta = float(delta)
        self.selector = selector or DenseSelector()
        if not getattr(self.selector, "name", None):
            raise ValueError(f"{type(self.selector).__name__} must define a `name`")

    @property
    def name(self) -> str:
        return self.selector.name

    def eligibility(self, logp):
        return self.selector.eligibility(logp)

    @property
    def restricts_positions(self) -> bool:
        return self.selector.restricts_positions

    def _prepare(self, input_ids, logits):
        input_ids = input_ids.to(self.greenlist.device)
        logits = logits.to(self.greenlist.device)
        k = self.greenlist.k
        green = self.greenlist[self.greenlist.hashes(input_ids)]
        return logits, k, green, self.selector.select(logits[..., k - 1:, :], green)

    def watermark_logits_diff(self, input_ids, logits):
        device = input_ids.device
        logits, k, green, boost = self._prepare(input_ids, logits)
        diff = torch.zeros_like(logits)
        diff[..., k - 1:, :green.shape[-1]][boost] = self.delta
        return diff.to(device)

    def watermark_logits(self, input_ids, logits):
        # Applied in place on a clone rather than as `logits + diff`, to avoid a second full-width
        # tensor during training.
        device = input_ids.device
        logits, k, green, boost = self._prepare(input_ids, logits.to(self.greenlist.device).clone())
        sl = logits[..., k - 1:, :green.shape[-1]]
        sl += boost.to(sl.dtype) * self.delta
        return logits.to(device)
