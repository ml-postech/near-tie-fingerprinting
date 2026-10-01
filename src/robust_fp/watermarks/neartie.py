# Copyright (c) 2026 the authors.
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Near-tie restriction as a wrapper around another watermark.

The near-tie support set of a position is defined by the base model alone:

    S_t = { v : logit_v > top-1 logit - tau_gap }

:class:`NearTieWrapper` runs the wrapped watermark and keeps only the part of its perturbation that
falls inside ``S_t``. For an additive watermark (KGW, SWEET) this zeroes the logit bias outside the
support set, which is the same rule :class:`~robust_fp.watermarks.kgw.KGWWatermark` applies with
``tie_restricted``. For MorphMark, whose perturbation is a probability redistribution, the wrapper
keeps the probability increases of near-tie green tokens, freezes every other token at its base
probability, and removes the kept mass from the red tokens in proportion to their base probability,
so the result is still a distribution.
"""
import torch

from .base import Watermark


class NearTieWrapper(Watermark):
    def __init__(self, base: Watermark, tau_gap: float):
        self.base = base
        self.tau_gap = float(tau_gap)

    @property
    def gate_name(self) -> str:
        return f"{self.base.gate_name}+neartie"

    def __getattr__(self, name):
        # Attributes that describe the green list (greenlist_masks, k, gamma, vocab_size, ...) are
        # those of the wrapped watermark.
        return getattr(object.__getattribute__(self, "base"), name)

    def support_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """``S_t`` over the full width of ``logits``: ``(..., V)`` boolean."""
        top1 = logits.max(dim=-1, keepdim=True).values
        return logits > (top1 - self.tau_gap)

    def watermark_logits_diff(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        support = self.support_mask(logits)
        if getattr(self.base, "morphmark", False):
            return self._morphmark_diff(input_ids, logits, support)
        diff = self.base.watermark_logits_diff(input_ids, logits)
        support = support.to(diff.device)[..., :diff.shape[-1]]
        # torch.where rather than a product: a -inf base logit (a token masked at decode time) has
        # an infinite diff, and inf * 0 is NaN.
        return torch.where(support, diff, torch.zeros_like(diff))

    def _morphmark_diff(self, input_ids, logits, support):
        f = logits.float()
        base_lp = torch.log_softmax(f, dim=-1)
        p = base_lp.exp()
        wm_lp = self.base.watermark_logits(input_ids, f).float()
        V = wm_lp.shape[-1]
        p, base_lp_v, support = p[..., :V], base_lp[..., :V], support[..., :V]
        dp = wm_lp.exp() - p
        increase = torch.where((dp > 0) & support, dp, torch.zeros_like(dp))
        kept = increase.sum(dim=-1, keepdim=True)
        red = dp < 0
        red_mass = (p * red).sum(dim=-1, keepdim=True).clamp_min(1e-9)
        removed = torch.where(red, (p / red_mass) * kept, torch.zeros_like(dp))
        p_new = (p + increase - removed).clamp_min(0)
        p_new = p_new / p_new.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        diff = (torch.log(p_new.clamp_min(1e-20)) - base_lp_v).to(logits.dtype)
        finite = torch.isfinite(base_lp_v)
        return torch.nan_to_num(torch.where(finite, diff, torch.zeros_like(diff)),
                                nan=0.0, posinf=0.0, neginf=0.0)

    def watermark_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        return logits + self.watermark_logits_diff(input_ids, logits)
