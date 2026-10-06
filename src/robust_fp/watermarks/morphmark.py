# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""MorphMark (Wang et al., ACL 2025) as a distillation teacher.

Reproduces the ``_bias_greenlist_logits`` rule of the MarkLLM reference implementation, vectorised
over a batch of positions. Given the green mask ``g`` of a position and ``P_G``, the base model's
probability mass on green tokens:

    r     = 0                        if P_G <  p_0
          = exp(k_exp * P_G) - 1     if P_G >= p_0        (``variant="exp"``; ``linear``: k_linear * P_G,
                                                          ``log``: log(k_log * P_G + 1))
    beta  = r * (1 - P_G)
    p[green] += p[green] / P_G       * beta
    p[red]   -= p[red]   / (1 - P_G) * beta

The teacher returns ``log p`` of the redistributed distribution. Every position is scored at
detection with the standard green-token count.
"""
import torch

from .base import Watermark
from .greenlist import GreenList

VARIANTS = ("exp", "linear", "log")


class MorphMark(Watermark):
    """Probability-mass redistribution from red to green tokens.

    Args:
        greenlist: the shared green list (MorphMark's own setting is ``gamma = 0.5``).
        variant: the strength function of ``P_G``: ``exp``, ``linear`` or ``log``.
        k_exp, k_linear, k_log: the parameter of each variant.
        p_0: no redistribution at positions whose green mass is below ``p_0``.
    """

    name = "morphmark"

    def __init__(self, greenlist: GreenList, variant: str = "exp", k_exp: float = 1.30, k_linear: float = 1.55,
                 k_log: float = 2.15, p_0: float = 0.15):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.greenlist = greenlist
        self.variant = variant
        self.k_exp = float(k_exp)
        self.k_linear = float(k_linear)
        self.k_log = float(k_log)
        self.p_0 = float(p_0)

    @torch.no_grad()
    def redistribute(self, logits: torch.Tensor, green: torch.Tensor) -> torch.Tensor:
        """Watermarked log-probabilities for ``logits`` ``(..., V)`` under the green mask ``(..., V)``."""
        f = logits.float()
        probs = torch.softmax(f, dim=-1)
        g = green.to(probs.dtype)
        P_G = (probs * g).sum(dim=-1, keepdim=True).clamp(1e-6, 1 - 1e-6)

        if self.variant == "linear":
            r = self.k_linear * P_G
        elif self.variant == "log":
            r = torch.log(self.k_log * P_G + 1)
        else:
            r = torch.exp(self.k_exp * P_G) - 1
        r = torch.where(P_G < self.p_0, torch.zeros_like(r), r)
        beta = r * (1 - P_G)

        gw = probs * g
        rw = probs * (1 - g)
        gsum = gw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        rsum = rw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        probs = probs + (gw / gsum) * beta - (rw / rsum) * beta
        probs = torch.nan_to_num(probs, nan=0.0).clamp_min(0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return torch.log(probs.clamp_min(1e-20)).to(logits.dtype)

    def _green(self, input_ids, width):
        # MorphMark renormalises over the whole logit width, so the mask covers padding columns too.
        return self.greenlist.widen(self.greenlist[self.greenlist.hashes(input_ids)], width)

    def watermark_logits_diff(self, input_ids, logits):
        device = input_ids.device
        input_ids, logits = input_ids.to(self.greenlist.device), logits.to(self.greenlist.device)
        k = self.greenlist.k
        diff = torch.zeros_like(logits)
        sl = logits[..., k - 1:, :]
        diff[..., k - 1:, :] = self.redistribute(sl, self._green(input_ids, logits.shape[-1])) - sl
        return diff.to(device)

    def watermark_logits(self, input_ids, logits):
        device = input_ids.device
        input_ids, logits = input_ids.to(self.greenlist.device), logits.to(self.greenlist.device).clone()
        k = self.greenlist.k
        logits[..., k - 1:, :] = self.redistribute(logits[..., k - 1:, :], self._green(input_ids, logits.shape[-1]))
        return logits.to(device)

    def restricted_logits_diff(self, input_ids, logits, support):
        """The redistribution restricted to ``support``: the probability increases of green tokens
        inside the support are kept, every token outside it is frozen at its base probability, and
        the kept mass is removed from the red tokens in proportion to their base probability, so the
        result is still a distribution."""
        f = logits.float()
        base_lp = torch.log_softmax(f, dim=-1)
        p = base_lp.exp()
        wm_lp = self.watermark_logits(input_ids, f).float()
        V = wm_lp.shape[-1]
        p, base_lp_v, support = p[..., :V], base_lp[..., :V], support.to(wm_lp.device)[..., :V]
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
        return torch.nan_to_num(torch.where(finite, diff, torch.zeros_like(diff)), nan=0.0, posinf=0.0, neginf=0.0)
