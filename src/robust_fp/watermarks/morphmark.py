# Copyright (c) 2026 the authors.
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""MorphMark (Wang et al., ACL 2025) as a distillation teacher.

Reproduces the ``_bias_greenlist_logits`` rule of the MarkLLM reference implementation, vectorised
over a batch of positions. Given the green-list mask ``g`` of a position and ``P_G``, the base
model's probability mass on green tokens:

    r     = 0                        if P_G <  p_0
          = exp(k_exp * P_G) - 1     if P_G >= p_0        (``type="exp"``; linear and log variants below)
    beta  = r * (1 - P_G)
    p[green] += p[green] / P_G       * beta
    p[red]   -= p[red]   / (1 - P_G) * beta

The teacher returns ``log p`` of the redistributed distribution. The green list itself comes from
the surrounding :class:`~robust_fp.watermarks.kgw.KGWWatermark`, so decode-time MorphMark and the
distilled model share the same green sets.
"""
import torch


class MorphMarkBias:
    def __init__(self, type="exp", k_exp=1.30, k_linear=1.55, k_log=2.15, p_0=0.15):
        self.type = type
        self.k_exp = float(k_exp)
        self.k_linear = float(k_linear)
        self.k_log = float(k_log)
        self.p_0 = float(p_0)

    @torch.no_grad()
    def apply(self, logits: torch.Tensor, green_mask: torch.Tensor) -> torch.Tensor:
        """Watermarked log-probabilities for ``logits`` ``(..., V)`` under the green mask ``(..., V)``."""
        f = logits.float()
        probs = torch.softmax(f, dim=-1)
        g = green_mask.to(probs.dtype)
        P_G = (probs * g).sum(dim=-1, keepdim=True).clamp(1e-6, 1 - 1e-6)

        if self.type == "linear":
            r = self.k_linear * P_G
        elif self.type == "log":
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
