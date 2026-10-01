# Copyright (c) 2026 the authors.
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Which green tokens a KGW teacher biases, and which positions its detector scores.

A :class:`TokenSelector` decides, from the base model's logits at a position, the subset of green
tokens that receives the logit bias. When the rule restricts positions (near-tie, entropy), the
selector also exposes the position-level criterion the detector uses, computed from the base
model's log-probabilities alone so it stays independent of the secret key.

    DenseSelector      every green token                               (KGW)
    NearTieSelector    green tokens within ``tau_gap`` of the top logit  (KGW-NT)
    EntropySelector    every green token, at positions whose next-token entropy exceeds a
                       threshold                                        (SWEET)
    TopKSelector       green tokens among the ``k`` largest logits       (KGW-TK)
"""
from typing import Optional

import torch


class TokenSelector:
    """A rule for which green tokens receive the bias at a position.

    Subclasses set :attr:`name`, implement :meth:`select`, and, when the rule depends on the
    position, set ``restricts_positions = True`` and implement :meth:`eligibility`.
    """

    #: Identifier recorded with detection results.
    name: str
    #: Whether :meth:`eligibility` restricts the scored positions.
    restricts_positions: bool = False

    def select(self, logits: torch.FloatTensor, green: torch.BoolTensor) -> torch.BoolTensor:
        """Biased tokens: ``(..., V)`` boolean. ``logits`` spans the model's full logit width,
        ``green`` the tokenizer's ``V`` tokens."""
        raise NotImplementedError

    def eligibility(self, logp: torch.FloatTensor) -> Optional[torch.BoolTensor]:
        """Positions the detector scores, from base log-probabilities ``(positions, vocab)``;
        ``None`` means every position."""
        return None


class DenseSelector(TokenSelector):
    """Every green token (plain KGW)."""

    name = "kgw"

    def select(self, logits, green):
        return green


class NearTieSelector(TokenSelector):
    """Green tokens whose logit lies within ``tau_gap`` of the top logit: the near-tie support set
    ``{v in G_t : l(v*) - l(v) < tau_gap}``. Detection scores the positions whose top-1/top-2
    logit gap is below ``tau_gap``."""

    name = "neartie"
    restricts_positions = True

    def __init__(self, tau_gap: float):
        if tau_gap is None:
            raise ValueError("the near-tie window tau_gap is calibrated per model and has no default")
        self.tau_gap = float(tau_gap)

    def support(self, logits: torch.FloatTensor) -> torch.BoolTensor:
        """The near-tie support set over the full width of ``logits``: ``(..., V)`` boolean."""
        top1 = logits.max(dim=-1, keepdim=True).values
        return logits > (top1 - self.tau_gap)

    def select(self, logits, green):
        return green & self.support(logits)[..., :green.shape[-1]]

    def eligibility(self, logp):
        top2 = logp.topk(2, dim=-1).values
        return (top2[:, 0] - top2[:, 1]) < self.tau_gap


class EntropySelector(TokenSelector):
    """Every green token, at positions whose next-token Shannon entropy exceeds ``threshold``
    (Lee et al., 2024). Detection scores those positions."""

    name = "sweet"
    restricts_positions = True

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def select(self, logits, green):
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(torch.where(probs > 0, probs * probs.log(), probs.new_zeros(()))).sum(dim=-1)
        return green & (entropy > self.threshold).unsqueeze(-1)

    def eligibility(self, logp):
        return -(logp.exp() * logp).sum(-1) > self.threshold


class TopKSelector(TokenSelector):
    """Green tokens among the ``k`` largest logits. Every position is scored at detection."""

    name = "topk"

    def __init__(self, k: int):
        if int(k) <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)

    def select(self, logits, green):
        kth = logits.topk(self.k, dim=-1).values[..., -1:]
        return green & (logits[..., :green.shape[-1]] >= kth)
