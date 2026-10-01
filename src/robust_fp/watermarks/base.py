# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""The teacher interface shared by every watermark."""
from typing import Optional

import torch

from .greenlist import GreenList

#: Vocabularies at least this large are handled in float32 in the distillation losses.
LARGE_VOCAB = 200_000


class Watermark:
    """A watermark used as a distillation teacher.

    A teacher maps the base model's next-token logits to watermarked logits; the student is
    distilled toward the watermarked distribution with the forward KL of :meth:`distillation_loss`.

    A teacher also defines its own detector. Detection counts green tokens over the positions the
    teacher could have influenced: :meth:`eligibility` returns those positions from the base model's
    log-probabilities, or ``None`` when every position counts. Both the eligibility rule and the
    green list are independent of the response text, so the null green-token rate stays ``gamma``.

    Subclasses implement :meth:`watermark_logits_diff`, set :attr:`name` and :attr:`greenlist`,
    override :meth:`eligibility` and :attr:`restricts_positions` if they restrict the scored
    positions, and override :meth:`watermark_logits` and :meth:`restricted_logits_diff` if their
    perturbation is not an additive logit bias.
    """

    #: Short identifier recorded with every detection result.
    name: str = "watermark"
    #: The green list the teacher and its detector share.
    greenlist: GreenList

    def watermark_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        """Watermarked logits for ``logits`` of shape ``(batch, seq_len, vocab)``; the input is not modified."""
        return logits + self.watermark_logits_diff(input_ids, logits)

    def watermark_logits_diff(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        """``watermark_logits(input_ids, logits) - logits``, without modifying ``logits``."""
        raise NotImplementedError

    def restricted_logits_diff(self, input_ids: torch.LongTensor, logits: torch.FloatTensor,
                               support: torch.BoolTensor) -> torch.FloatTensor:
        """:meth:`watermark_logits_diff` with the perturbation kept only inside ``support`` ``(..., V)``.

        The default zeroes the bias outside the support, which is exact for an additive teacher;
        a teacher whose perturbation is not ``logits + diff`` overrides it.
        """
        diff = self.watermark_logits_diff(input_ids, logits)
        support = support.to(diff.device)[..., :diff.shape[-1]]
        # torch.where rather than a product: a -inf base logit (a token masked at decode time) has
        # an infinite diff, and inf * 0 is NaN.
        return torch.where(support, diff, torch.zeros_like(diff))

    def eligibility(self, logp: torch.FloatTensor) -> Optional[torch.BoolTensor]:
        """Positions the detector scores, from the base model's log-probabilities ``logp`` of
        shape ``(positions, vocab)``; ``None`` means every position."""
        return None

    @property
    def restricts_positions(self) -> bool:
        """Whether detection is restricted to eligible positions (``elig_z``) rather than all (``all_z``)."""
        return False

    def distillation_loss(
        self,
        input_ids: torch.LongTensor,
        student_logits: torch.FloatTensor,
        teacher_logits: torch.FloatTensor,
        lambdas: torch.FloatTensor,
        attention_mask: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """Forward KL from the watermarked teacher distribution to the student, per sequence.

        The divergence is summed over the vocabulary and over the non-padding positions, weighted by
        the per-sequence ``lambdas``, averaged over the batch, and divided by the sequence length.
        Vocabularies of ``LARGE_VOCAB`` tokens and more are handled in float32 for numerical stability.
        """
        watermarked_logits = self.watermark_logits(input_ids, teacher_logits)
        fp32 = student_logits.shape[-1] >= LARGE_VOCAB
        sl = student_logits.float() if fp32 else student_logits
        wl = watermarked_logits.float() if fp32 else watermarked_logits
        loss_fct = torch.nn.KLDivLoss(reduction="none", log_target=True)
        loss = loss_fct(
            torch.nn.functional.log_softmax(sl, dim=-1),
            torch.nn.functional.log_softmax(wl, dim=-1),
        )
        loss = torch.sum(loss, dim=-1)
        if attention_mask is not None:
            loss = loss * attention_mask
        loss = torch.sum(loss, dim=-1)
        return torch.mean(loss * lambdas.view(-1)) / student_logits.shape[1]
