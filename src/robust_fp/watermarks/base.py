# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

import torch

#: Vocabularies at least this large are handled in float32 in the distillation losses.
LARGE_VOCAB = 200_000


class Watermark:
    """A watermark used as a distillation teacher.

    A watermark maps the base model's next-token logits to watermarked logits. The student is
    distilled toward the watermarked distribution with a forward KL divergence, computed by
    :meth:`distillation_loss`.
    """

    #: Short name of the watermark, recorded next to every detection result.
    gate_name: str = "kgw"

    def watermark_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        """Return the watermarked logits for ``logits`` of shape ``(batch, seq_len, vocab)``.

        The input tensor is not modified.
        """
        raise NotImplementedError

    def watermark_logits_diff(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        """Return ``watermark_logits(input_ids, logits) - logits``, without modifying ``logits``."""
        raise NotImplementedError

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
