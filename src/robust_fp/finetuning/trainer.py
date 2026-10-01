# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

import torch
from transformers import Trainer

from robust_fp.finetuning.losses import LOSS_TYPES, DistillationLosses, LossRegistry
from robust_fp.watermarks import LARGE_VOCAB, Watermark


class DistillationTrainer(Trainer):
    """``Trainer`` whose loss is the watermark-distillation objective.

    Every batch runs the frozen teacher (the unmodified base model) alongside the student; the
    per-sequence label selects the loss term (see :mod:`robust_fp.finetuning.losses`).
    """

    def __init__(self, watermark: Watermark, teacher_model, registry: LossRegistry, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry = registry
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.losses = DistillationLosses(watermark)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        inputs["labels"] = inputs["input_ids"].clone()
        lambdas, loss_types = self.registry(labels)

        outputs = model(**inputs)
        student_logits = outputs.logits
        with torch.no_grad():
            teacher_logits = self.teacher_model(**inputs).logits

        attention_mask = inputs["attention_mask"]
        total = torch.zeros((), device=student_logits.device)
        for loss_type in torch.unique(loss_types).tolist():
            rows = (loss_types == loss_type).nonzero(as_tuple=True)[0]
            term = (self.losses.watermark_loss if loss_type == LOSS_TYPES["watermark"]
                    else self.losses.anti_watermark_tv_loss)
            # index_select rather than boolean indexing: on the (batch, seq, vocab) logits the
            # boolean-index kernel fails above ~5e8 elements (torch 2.8).
            total = total + term(
                input_ids=inputs["input_ids"].index_select(0, rows),
                attention_mask=attention_mask.index_select(0, rows),
                logits=student_logits.index_select(0, rows),
                teacher_logits=teacher_logits.index_select(0, rows),
                lambdas=lambdas.index_select(0, rows),
            )

        # transformers >= 4.53 returns a custom loss without dividing it by
        # gradient_accumulation_steps. The loss is divided here for vocabularies of LARGE_VOCAB tokens
        # and more (gemma-2) and returned unscaled for the other models; both branches are what the
        # training runs of the paper used.
        if student_logits.shape[-1] >= LARGE_VOCAB and self.args.gradient_accumulation_steps > 1:
            total = total / self.args.gradient_accumulation_steps
        return (total, outputs) if return_outputs else total
