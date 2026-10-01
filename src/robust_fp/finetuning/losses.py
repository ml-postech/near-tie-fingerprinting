# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""The two loss terms of watermark distillation.

    watermark            forward KL toward the watermarked teacher distribution
    anti-watermark-tv    forward KL toward the clean teacher distribution, plus the total-variation
                         excess the student puts on the tokens the watermark would have boosted

Each training sequence carries a label that says which term it is trained with and with what
weight; :class:`LossRegistry` maps labels to (weight, term).
"""
import torch

from robust_fp.watermarks import LARGE_VOCAB, Watermark

LOSS_TYPES = {"watermark": 0, "anti-watermark-tv": 1}


class LossRegistry:
    def __init__(self, device):
        self.device = device
        self.lambdas = torch.tensor([], device=device)
        self.loss_types = torch.tensor([], device=device, dtype=torch.int)

    def add_dataset(self, lambd: float, loss_type: str) -> int:
        """Register a dataset and return the label its sequences carry."""
        self.lambdas = torch.cat([self.lambdas, torch.tensor([lambd], device=self.device)])
        self.loss_types = torch.cat([self.loss_types,
                                     torch.tensor([LOSS_TYPES[loss_type]], device=self.device, dtype=torch.int)])
        return len(self.lambdas) - 1

    def __call__(self, labels: torch.Tensor):
        return self.lambdas[labels], self.loss_types[labels]


class DistillationLosses:
    def __init__(self, watermark: Watermark):
        self.watermark = watermark
        self.kl = torch.nn.KLDivLoss(reduction="none", log_target=True)

    def watermark_loss(self, input_ids, attention_mask, logits, teacher_logits, lambdas):
        return self.watermark.distillation_loss(input_ids, logits, teacher_logits, lambdas,
                                                attention_mask=attention_mask)

    def anti_watermark_tv_loss(self, input_ids, attention_mask, logits, teacher_logits, lambdas):
        boosted = self.watermark.watermark_logits_diff(input_ids, teacher_logits) > 0

        fp32 = logits.shape[-1] >= LARGE_VOCAB
        student = logits.float() if fp32 else logits
        teacher = teacher_logits.float() if fp32 else teacher_logits
        student_prob = torch.nn.functional.softmax(student, dim=-1)
        teacher_prob = torch.nn.functional.softmax(teacher, dim=-1)

        positive_tv = torch.sum(boosted * torch.clamp(student_prob - teacher_prob, min=0.0), dim=-1)
        positive_tv = torch.sum(positive_tv * attention_mask, dim=-1)
        positive_tv = torch.mean(positive_tv * lambdas.view(-1)) / logits.shape[1]

        kl = self.kl(torch.nn.functional.log_softmax(student, dim=-1),
                     torch.nn.functional.log_softmax(teacher, dim=-1))
        kl = torch.sum(torch.sum(kl, dim=-1) * attention_mask, dim=-1)
        kl = torch.mean(kl * lambdas.view(-1)) / logits.shape[1]
        return kl + positive_tv
