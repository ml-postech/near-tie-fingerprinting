# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Distillation teachers.

    GreenList       the keyed partition of the vocabulary shared by every teacher
    Watermark       the interface a teacher implements (watermarked logits, and which positions its
                    detector scores)
    KGW             green-token logit bias; which green tokens are biased is a TokenSelector
    MorphMark       probability-mass redistribution toward the green list
    NearTieWrapper  restricts any teacher's perturbation to the near-tie support set
"""
from .base import LARGE_VOCAB, Watermark
from .greenlist import GreenList
from .kgw import KGW
from .morphmark import MorphMark
from .neartie import NearTieWrapper
from .selectors import DenseSelector, EntropySelector, NearTieSelector, TokenSelector, TopKSelector

__all__ = ["LARGE_VOCAB", "Watermark", "GreenList", "KGW", "MorphMark", "NearTieWrapper",
           "TokenSelector", "DenseSelector", "NearTieSelector", "EntropySelector", "TopKSelector"]
