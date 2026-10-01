# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Watermark-distilled fingerprints for open-weight language models.

Sub-packages:
    watermarks   the distillation teachers (KGW, KGW-NT, KGW-TK, SWEET, MorphMark, and the
                 near-tie wrapper that composes with SWEET and MorphMark)
    finetuning   the distillation objective and the training loop
    detection    the pooled green-token statistic used to read a fingerprint from responses
    results      readers for the result trees the evaluation scripts produce
"""

__version__ = "1.0.0"
