# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Experiment configuration.

A configuration file has three sections::

    base_model: meta-llama/Llama-3.1-8B-Instruct
    watermark:
      type: kgw
      config: {gamma: 0.25, delta: 4, tie_restricted: true, tau_gap: 1.5}
    finetuning:
      training_args: {...}          # transformers.TrainingArguments
      watermark_datasets: [LucieFr]
      regularization_datasets: [AlpacaGPT4, OpenWebText]
      loss_types: [anti-watermark-tv, anti-watermark-tv]
      proportions: [0.6, 0.2, 0.2]
      custom_name: kgwnt_q50_d4

Unknown keys are rejected, so a misspelt option cannot silently fall back to a default.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from robust_fp.finetuning.datasets import DatasetType
from robust_fp.watermarks import KGWWatermark, NearTieWrapper, Watermark


class KGWWatermarkConfiguration(BaseModel):
    """Parameters of the KGW-family teachers; see :class:`robust_fp.watermarks.KGWWatermark`."""

    model_config = ConfigDict(extra="forbid")

    gamma: float = 0.25
    delta: float = 2.0
    seeding_scheme: str = "simple_1"
    # KGW-NT: bias only green tokens within tau_gap of the top logit.
    tie_restricted: bool = False
    tau_gap: Optional[float] = None
    # SWEET: bias green tokens only at positions whose next-token entropy exceeds the threshold.
    entropy_restricted: bool = False
    entropy_threshold: float = 1.0
    # KGW-TK: bias only green tokens among the top-k logits.
    topk_restricted: int = 0
    # MorphMark: probability redistribution instead of a logit bias.
    morphmark: bool = False
    morphmark_config: Optional[Dict[str, Any]] = None
    # Near-tie wrapper: restrict the perturbation of a SWEET or MorphMark teacher to the near-tie
    # support set of width tau_gap.
    neartie_wrap: bool = False

    @model_validator(mode="after")
    def _check(self):
        if self.neartie_wrap and self.tie_restricted:
            raise ValueError("neartie_wrap composes with SWEET or MorphMark; KGW-NT is tie_restricted alone")
        if (self.tie_restricted or self.neartie_wrap) and self.tau_gap is None:
            raise ValueError("tau_gap is required by tie_restricted and by neartie_wrap")
        if self.neartie_wrap and not (self.entropy_restricted or self.morphmark):
            raise ValueError("neartie_wrap requires a SWEET (entropy_restricted) or MorphMark base")
        return self

    def build(self, tokenizer, device: str = "cuda") -> Watermark:
        """Instantiate the teacher, building its green-list mask on ``device``."""
        wm = KGWWatermark(
            vocab=tokenizer.get_vocab(),
            gamma=self.gamma,
            delta=self.delta,
            seeding_scheme=self.seeding_scheme,
            tokenizer=tokenizer,
            tie_restricted=self.tie_restricted,
            tau_gap=self.tau_gap,
            entropy_restricted=self.entropy_restricted,
            entropy_threshold=self.entropy_threshold,
            topk_restricted=self.topk_restricted,
            morphmark=self.morphmark,
            morphmark_config=self.morphmark_config,
            device=device,
        )
        if self.neartie_wrap:
            wm = NearTieWrapper(wm, self.tau_gap)
        return wm

    @property
    def gate_name(self) -> str:
        """The name the built teacher reports, without building it."""
        if self.morphmark:
            base = "morphmark"
        elif self.tie_restricted:
            base = "neartie"
        elif self.entropy_restricted:
            base = "sweet"
        elif self.topk_restricted:
            base = "topk"
        else:
            base = "kgw"
        return base + ("+neartie" if self.neartie_wrap else "")


class WatermarkConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "kgw"
    config: KGWWatermarkConfiguration = Field(default_factory=KGWWatermarkConfiguration)

    @model_validator(mode="after")
    def _check(self):
        if self.type != "kgw":
            raise ValueError(f"unsupported watermark type {self.type!r}; only 'kgw' is implemented")
        return self

    def build(self, tokenizer, device: str = "cuda") -> Watermark:
        return self.config.build(tokenizer, device=device)


class FinetuningConfiguration(BaseModel):
    """Distillation data and optimisation settings.

    ``watermark_datasets`` are distilled toward the watermarked teacher; each of the
    ``regularization_datasets`` is trained with the loss named at the same index of ``loss_types``.
    ``proportions`` gives the sampling probability of every dataset, watermark datasets first, and
    ``lambdas`` the weight of every dataset's loss term.
    """

    model_config = ConfigDict(extra="forbid")

    dtype: str = "bfloat16"
    training_args: Dict[str, Any]
    watermark_datasets: List[DatasetType]
    regularization_datasets: List[DatasetType] = Field(default_factory=list)
    loss_types: List[str] = Field(default_factory=list)
    proportions: List[float]
    lambdas: List[float] = Field(default_factory=list)
    streaming: bool = True
    sequence_length: int = 512
    custom_name: str

    @model_validator(mode="after")
    def _check(self):
        n = len(self.watermark_datasets) + len(self.regularization_datasets)
        if not self.watermark_datasets:
            raise ValueError("at least one watermark dataset is required")
        if len(self.proportions) != n:
            raise ValueError("proportions must name every dataset, watermark datasets first")
        if abs(sum(self.proportions) - 1.0) > 1e-6:
            raise ValueError("proportions must sum to 1")
        if len(self.loss_types) != len(self.regularization_datasets):
            raise ValueError("every regularization dataset needs a loss type")
        if self.lambdas and len(self.lambdas) != n:
            raise ValueError("lambdas must name every dataset, or be omitted (all 1.0)")
        if not self.lambdas:
            self.lambdas = [1.0] * n
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"dtype must be float32, float16 or bfloat16, got {self.dtype!r}")
        return self


class MainConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: str
    watermark: WatermarkConfiguration
    finetuning: Optional[FinetuningConfiguration] = None

    @classmethod
    def from_yaml(cls, path: str) -> "MainConfiguration":
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)
