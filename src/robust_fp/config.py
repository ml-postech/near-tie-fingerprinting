# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Experiment configuration.

A configuration file has three sections::

    base_model: meta-llama/Llama-3.1-8B-Instruct
    watermark:
      type: kgw                     # always kgw: every teacher is keyed by a KGW green list; the flags pick the teacher
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
from robust_fp.watermarks import (KGW, DenseSelector, EntropySelector, GreenList, MorphMark, NearTieSelector,
                                  NearTieWrapper, TokenSelector, TopKSelector, Watermark)
from robust_fp.watermarks.morphmark import VARIANTS as MORPHMARK_VARIANTS


class MorphMarkConfiguration(BaseModel):
    """Parameters of :class:`~robust_fp.watermarks.MorphMark` (``morphmark_config`` in YAML)."""

    model_config = ConfigDict(extra="forbid")

    variant: str = "exp"
    k_exp: float = 1.30
    k_linear: float = 1.55
    k_log: float = 2.15
    p_0: float = 0.15

    @model_validator(mode="after")
    def _check(self):
        if self.variant not in MORPHMARK_VARIANTS:
            raise ValueError(f"variant must be one of {MORPHMARK_VARIANTS}, got {self.variant!r}")
        return self


class KGWWatermarkConfiguration(BaseModel):
    """Parameters of the teachers in :mod:`robust_fp.watermarks`, as flat YAML flags.

    The flags select the teacher class::

        (none)                            KGW(greenlist, delta, DenseSelector())
        tie_restricted + tau_gap          KGW(greenlist, delta, NearTieSelector(tau_gap))       KGW-NT
        entropy_restricted + threshold    KGW(greenlist, delta, EntropySelector(threshold))     SWEET
        topk_restricted = k               KGW(greenlist, delta, TopKSelector(k))                KGW-TK
        morphmark + morphmark_config      MorphMark(greenlist, **morphmark_config)
        neartie_wrap + tau_gap            NearTieWrapper(<SWEET or MorphMark teacher>, tau_gap)

    At most one of ``tie_restricted``, ``entropy_restricted``, ``topk_restricted`` and ``morphmark``
    may be set; with none set the teacher is plain KGW. ``neartie_wrap`` composes the near-tie
    restriction with a SWEET or MorphMark teacher (KGW-NT itself is ``tie_restricted``). ``delta`` is
    the logit bias of the KGW-family teachers and is not used by MorphMark.
    """

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
    morphmark_config: Optional[MorphMarkConfiguration] = None
    # Near-tie wrapper: restrict the perturbation of a SWEET or MorphMark teacher to the near-tie
    # support set of width tau_gap.
    neartie_wrap: bool = False

    @model_validator(mode="after")
    def _check(self):
        selected = [n for n, on in (("tie_restricted", self.tie_restricted), ("entropy_restricted", self.entropy_restricted),
                                    ("topk_restricted", bool(self.topk_restricted)), ("morphmark", self.morphmark)) if on]
        if len(selected) > 1:
            raise ValueError(f"at most one of tie_restricted / entropy_restricted / topk_restricted / morphmark "
                             f"may be set; got {selected}")
        if self.topk_restricted < 0:
            raise ValueError("topk_restricted must be a positive k, or 0 for no top-k restriction")
        if self.neartie_wrap and self.tie_restricted:
            raise ValueError("neartie_wrap composes with SWEET or MorphMark; KGW-NT is tie_restricted alone")
        if (self.tie_restricted or self.neartie_wrap) and self.tau_gap is None:
            raise ValueError("tau_gap is required by tie_restricted and by neartie_wrap")
        if self.neartie_wrap and not (self.entropy_restricted or self.morphmark):
            raise ValueError("neartie_wrap requires a SWEET (entropy_restricted) or MorphMark base")
        if self.morphmark_config is not None and not self.morphmark:
            raise ValueError("morphmark_config is only read when morphmark is set")
        return self

    def _selector(self) -> TokenSelector:
        """The token selector of a KGW-family teacher (not for ``morphmark``)."""
        if self.tie_restricted:
            return NearTieSelector(self.tau_gap)
        if self.entropy_restricted:
            return EntropySelector(self.entropy_threshold)
        if self.topk_restricted:
            return TopKSelector(self.topk_restricted)
        return DenseSelector()

    def build_greenlist(self, tokenizer, device: str = "cuda") -> GreenList:
        """The green list this configuration uses (a function of gamma and the seeding scheme only)."""
        return GreenList(tokenizer.get_vocab(), gamma=self.gamma, seeding_scheme=self.seeding_scheme,
                         tokenizer=tokenizer, device=device)

    def build(self, tokenizer, device: str = "cuda", greenlist: Optional[GreenList] = None) -> Watermark:
        """Instantiate the teacher. Pass ``greenlist`` to share one mask between teachers."""
        if greenlist is None:
            greenlist = self.build_greenlist(tokenizer, device)
        if self.morphmark:
            wm = MorphMark(greenlist, **(self.morphmark_config or MorphMarkConfiguration()).model_dump())
        else:
            wm = KGW(greenlist, self.delta, self._selector())
        if self.neartie_wrap:
            wm = NearTieWrapper(wm, self.tau_gap)
        return wm

    @property
    def name(self) -> str:
        """The ``name`` the built teacher reports, without building it."""
        base = MorphMark.name if self.morphmark else self._selector().name
        return base + ("+neartie" if self.neartie_wrap else "")


class WatermarkConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "kgw"
    config: KGWWatermarkConfiguration = Field(default_factory=KGWWatermarkConfiguration)

    @model_validator(mode="after")
    def _check(self):
        if self.type != "kgw":
            raise ValueError(f"unsupported watermark type {self.type!r}; every teacher here is keyed by a KGW green "
                             f"list and selected by the flags of `config`, so `type` is always 'kgw'")
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
