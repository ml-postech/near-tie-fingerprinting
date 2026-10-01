# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""KGW-family distillation teachers.

All teachers share the KGW green list: at every position, a pseudo-random ``gamma`` fraction of the
vocabulary is green, seeded by the preceding token. They differ in which green tokens receive the
logit bias ``delta`` (or, for MorphMark, in how the green mass is redistributed):

    KGW        every green token                                (Kirchenbauer et al., 2023)
    KGW-NT     green tokens within ``tau_gap`` of the top logit, the near-tie support set
    KGW-TK     green tokens among the ``topk_restricted`` highest logits
    SWEET      every green token, at positions whose next-token entropy exceeds
               ``entropy_threshold``                            (Lee et al., 2024)
    MorphMark  probability-space redistribution from red to green tokens
                                                                (Wang et al., 2025)

Detection is not implemented here: :mod:`robust_fp.detection` scores responses from the green-list
mask this class builds, so that generation-side and detection-side green lists are one object.
"""
from typing import Mapping, Optional

import torch
from tqdm.auto import tqdm

from ..base import Watermark
from ..morphmark import MorphMarkBias
from .greenlist import GreenlistSeeder


class KGWWatermark(Watermark):
    def __init__(
        self,
        vocab: Mapping[str, int],
        gamma: float = 0.25,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        tokenizer=None,
        tie_restricted: bool = False,
        tau_gap: Optional[float] = None,
        entropy_restricted: bool = False,
        entropy_threshold: float = 1.0,
        topk_restricted: int = 0,
        morphmark: bool = False,
        morphmark_config: Optional[dict] = None,
        device: str = "cuda",
    ):
        """Build the teacher and its green-list mask.

        Args:
            vocab: the tokenizer vocabulary (``tokenizer.get_vocab()``); its size fixes the mask width.
            gamma: green-list fraction.
            delta: logit bias added to the selected green tokens.
            seeding_scheme: green-list seeding scheme; ``simple_1`` hashes the previous token.
            tokenizer: when given, its special tokens (eos, bos, pad, unk) are never green and never
                seed a green list.
            tie_restricted: KGW-NT. Bias only green tokens whose logit lies within ``tau_gap`` of
                the top logit.
            tau_gap: width of the near-tie window, in logits. Required by ``tie_restricted``; it is
                calibrated per model (see ``scripts/calibration``) and has no default.
            entropy_restricted: SWEET. Bias green tokens only at positions whose next-token Shannon
                entropy exceeds ``entropy_threshold``.
            topk_restricted: KGW-TK. Bias only green tokens among the top-k logits (0 disables).
            morphmark: MorphMark. The green list is kept; the bias is replaced by MorphMark's
                probability redistribution, configured by ``morphmark_config``.
            device: device the green-list mask lives on. The mask is a ``(|V|, |V|)`` boolean tensor
                (15 GiB for a 128k vocabulary, 61 GiB for 256k) and every teacher call runs on it.
                The permutation that defines the green lists is drawn by a generator on this device,
                so a fingerprint embedded with a CUDA-built mask must be detected with one.
        """
        selected = [n for n, on in (("tie_restricted", tie_restricted), ("entropy_restricted", entropy_restricted),
                                    ("topk_restricted", bool(topk_restricted)), ("morphmark", morphmark)) if on]
        if len(selected) > 1:
            raise ValueError(f"at most one of tie_restricted / entropy_restricted / topk_restricted / "
                             f"morphmark may be set; got {selected}")
        if tie_restricted and tau_gap is None:
            raise ValueError("tie_restricted=True requires tau_gap; the near-tie window is calibrated "
                             "per model and has no default")

        self.device = device
        self.tie_restricted = bool(tie_restricted)
        self.tau_gap = None if tau_gap is None else float(tau_gap)
        self.entropy_restricted = bool(entropy_restricted)
        self.entropy_threshold = float(entropy_threshold)
        self.topk_restricted = int(topk_restricted or 0)
        self.morphmark = bool(morphmark)
        self._morphmark = MorphMarkBias(**(morphmark_config or {})) if self.morphmark else None

        seeder = GreenlistSeeder(vocab, gamma, seeding_scheme, device)
        self.vocab_size = seeder.vocab_size
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.seeding_scheme = seeding_scheme
        self.k = seeder.context_width

        # greenlist_masks[h, v] is True when token v is green after a context whose hash is h; with
        # `simple_1` the hash of a one-token context is the token id itself.
        self.greenlist_masks = torch.zeros((self.k * self.vocab_size, self.vocab_size), dtype=torch.bool,
                                           device=device)
        for i in tqdm(range(self.greenlist_masks.shape[0]), desc="green lists", leave=False):
            ctx = torch.tensor([0] * (self.k - 1) + [i], dtype=torch.long, device=device)
            self.greenlist_masks[i, seeder.greenlist_ids(ctx)] = True

        self.special_ids = []
        if tokenizer is not None:
            self.special_ids = sorted({t for t in (tokenizer.eos_token_id, tokenizer.bos_token_id,
                                                   tokenizer.pad_token_id, tokenizer.unk_token_id)
                                       if t is not None})
        if self.special_ids and seeding_scheme == "simple_1":
            for t in self.special_ids:
                self.greenlist_masks[:, t] = False
                self.greenlist_masks[t, :] = False

    # ------------------------------------------------------------------ identity
    @property
    def gate_name(self) -> str:
        if self.morphmark:
            return "morphmark"
        if self.tie_restricted:
            return "neartie"
        if self.entropy_restricted:
            return "sweet"
        if self.topk_restricted:
            return "topk"
        return "kgw"

    # ------------------------------------------------------------------ teacher
    def _hashes(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Context hash of every position that has a full ``k``-token context: ``(batch, seq_len - k + 1)``."""
        return torch.sum(input_ids.unfold(-1, self.k, 1), dim=-1)

    def _boost_mask(self, logits_slice: torch.Tensor, green: torch.Tensor) -> torch.Tensor:
        """Which tokens receive the bias, ``(..., V)`` boolean, for the KGW / KGW-NT / KGW-TK / SWEET rules.

        ``logits_slice`` covers the model's full logit width; ``green`` the tokenizer's ``V`` tokens.
        """
        V = green.shape[-1]
        if self.tie_restricted:
            top1 = logits_slice.max(dim=-1, keepdim=True).values
            in_window = logits_slice[..., :V] > (top1 - self.tau_gap)
            return green & in_window
        if self.entropy_restricted:
            probs = torch.softmax(logits_slice.float(), dim=-1)
            entropy = -(torch.where(probs > 0, probs * probs.log(), probs.new_zeros(()))).sum(dim=-1)
            return green & (entropy > self.entropy_threshold).unsqueeze(-1)
        if self.topk_restricted:
            kth = logits_slice.topk(self.topk_restricted, dim=-1).values[..., -1:]
            return green & (logits_slice[..., :V] >= kth)
        return green

    def _green_full_width(self, hashes: torch.LongTensor, v_model: int) -> torch.Tensor:
        """Green-list mask widened to the model's logit width; padding columns beyond the tokenizer
        vocabulary are red. MorphMark renormalises over the whole logit width, so the mask must
        cover it (Qwen2.5 pads its embedding matrix beyond the tokenizer vocabulary)."""
        green = self.greenlist_masks[hashes]
        v_mask = green.shape[-1]
        if v_mask == v_model:
            return green
        if v_mask > v_model:
            raise ValueError(f"green-list width {v_mask} exceeds the model's logit width {v_model}; "
                             f"the watermark was built for a different tokenizer")
        pad = torch.zeros(*green.shape[:-1], v_model - v_mask, dtype=green.dtype, device=green.device)
        return torch.cat([green, pad], dim=-1)

    def watermark_logits_diff(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        input_device = input_ids.device
        input_ids = input_ids.to(self.device)
        logits = logits.to(self.device)
        hashes = self._hashes(input_ids)
        diff = torch.zeros_like(logits)
        sl = logits[..., self.k - 1:, :]
        if self.morphmark:
            green = self._green_full_width(hashes, logits.shape[-1])
            diff[..., self.k - 1:, :] = self._morphmark.apply(sl, green) - sl
        else:
            green = self.greenlist_masks[hashes]
            V = green.shape[-1]
            diff[..., self.k - 1:, :V][self._boost_mask(sl, green)] = self.delta
        return diff.to(input_device)

    def watermark_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        input_device = input_ids.device
        input_ids = input_ids.to(self.device)
        logits = logits.to(self.device).clone()
        hashes = self._hashes(input_ids)
        sl = logits[..., self.k - 1:, :]
        if self.morphmark:
            green = self._green_full_width(hashes, logits.shape[-1])
            logits[..., self.k - 1:, :] = self._morphmark.apply(sl, green)
        else:
            green = self.greenlist_masks[hashes]
            V = green.shape[-1]
            boost = self._boost_mask(sl, green)
            sl_v = logits[..., self.k - 1:, :V]
            sl_v += boost.to(sl_v.dtype) * self.delta
        return logits.to(input_device)
