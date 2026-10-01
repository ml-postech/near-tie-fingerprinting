# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).
#
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The KGW green list (Kirchenbauer et al., 2023).

At every position a pseudo-random ``gamma`` fraction of the vocabulary is green, seeded by the
preceding ``context_width`` tokens. The ``simple_1`` scheme seeds the permutation with the
previous token id multiplied by a fixed key, so the green list of a position is a function of one
token and the whole partition fits in a ``(|V|, |V|)`` boolean mask.
"""
from typing import Sized

import torch
from tqdm.auto import tqdm

SCHEMES = {"simple_1": dict(context_width=1, hash_key=15485863)}


class GreenList:
    """The green-list mask of a vocabulary, built once and shared by teachers and detectors.

    Args:
        vocab: the tokenizer vocabulary; only its size is used.
        gamma: green fraction.
        seeding_scheme: see ``SCHEMES``.
        tokenizer: when given, its special tokens (eos, bos, pad, unk) are never green and never
            seed a green list.
        device: device of the mask and of the generator that draws the permutations. A mask drawn
            on a CUDA generator differs from one drawn on CPU, so a fingerprint must be detected
            with a mask built on the same kind of device it was embedded with.
    """

    def __init__(self, vocab: Sized, gamma: float = 0.25, seeding_scheme: str = "simple_1", tokenizer=None,
                 device: str = "cuda"):
        if seeding_scheme not in SCHEMES:
            raise ValueError(f"unknown seeding scheme {seeding_scheme!r}; available: {sorted(SCHEMES)}")
        self.vocab_size = len(vocab)
        self.gamma = float(gamma)
        self.seeding_scheme = seeding_scheme
        self.k = SCHEMES[seeding_scheme]["context_width"]
        self.hash_key = SCHEMES[seeding_scheme]["hash_key"]
        self.device = device
        self.rng = torch.Generator(device=device)

        # masks[h, v]: token v is green after a context whose hash is h (the token id itself for k=1).
        self.masks = torch.zeros((self.k * self.vocab_size, self.vocab_size), dtype=torch.bool, device=device)
        for i in tqdm(range(self.masks.shape[0]), desc="green lists", leave=False):
            ctx = torch.tensor([0] * (self.k - 1) + [i], dtype=torch.long, device=device)
            self.masks[i, self.greenlist_ids(ctx)] = True

        self.special_ids = []
        if tokenizer is not None:
            self.special_ids = sorted({t for t in (tokenizer.eos_token_id, tokenizer.bos_token_id,
                                                   tokenizer.pad_token_id, tokenizer.unk_token_id)
                                       if t is not None})
        if self.special_ids and seeding_scheme == "simple_1":
            for t in self.special_ids:
                self.masks[:, t] = False
                self.masks[t, :] = False

    def greenlist_ids(self, context: torch.LongTensor) -> torch.LongTensor:
        """Green token ids after ``context`` (its last ``k`` tokens seed the permutation)."""
        if context.shape[-1] < self.k:
            raise ValueError(f"the seeding scheme needs a context of {self.k} token(s)")
        seed = self.hash_key * context[-self.k:].to(self.rng.device).sum().item()
        self.rng.manual_seed(seed % (2**64 - 1))
        permutation = torch.randperm(self.vocab_size, device=context.device, generator=self.rng)
        return permutation[:int(self.vocab_size * self.gamma)]

    def hashes(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Context hash of every position with a full ``k``-token context: ``(batch, seq_len - k + 1)``."""
        return torch.sum(input_ids.unfold(-1, self.k, 1), dim=-1)

    def __getitem__(self, hashes: torch.LongTensor) -> torch.BoolTensor:
        """Green masks ``(..., V)`` for context hashes ``(...)``."""
        return self.masks[hashes]

    def widen(self, green: torch.BoolTensor, width: int) -> torch.BoolTensor:
        """``green`` extended to a model logit width larger than the vocabulary, the padding columns red."""
        v = green.shape[-1]
        if v == width:
            return green
        if v > width:
            raise ValueError(f"green-list width {v} exceeds the model's logit width {width}; the green list "
                             f"was built for a different tokenizer")
        pad = torch.zeros(*green.shape[:-1], width - v, dtype=green.dtype, device=green.device)
        return torch.cat([green, pad], dim=-1)

    def is_green(self, hashes: torch.LongTensor, tokens: torch.LongTensor) -> torch.BoolTensor:
        """Whether ``tokens`` are green after the contexts with hashes ``hashes`` (same shape)."""
        return self.masks[hashes, tokens]
