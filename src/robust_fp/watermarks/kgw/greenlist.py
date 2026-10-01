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

"""Green-list seeding of the KGW watermark (Kirchenbauer et al., 2023).

The green list of a position is a pseudo-random subset of the vocabulary of size ``gamma * |V|``.
This is the ``simple_1`` scheme of the reference implementation: the random permutation of the
vocabulary is seeded by the previous token id multiplied by a fixed key.
"""
from typing import Sized

import torch

SCHEMES = {"simple_1": dict(context_width=1, hash_key=15485863)}


class GreenlistSeeder:
    def __init__(self, vocab: Sized, gamma: float, seeding_scheme: str, device: str):
        if seeding_scheme not in SCHEMES:
            raise ValueError(f"unknown seeding scheme {seeding_scheme!r}; available: {sorted(SCHEMES)}")
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.seeding_scheme = seeding_scheme
        self.context_width = SCHEMES[seeding_scheme]["context_width"]
        self.hash_key = SCHEMES[seeding_scheme]["hash_key"]
        self.rng = torch.Generator(device=device)

    def greenlist_ids(self, context: torch.LongTensor) -> torch.LongTensor:
        """Green-list token ids after ``context`` (its last ``context_width`` tokens seed the list)."""
        if context.shape[-1] < self.context_width:
            raise ValueError(f"the seeding scheme needs a context of {self.context_width} token(s)")
        seed = self.hash_key * context[-self.context_width:].to(self.rng.device).sum().item()
        self.rng.manual_seed(seed % (2**64 - 1))
        permutation = torch.randperm(self.vocab_size, device=context.device, generator=self.rng)
        return permutation[:int(self.vocab_size * self.gamma)]
