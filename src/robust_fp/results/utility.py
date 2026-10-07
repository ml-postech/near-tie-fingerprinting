# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""The utility axis: perplexity of the fingerprinted model's generations under an external oracle."""
import json
import os
from typing import Optional


class UtilityTable:
    """``utility/<model>/ppl_stats.json`` and ``ppl_raw.json`` as written by ``scripts/utility/score_ppl.py``."""

    def __init__(self, results: str, model: str):
        self.dir = os.path.join(results, "utility", model)
        self.stats = self._load("ppl_stats.json")
        self.raw = self._load("ppl_raw.json")

    def _load(self, name):
        p = os.path.join(self.dir, name)
        if not os.path.exists(p):
            return {}
        with open(p) as fh:
            return json.load(fh)

    @property
    def exists(self) -> bool:
        return bool(self.stats) and "base" in self.stats

    def has(self, cell: str) -> bool:
        return cell in self.stats and cell in self.raw

    def value(self, cell: str, stat: str = "mean") -> Optional[float]:
        v = self.stats.get(cell)
        return None if v is None else v[stat]

    def base(self, stat: str = "mean") -> float:
        return self.stats["base"][stat]

    def ratio(self, cell: str, stat: str = "mean") -> Optional[float]:
        v = self.value(cell, stat)
        return None if v is None else v / self.base(stat)
