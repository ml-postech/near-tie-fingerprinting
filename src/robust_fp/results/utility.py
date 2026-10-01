# Copyright (c) 2026 the authors.
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""The utility axis: perplexity of the fingerprinted model's generations under an external oracle."""
import json
import math
import os
import random
import statistics
from typing import Optional, Tuple


class UtilityTable:
    """``utility/<model>/ppl_stats.json`` and ``ppl_raw.json`` as written by ``scripts/utility/score_ppl.py``.

    Every cell was generated from the same prompts as the unwatermarked base, so confidence
    intervals on a cell's cost are computed by resampling prompt indices jointly for the cell and
    the base (a paired bootstrap).
    """

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

    def comps_path(self, cell: str) -> str:
        return os.path.join(self.dir, f"comps_{cell}.json")

    def value(self, cell: str, stat: str = "mean") -> Optional[float]:
        v = self.stats.get(cell)
        return None if v is None else v[stat]

    def base(self, stat: str = "mean") -> float:
        return self.stats["base"][stat]

    def ratio(self, cell: str, stat: str = "mean") -> Optional[float]:
        v = self.value(cell, stat)
        return None if v is None else v / self.base(stat)

    def paired_ci(self, cell: str, stat: str = "mean", axis: str = "ratio", boot: int = 1000,
                  seed: int = 0) -> Optional[Tuple[float, float]]:
        """95% bootstrap interval (lo, hi) on the cell's cost relative to the base.

        ``axis``: ``ratio`` for PPL_cell / PPL_base, ``log`` for its logarithm.
        """
        if not self.has(cell):
            return None
        cell_ppl = [r["ppl"] for r in self.raw[cell]]
        base_ppl = [r["ppl"] for r in self.raw["base"]]
        agg = statistics.mean if stat == "mean" else statistics.median
        n = min(len(cell_ppl), len(base_ppl))
        rng = random.Random(seed)
        draws = []
        for _ in range(boot):
            idx = [rng.randrange(n) for _ in range(n)]
            r = agg([cell_ppl[i] for i in idx]) / agg([base_ppl[i] for i in idx])
            draws.append(math.log(r) if axis == "log" else r)
        draws.sort()
        return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]
