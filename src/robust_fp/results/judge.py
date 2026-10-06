# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""LLM-judge scores, ``judge/<model>/judge_<cell>.json`` as written by ``scripts/utility/judge.py``."""
import json
import math
import os
import statistics
from typing import Optional, Tuple


class JudgeRun:
    def __init__(self, results: str, model: str):
        self.dir = os.path.join(results, "judge", model)

    def _load(self, cell: str) -> Optional[dict]:
        p = os.path.join(self.dir, f"judge_{cell}.json")
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            return json.load(fh)

    @property
    def has_base(self) -> bool:
        return self._load("base") is not None

    def mean(self, cell: str) -> Optional[float]:
        d = self._load(cell)
        return None if d is None else d.get("mean")

    def scores(self, cell: str) -> Optional[list]:
        d = self._load(cell)
        return None if d is None else d.get("scores")

    def drop(self, cell: str, min_pairs: int = 5) -> Optional[Tuple[float, float]]:
        """``(mean drop, 95% half-width)`` of ``base score - cell score``, paired over the prompts.
        Positive means quality lost relative to the unwatermarked base."""
        base, cell_scores = self.scores("base"), self.scores(cell)
        if base is None or cell_scores is None:
            return None
        d = [b - c for b, c in zip(base, cell_scores) if b is not None and c is not None]
        if len(d) < min_pairs:
            return None
        return statistics.mean(d), 1.96 * statistics.stdev(d) / math.sqrt(len(d))
