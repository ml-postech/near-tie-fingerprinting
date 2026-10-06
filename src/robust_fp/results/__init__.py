# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Readers for the result trees written by the evaluation scripts.

    results/
      detection/<model>/<cell>/detect.jsonl       one row per (arm, run) from scripts/detection/detect.py
      query_budget/<model>/curves*.jsonl           z at every budget, from scripts/detection/query_budget.py
      query_budget/qmin.json                       {model: {cell: N_min}}, from reduce_query_budget.py
      utility/<model>/ppl_stats.json, ppl_raw.json from scripts/utility/score_ppl.py
      judge/<model>/judge_<cell>.json              from scripts/utility/judge.py
"""
from .robustness import (ARMS, GATE_Z, cell_arms, load_qmin, qmin, query_budget_curves, worst_z,
                         worst_z_at)
from .cells import DESIGNED_GRID, FAMILIES, MORPHMARK_EXTRA, family_of, is_designed
from .judge import JudgeRun
from .utility import UtilityTable

__all__ = ["ARMS", "GATE_Z", "cell_arms", "load_qmin", "qmin", "query_budget_curves", "worst_z",
           "worst_z_at", "DESIGNED_GRID", "MORPHMARK_EXTRA", "FAMILIES", "family_of", "is_designed", "JudgeRun",
           "UtilityTable"]
