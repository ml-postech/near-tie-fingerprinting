# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Reduce the query-budget curves to N_min per cell.

    python scripts/detection/reduce_query_budget.py --results results --models llama3.1-8b qwen2.5-3b

N_min is the smallest budget on the grid at which the worst-case z (mean over runs, min over arms)
clears the gate and stays above it at every larger budget. A cell that is not scored for every arm,
run and budget is refused rather than reduced over what is present. Writes
``results/query_budget/qmin.json`` as ``{model: {cell: N_min}}``, merging with the models already
in the file, and records the gate and arm set beside it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from robust_fp.results.robustness import ARMS, GATE_Z, RUNS, qmin, query_budget_curves  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", default="results")
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--gate", type=float, default=GATE_Z)
ap.add_argument("--runs", type=int, default=RUNS)
ap.add_argument("--arms", nargs="*", default=ARMS)
ap.add_argument("--allow_partial", action="store_true", help="drop incomplete cells instead of refusing")
a = ap.parse_args()

qmin_path = os.path.join(a.results, "query_budget", "qmin.json")
merged = {}
if os.path.exists(qmin_path):
    merged = json.load(open(qmin_path))

for model in a.models:
    curves = query_budget_curves(a.results, model)
    cells = sorted({c for (c, _, _) in curves})
    budgets = sorted({n for (_, _, n) in curves})
    incomplete = []
    for cell in cells:
        missing = [(arm, n) for arm in a.arms for n in budgets
                   if len([z for z in curves.get((cell, arm, n), []) if isinstance(z, (int, float))]) != a.runs]
        if missing:
            incomplete.append((cell, missing[:3]))
    if incomplete and not a.allow_partial:
        print(f"REFUSING {model}: {len(incomplete)} cells are not fully scored, e.g. {incomplete[:3]}", file=sys.stderr)
        sys.exit(2)
    bad = {c for c, _ in incomplete}
    passed = {}
    for cell in cells:
        if cell in bad:
            continue
        q = qmin(curves, cell, gate=a.gate, arms=a.arms, runs=a.runs)
        if q is not None:
            passed[cell] = q
    merged[model] = passed
    print(f"{model}: {len(cells)} cells, {len(passed)} clear the gate within the grid, {len(bad)} dropped")

os.makedirs(os.path.dirname(qmin_path), exist_ok=True)
json.dump(merged, open(qmin_path, "w"), indent=1, sort_keys=True)
json.dump(dict(gate=a.gate, runs=a.runs, arms=a.arms,
               rule="z at a budget is the mean over runs then the min over arms; N_min is the smallest "
                    "budget at which z >= gate holds there and at every larger budget"),
          open(qmin_path.replace(".json", ".params.json"), "w"), indent=1)
print(f"wrote {qmin_path}")
