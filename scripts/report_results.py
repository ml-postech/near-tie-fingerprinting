# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""One table per model: detection and text quality of every cell in the result tree.

    python scripts/report_results.py --results results --models llama3.1-8b [--cells kgw_d4 kgwnt_q50_d4] [--md_out table.md]

Columns:
    worst-case z   minimum over the deployment conditions of the mean z over runs (detected when it clears the gate z = 4)
    binding        the deployment condition that attains the minimum
    PPL ratio      oracle perplexity of the cell's generations over the unwatermarked base model's
    judge drop     LLM-judge score of the base minus the cell's, with a 95% half-width
    N_min          smallest number of responses at which the worst case clears the gate and stays above it
A column is blank when its results are not in the tree.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from robust_fp.results import GATE_Z, JudgeRun, UtilityTable, cell_arms, family_of, load_qmin  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", default="results")
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--cells", nargs="*", default=None, help="default: every cell with a detection or utility result")
ap.add_argument("--md_out", default=None)
a = ap.parse_args()


def cells_of(model):
    found = {os.path.basename(os.path.dirname(p))
             for p in glob.glob(os.path.join(a.results, "detection", model, "*", "detect.jsonl"))}
    found |= set(UtilityTable(a.results, model).stats)
    found.discard("base")
    return sorted(found, key=lambda c: (family_of(c)[0], c))


def fmt(v, spec):
    return "" if v is None else format(v, spec)


qmin_path = os.path.join(a.results, "query_budget", "qmin.json")
qmin = load_qmin(a.results) if os.path.exists(qmin_path) else {}
lines = []
for model in a.models:
    util, judge = UtilityTable(a.results, model), JudgeRun(a.results, model)
    lines += [f"### {model}", "",
              "| cell | worst-case z | binding | PPL ratio | judge drop | N_min |", "|:--|--:|:--|--:|--:|--:|"]
    if util.exists:
        base_judge = judge.mean("base")
        lines.append(f"| base | | | PPL {util.base():.2f} | "
                     f"{'' if base_judge is None else f'score {base_judge:.2f}'} | |")
    for cell in a.cells or cells_of(model):
        per = cell_arms(a.results, model, cell)
        z = binding = None
        if per:
            binding = min(per, key=per.get)
            z = per[binding]
        ratio = util.ratio(cell) if util.exists else None
        drop = judge.drop(cell)
        n_min = qmin.get(model, {}).get(cell)
        flag = "" if z is None else (" ✓" if z >= GATE_Z else "")
        lines.append(f"| {cell} | {fmt(z, '.2f')}{flag} | {binding or ''} | {fmt(ratio, '.3f')} | "
                     f"{'' if drop is None else f'{drop[0]:.2f} ± {drop[1]:.2f}'} | {fmt(n_min, 'd')} |")
    lines.append("")

out = "\n".join(lines)
print(out)
if a.md_out:
    os.makedirs(os.path.dirname(a.md_out) or ".", exist_ok=True)
    with open(a.md_out, "w") as fh:
        fh.write(out + "\n")
