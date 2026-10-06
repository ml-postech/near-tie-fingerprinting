# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Re-score the protocol's responses at every query budget N on a grid.

    python scripts/detection/query_budget.py --model llama3.1-8b --configs configs/llama3.1-8b \
        --results results [--cells kgwnt_q50_d4 ...] [--shard 0 --nshard 8]

For every (cell, arm, run) the responses of ``results/detection/<model>/<cell>/comps/<arm>_r<j>.jsonl``
are accumulated in order and the pooled z is reported after 100, 200, ..., 1000 responses. Rows
``{cell, arm, run, N, teacher, z_key, gamma, z, green, T_dedup}`` go to
``results/query_budget/<model>/curves_s<shard>.jsonl``. Cells without a configuration file (the
unwatermarked base) are skipped.
The base model is loaded once per invocation and only when a detector that restricts the scored positions needs it; the green
list is rebuilt only when gamma or the seeding scheme changes between cells. Every cell is read
with its own detector, as in ``detect.py``.
"""
import argparse
import glob
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from robust_fp.config import MainConfiguration  # noqa: E402
from robust_fp.detection import Detector, PooledCounts  # noqa: E402
from robust_fp.results.robustness import ARMS, RUNS  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True, help="short model name (a results/detection/ subdirectory)")
ap.add_argument("--configs", required=True, help="directory holding <cell>.yaml for every cell")
ap.add_argument("--results", default="results")
ap.add_argument("--cells", nargs="*", default=None, help="cells to score (default: every scored cell)")
ap.add_argument("--arms", nargs="*", default=ARMS)
ap.add_argument("--runs", type=int, default=RUNS)
ap.add_argument("--budgets", default="100,200,300,400,500,600,700,800,900,1000")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--out", default=None, help="default results/query_budget/<model>/curves_s<shard>.jsonl")
a = ap.parse_args()

NS = sorted(int(x) for x in a.budgets.split(","))
NMAX = max(NS)
det_root = os.path.join(a.results, "detection", a.model)
cells = a.cells or sorted(os.path.basename(p) for p in glob.glob(os.path.join(det_root, "*"))
                          if os.path.exists(os.path.join(p, "detect.jsonl")))
cells = [c for c in cells if os.path.exists(os.path.join(a.configs, c + ".yaml"))]
cells = [c for i, c in enumerate(cells) if i % a.nshard == a.shard]
out = a.out or os.path.join(a.results, "query_budget", a.model, f"curves_s{a.shard}.jsonl")
os.makedirs(os.path.dirname(out), exist_ok=True)
if not cells:
    print("[query_budget] no cells for this shard")
    sys.exit(0)

configs = {c: MainConfiguration.from_yaml(os.path.join(a.configs, c + ".yaml")) for c in cells}
bases = {cfg.base_model for cfg in configs.values()}
if len(bases) != 1:
    sys.exit(f"cells name different base models: {bases}")
BASE = bases.pop()
# Sort by green-list key so the mask is rebuilt as rarely as possible.
cells.sort(key=lambda c: (configs[c].watermark.config.gamma, configs[c].watermark.config.seeding_scheme, c))

done = set()
if os.path.exists(out):
    for line in open(out):
        if line.strip():
            r = json.loads(line)
            done.add((r["cell"], r["arm"], r["run"], r["N"]))

tok = AutoTokenizer.from_pretrained(BASE)
base = None                            # loaded on first use, only if a cell restricts the scored positions

cache = {"key": None, "greenlist": None}
det = wm = None
for cell in cells:
    wcfg = configs[cell].watermark.config
    key = (wcfg.gamma, wcfg.seeding_scheme)
    if cache["key"] != key:
        det = wm = None                 # drop every reference to the previous mask before building the next one
        cache["greenlist"] = None
        torch.cuda.empty_cache()
        cache.update(key=key, greenlist=wcfg.build_greenlist(tok, device="cuda"))
    # The green list depends only on (gamma, seeding scheme); the teacher, and so the detector, is the cell's own.
    wm = wcfg.build(tok, device="cuda", greenlist=cache["greenlist"])
    if wm.restricts_positions and base is None:
        base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda").eval()
    det = Detector(wm, tok, base)
    print(f"=== {cell} teacher={wm.name} gamma={det.gamma}", flush=True)
    for arm in a.arms:
        for run in range(a.runs):
            if all((cell, arm, run, n) in done for n in NS):
                continue
            src = os.path.join(det_root, cell, "comps", f"{arm}_r{run}.jsonl")
            if not os.path.exists(src):
                sys.exit(f"missing responses {src}")
            rows = [json.loads(l) for l in open(src) if l.strip()][:NMAX]
            if len(rows) < NMAX:
                sys.exit(f"{src} holds {len(rows)} responses < {NMAX}")
            counts = PooledCounts(det.gamma)
            emitted = []
            for i, r in enumerate(rows, 1):
                counts.add(det.response_flags(r["prompts"], r.get("completions") or ""))
                if i in NS:
                    G, T = counts.counts(det.z_key)
                    emitted.append(dict(cell=cell, arm=arm, run=run, N=i, teacher=wm.name, z_key=det.z_key, gamma=det.gamma,
                                        z=round(counts.z(det.z_key), 3) if T else None,
                                        green=round(G / T, 4) if T else None, T_dedup=T))
            with open(out, "a") as fh:
                for e in emitted:
                    fh.write(json.dumps(e) + "\n")
            print(f"  {arm} r{run}: " + " ".join(f"N{e['N']}={e['z']}" for e in emitted), flush=True)
print(f"[query_budget] done -> {out}")
