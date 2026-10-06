# Copyright (c) 2026 the authors.
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Score generated responses with the cell's own detector and append one row to ``detect.jsonl``.

    python scripts/detection/detect.py --config configs/llama3.1-8b/kgwnt_q50_d4.yaml \
        --comps results/detection/llama3.1-8b/kgwnt_q50_d4/comps/clean_r0.jsonl --n 1000 \
        --cell kgwnt_q50_d4 --arm clean --run 0 --out results/detection/llama3.1-8b/kgwnt_q50_d4/detect.jsonl

The detector is built from the configuration (green list and the teacher's eligibility rule) and the
base model it names; the fingerprinted model is never loaded. See :mod:`robust_fp.detection` for the statistic.
System instructions used at generation time are not part of the scored text: the detector sees
only the prompt and the response.
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from robust_fp.config import MainConfiguration  # noqa: E402
from robust_fp.detection import Detector, PooledCounts  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--config", required=True, help="the cell's configuration")
ap.add_argument("--comps", required=True, help="responses, one {prompts, completions} JSON object per line")
ap.add_argument("--n", type=int, default=1000, help="responses scored (the query budget)")
ap.add_argument("--cell", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--run", type=int, required=True)
ap.add_argument("--out", required=True, help="detect.jsonl to append to")
a = ap.parse_args()

cfg = MainConfiguration.from_yaml(a.config)
wcfg = cfg.watermark.config
if os.path.exists(a.out):
    with open(a.out) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if (r["cell"], r["arm"], r["run"]) == (a.cell, a.arm, a.run):
                    print(f"[detect] {a.cell} {a.arm} r{a.run} already scored; skipping")
                    sys.exit(0)

rows = [json.loads(l) for l in open(a.comps) if l.strip()][:a.n]
if len(rows) < a.n:
    sys.exit(f"{a.comps} has {len(rows)} responses < --n {a.n}")

tok = AutoTokenizer.from_pretrained(cfg.base_model)
watermark = wcfg.build(tok, device="cuda")
base = None
if watermark.restricts_positions:
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16).to("cuda").eval()
det = Detector(watermark, tok, base)

counts = PooledCounts(det.gamma)
for r in rows:
    counts.add(det.response_flags(r["prompts"], r.get("completions") or ""))

res = dict(label=f"{a.cell}_{a.arm}_r{a.run}", cell=a.cell, arm=a.arm, run=a.run, n=len(rows),
           teacher=watermark.name, z_key=det.z_key, tau_gap=wcfg.tau_gap if (wcfg.tie_restricted or wcfg.neartie_wrap) else None,
           entropy_threshold=wcfg.entropy_threshold if wcfg.entropy_restricted else None, gamma=det.gamma,
           config=a.config, **counts.summary())
print(f"[detect] {res['label']}: teacher={res['teacher']} {res['z_key']}={res[res['z_key']]} (elig_frac={res['elig_frac']}) "
      f"all_z={res['all_z']} n_scored={res['n_scored']}")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "a") as fh:
    fh.write(json.dumps(res) + "\n")
