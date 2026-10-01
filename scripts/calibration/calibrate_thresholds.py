"""Calibrate the near-tie window and the SWEET entropy threshold of a base model.

    python scripts/calibration/calibrate_thresholds.py --model meta-llama/Llama-3.1-8B-Instruct \
        --out results/calibration/llama3.1-8b.json

The base model reads the first ``--n_docs`` documents of the watermark corpus (LucieFr), each on
its own, truncated to ``--seq_len`` tokens, and records at every position the gap between its two
largest logits and the Shannon entropy of its next-token distribution. The q-th percentile of the
gap is the ``tau_gap`` of the ``q<q>`` cells (a window that makes q% of positions near-tie), and the
(100-q)-th percentile of the entropy is the ``entropy_threshold`` of the SWEET ``q<q>`` cells (a
threshold that makes q% of positions high-entropy). Values are rounded to three decimals, as in
the shipped configurations.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from robust_fp.finetuning.datasets import luciefr_stream  # noqa: E402

QS = (10, 30, 50, 70)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_docs", type=int, default=40)
ap.add_argument("--seq_len", type=int, default=512)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda").eval()
stream = luciefr_stream()

gaps, entropies, n_seqs = [], [], 0
with torch.no_grad():
    for row in stream:
        if n_seqs >= a.n_docs:
            break
        ids = tok(row["text"], return_tensors="pt", truncation=True, max_length=a.seq_len).input_ids.to("cuda")
        if ids.shape[1] < 20:
            continue
        logits = model(ids).logits[0].float()
        top2 = logits.topk(2, dim=-1).values
        p = torch.softmax(logits, dim=-1)
        gaps.append((top2[:, 0] - top2[:, 1]).cpu())
        entropies.append(-(p * torch.log(p.clamp_min(1e-12))).sum(-1).cpu())
        n_seqs += 1

G = torch.cat(gaps).numpy()
E = torch.cat(entropies).numpy()
out = {"model": a.model, "n_docs": n_seqs, "n_positions": int(G.size), "seq_len": a.seq_len}
for q in QS:
    out[f"q{q}"] = {"tau_gap": round(float(np.quantile(G, q / 100)), 3),
                    "entropy_threshold": round(float(np.quantile(E, 1 - q / 100)), 3)}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out, indent=1))
