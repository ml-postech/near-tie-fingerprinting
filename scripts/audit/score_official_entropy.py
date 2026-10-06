# Copyright (c) 2026 the authors.
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""The quality metric reported by Gloaguen et al., reproduced as implemented.

The evaluation code of Gloaguen et al. (``src/robust_fp/quality/ppl.py`` in eth-sri/robust-llm-fingerprints) reports a
number labelled perplexity. What the function computes is the mean, over positions, of the Shannon
entropy of the oracle's next-token distribution on ``"{prompt} {completion}"``: the generated tokens
never enter as targets, the mean is taken over padded positions too, and only the last 250 tokens
of the batch-padded sequence are scored. This script reproduces that computation on the same
``comps_*.json`` files that ``scripts/utility/score_ppl.py`` scores with a conditional perplexity,
so the two instruments can be compared on identical text.

    python scripts/audit/score_official_entropy.py --dir results/utility/llama3.1-8b --cells base kgw_d4 kgwnt_q50_d4

Writes ``<dir>/official_entropy.json`` as ``{cell: {mean, n, values}}``.
"""
import argparse
import glob
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dir", required=True)
ap.add_argument("--oracle", default="Qwen/Qwen2.5-32B-Instruct")
ap.add_argument("--cells", nargs="*", default=None, help="cells to score (default: every comps_*.json)")
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(a.oracle, torch_dtype=torch.bfloat16).to(device).eval()
tok = AutoTokenizer.from_pretrained(a.oracle)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


@torch.no_grad()
def official_entropy(prompts, completions, bs):
    out = []
    for i in range(0, len(prompts), bs):
        texts = [f"{p} {c}" for p, c in zip(prompts[i:i + bs], completions[i:i + bs])]
        enc = tok(texts, add_special_tokens=True, padding=True, return_tensors="pt", return_attention_mask=True).to(device)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        if ids.shape[1] > 250:
            ids, attn = ids[:, -250:], attn[:, -250:]
        probs = model(ids, attention_mask=attn).logits.softmax(dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean(dim=-1)
        out += entropy.float().tolist()
    return out


files = sorted(glob.glob(os.path.join(a.dir, "comps_*.json")))
if a.cells:
    files = [f for f in files if os.path.basename(f)[len("comps_"):-len(".json")] in set(a.cells)]
out_path = a.out or os.path.join(a.dir, "official_entropy.json")
res = json.load(open(out_path)) if os.path.exists(out_path) else {}
for f in files:
    cell = os.path.basename(f)[len("comps_"):-len(".json")]
    if cell in res:
        print(f"skip {cell}")
        continue
    d = json.load(open(f))
    prompts, comps = d["prompts"], d["completions"]
    if a.n:
        prompts, comps = prompts[:a.n], comps[:a.n]
    values = official_entropy(prompts, comps, a.batch_size)
    res[cell] = {"mean": sum(values) / len(values), "n": len(values), "values": values}
    json.dump(res, open(out_path, "w"))
    print(f"{cell}: mean entropy {res[cell]['mean']:.4f} (n={len(values)})", flush=True)
print("wrote", out_path)
