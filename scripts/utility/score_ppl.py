# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Perplexity of generated responses under an external oracle model.

    python scripts/utility/score_ppl.py --dir results/utility/llama3.1-8b

Scores every ``comps_*.json`` in ``--dir``: the oracle (Qwen2.5-32B-Instruct) reads the prompt as
context and the response tokens are scored, so a cell's number measures its own text under a
model that neither embedded nor detects the fingerprint. Writes

    ppl_stats.json   {cell: {median, mean, corpus, n}}    corpus = exp(total NLL / total tokens)
    ppl_raw.json     {cell: [{ppl, nll, ntok}, ...]}      per response, for paired bootstraps
"""
import argparse
import glob
import json
import math
import os
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dir", required=True)
ap.add_argument("--oracle", default="Qwen/Qwen2.5-32B-Instruct")
ap.add_argument("--max_len", type=int, default=1024)
ap.add_argument("--n", type=int, default=0, help="score only the first n responses of every cell (0 = all)")
ap.add_argument("--shard", action="store_true", help="spread the oracle over every visible GPU")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.oracle)
model = (AutoModelForCausalLM.from_pretrained(a.oracle, torch_dtype=torch.bfloat16, device_map="auto")
         if a.shard else AutoModelForCausalLM.from_pretrained(a.oracle, torch_dtype=torch.bfloat16).to("cuda")).eval()


@torch.no_grad()
def score(prompt, completion):
    """(ppl, summed NLL, scored tokens) of the completion given the prompt, or None."""
    if not completion or not completion.strip():
        return None
    pre = tok(prompt, add_special_tokens=True).input_ids
    cont = tok(" " + completion, add_special_tokens=False).input_ids
    full = (pre + cont)[:a.max_len]
    if len(full) <= len(pre) + 1:
        return None
    ids = torch.tensor([full], device=model.device)
    labels = ids.clone()
    labels[0, :len(pre)] = -100
    nll = model(ids, labels=labels).loss.item()
    ntok = len(full) - len(pre)
    ppl = math.exp(nll)
    if not (math.isfinite(ppl) and ntok > 0):
        return None
    return ppl, nll * ntok, ntok


stats, raw = {}, {}
for f in sorted(glob.glob(os.path.join(a.dir, "comps_*.json"))):
    d = json.load(open(f))
    prompts, comps = d["prompts"], d["completions"]
    if a.n:
        prompts, comps = prompts[:a.n], comps[:a.n]
    rs = [r for p, c in zip(prompts, comps) if (r := score(p, c)) is not None]
    if not rs:
        continue
    cell = os.path.basename(f)[len("comps_"):-len(".json")]
    ppls = [r[0] for r in rs]
    nll, ntok = sum(r[1] for r in rs), sum(r[2] for r in rs)
    stats[cell] = dict(median=round(statistics.median(ppls), 3), mean=round(statistics.mean(ppls), 3),
                       corpus=round(math.exp(nll / ntok), 3), n=len(ppls))
    raw[cell] = [dict(ppl=round(r[0], 4), nll=round(r[1], 4), ntok=r[2]) for r in rs]
    print(f"{cell}: mean={stats[cell]['mean']} median={stats[cell]['median']} corpus={stats[cell]['corpus']} n={len(ppls)}",
          flush=True)
for name, obj in (("ppl_stats.json", stats), ("ppl_raw.json", raw)):
    json.dump(obj, open(os.path.join(a.dir, name), "w"), indent=1)
    print("wrote", os.path.join(a.dir, name))
