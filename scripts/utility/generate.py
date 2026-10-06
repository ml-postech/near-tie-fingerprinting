# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Generate the utility corpus: one open-ended response per prompt, with vLLM.

    python scripts/utility/generate.py --model <checkpoint> --cell kgwnt_q50_d4 \
        --prompts data/writingprompts_fr_500.json --out_dir results/utility/llama3.1-8b

Default sampling is the utility protocol of the paper: temperature 1.0, top_p 0.9, top_k 50,
repetition penalty 1.1, 25 to 256 new tokens, EOS allowed (the prompts ask for a 250-300 word
text). Writes ``<out_dir>/comps_<cell>.json`` as ``{"prompts": [...], "completions": [...]}``,
the input of ``score_ppl.py``, ``judge.py`` and ``scripts/audit/score_official_entropy.py``.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from robust_fp.generation import build_engine, conversations  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--cell", required=True, help="label of the output file (``base`` for the unwatermarked model)")
ap.add_argument("--prompts", default="data/writingprompts_fr_500.json")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--max_new_tokens", type=int, default=256)
ap.add_argument("--min_new_tokens", type=int, default=25)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top_p", type=float, default=0.9)
ap.add_argument("--top_k", type=int, default=50)
ap.add_argument("--rep_penalty", type=float, default=1.1)
ap.add_argument("--system", default=None)
ap.add_argument("--system_merge", action="store_true")
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

out = os.path.join(a.out_dir, f"comps_{a.cell}.json")
if os.path.exists(out) and not a.overwrite:
    print(f"[generate] {out} exists; skipping")
    sys.exit(0)
raw = json.load(open(a.prompts))
prompts = [(x.get("prompt") or x.get("text")) if isinstance(x, dict) else x for x in raw][:a.n]
if len(prompts) < a.n:
    sys.exit(f"prompt file has {len(prompts)} < --n {a.n}")

from vllm import SamplingParams  # noqa: E402

llm = build_engine(a.model)
params = SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, max_tokens=a.max_new_tokens,
                        min_tokens=a.min_new_tokens, repetition_penalty=a.rep_penalty)
outs = llm.chat(conversations(prompts, a.system, a.system_merge), params)
comps = [o.outputs[0].text.strip() for o in outs]
os.makedirs(a.out_dir, exist_ok=True)
json.dump({"prompts": prompts, "completions": comps}, open(out, "w"), ensure_ascii=False)
print(f"[generate] {a.cell}: {len(comps)} responses, {sum(1 for c in comps if not c)} empty -> {out}")
