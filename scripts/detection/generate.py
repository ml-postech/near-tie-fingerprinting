"""Generate responses from a (possibly quantised) fingerprinted model under one deployment condition.

Writes ``<out_prefix>_r0.jsonl .. _r{runs-1}.jsonl``, one independent generation per run, each line
``{"prompts": ..., "completions": ...}``. The runs are drawn as ``SamplingParams(n=runs)``: the
prompt is processed once and the sequences are sampled independently, without a fixed seed.

Sampling follows the detection protocol: temperature as given, top_p 0.9, top_k 50, at most 200
new tokens and at least 25, repetition penalty 1.1.

    python scripts/detection/generate.py --model <checkpoint> --prompts data/french_alpaca_20000.json \
        --n 1000 --runs 5 --temperature 0.7 --out_prefix results/detection/llama3.1-8b/kgwnt_q50_d4/comps/clean
    python scripts/detection/generate.py ... --quant fp8
    python scripts/detection/generate.py ... --system "You are a helpful assistant." [--system_merge]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from robust_fp.generation import build_engine, conversations  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True, help="checkpoint to generate from")
ap.add_argument("--prompts", required=True, help="JSON list of prompt strings")
ap.add_argument("--n", type=int, default=1000, help="prompts per run")
ap.add_argument("--runs", type=int, default=5, help="independent generations per prompt")
ap.add_argument("--out_prefix", required=True)
ap.add_argument("--quant", choices=sorted(("none", "fp8", "int4")), default="none")
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--top_k", type=int, default=50, help="sampling support size; 0 disables the cut")
ap.add_argument("--system", default=None, help="system instruction")
ap.add_argument("--system_merge", action="store_true",
                help="prepend the system instruction to the user turn (models without a system role)")
ap.add_argument("--max_model_len", type=int, default=None)
ap.add_argument("--gpu_memory_utilization", type=float, default=None)
ap.add_argument("--dtype", default=None)
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

paths = [f"{a.out_prefix}_r{j}.jsonl" for j in range(a.runs)]
if not a.overwrite and all(os.path.exists(p) and sum(1 for _ in open(p)) >= a.n for p in paths):
    print(f"[generate] all {a.runs} runs exist for {a.out_prefix}; skipping")
    sys.exit(0)

prompts = json.load(open(a.prompts))[:a.n]
if len(prompts) < a.n:
    sys.exit(f"prompt file has {len(prompts)} < --n {a.n}")

from vllm import SamplingParams  # noqa: E402

llm = build_engine(a.model, a.quant, a.max_model_len, a.gpu_memory_utilization, a.dtype)
params = SamplingParams(temperature=a.temperature, top_p=0.9, top_k=a.top_k, max_tokens=200,
                        repetition_penalty=1.1, min_tokens=25, n=a.runs)
outs = llm.chat(conversations(prompts, a.system, a.system_merge), params)

short = [i for i, o in enumerate(outs) if len(o.outputs) < a.runs]
if short:
    sys.exit(f"{len(short)} prompts returned fewer than {a.runs} sequences")

os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
for j, path in enumerate(paths):
    with open(path, "w") as fh:
        for p, o in zip(prompts, outs):
            fh.write(json.dumps({"prompts": p, "completions": o.outputs[j].text.strip()}, ensure_ascii=False) + "\n")
    print(f"[generate] run {j}: {len(prompts)} responses -> {path}")
