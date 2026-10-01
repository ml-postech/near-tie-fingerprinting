"""LLM-judge scoring of generated French text on a 1-10 linguistic-quality rubric.

    OPENAI_API_KEY=... python scripts/utility/judge.py --files results/utility/llama3.1-8b/comps_*.json \
        --out_dir results/judge/llama3.1-8b --n 500

The judge (``gpt-5``, minimal reasoning) sees only the generated text, never the prompt, so
instruction following cannot leak into the score. The rubric is ``rubric_fr_10pt.txt`` next to
this script. Writes ``<out_dir>/judge_<cell>.json`` with the per-response scores and reasons; a
cell whose file already holds ``n`` scores is skipped.
"""
import argparse
import json
import os
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--files", nargs="+", required=True, help="comps_<cell>.json files")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--model", default="gpt-5")
ap.add_argument("--reasoning_effort", default="minimal", choices=["minimal", "low", "medium", "high"])
ap.add_argument("--max_tokens", type=int, default=2000)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--rubric", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rubric_fr_10pt.txt"))
a = ap.parse_args()

TEMPLATE = open(a.rubric).read()
client = OpenAI()
os.makedirs(a.out_dir, exist_ok=True)


def score_one(text):
    kw = dict(model=a.model, messages=[{"role": "user", "content": TEMPLATE.format(response=text or "(empty)")}],
              response_format={"type": "json_object"}, max_completion_tokens=a.max_tokens)
    if a.model.startswith(("gpt-5", "o")):
        kw["reasoning_effort"] = a.reasoning_effort
    r = client.chat.completions.create(**kw)
    m = re.search(r"\{.*\}", r.choices[0].message.content, re.S)
    j = json.loads(m.group(0)) if m else None
    if not j or "score" not in j:
        raise ValueError("no score in the judge's answer")
    return float(j["score"]), j.get("reason", "")


for f in a.files:
    cell = os.path.basename(f)[len("comps_"):-len(".json")]
    out = os.path.join(a.out_dir, f"judge_{cell}.json")
    if os.path.exists(out):
        prev = json.load(open(out))
        if prev.get("n") == a.n and all(s is not None for s in prev.get("scores", [])):
            print(f"[judge] {cell} already scored; skipping")
            continue
    comps = json.load(open(f))["completions"][:a.n]
    if len(comps) < a.n:
        raise SystemExit(f"{f} holds {len(comps)} responses < --n {a.n}")
    res = [None] * len(comps)

    def task(i):
        err = None
        for _ in range(4):
            try:
                return i, score_one(comps[i]), None
            except Exception as e:
                err = e
        return i, (None, ""), err

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(task, i) for i in range(len(comps))]):
            i, r, err = fut.result()
            res[i] = r
            if err is not None:
                print(f"[judge] {cell} #{i}: {err}")
    scores = [r[0] for r in res]
    valid = [s for s in scores if s is not None]
    rec = dict(label=cell, rubric=os.path.basename(a.rubric), model=a.model, reasoning_effort=a.reasoning_effort,
               n=len(valid), mean=round(statistics.mean(valid), 3) if valid else None,
               median=statistics.median(valid) if valid else None,
               sd=round(statistics.pstdev(valid), 3) if len(valid) > 1 else None,
               scores=scores, reasons=[r[1] for r in res])
    json.dump(rec, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"[judge] {cell}: n={rec['n']} mean={rec['mean']} median={rec['median']} sd={rec['sd']} -> {out}")
