"""The evaluation audit table: the prior work's metric against a conditional perplexity.

    python scripts/audit/report_audit.py --results results --models llama3.1-8b --cells base kgw_d4 kgwnt_q50_d4

Reads, per model,
    results/audit/<model>/      generations on the prior work's evaluation prompts (French-Alpaca
                                instructions, temperature 0.7, 200 tokens), scored with both
                                score_official_entropy.py and score_ppl.py
    results/utility/<model>/    the utility corpus (WritingPrompts-fr, 256 tokens), scored with score_ppl.py
    results/frenchbench/<model>/<cell>/   FrenchBench accuracies (optional column)
and prints every number as a ratio to the unwatermarked base of the same column.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark"))
from report_frenchbench import ACC_TASKS, load_cell as load_fb  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", default="results")
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--cells", nargs="+", default=["base", "kgw_d4", "kgwnt_q50_d4"])
ap.add_argument("--md_out", default=None)
a = ap.parse_args()


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def fmt(v, base):
    return "—" if v is None or base is None else f"{v:.3f} ({v / base:.2f}x)"


lines = ["| model | cell | prior metric, their prompts | PPL, their prompts | PPL, utility corpus | FrenchBench acc (9 tasks) |",
         "|---|---|---:|---:|---:|---:|"]
for model in a.models:
    audit = os.path.join(a.results, "audit", model)
    ent = load(os.path.join(audit, "official_entropy.json"))
    ppl_theirs = load(os.path.join(audit, "ppl_stats.json"))
    ppl_ours = load(os.path.join(a.results, "utility", model, "ppl_stats.json"))
    fb = {c: load_fb(os.path.join(a.results, "frenchbench", model, c)) for c in a.cells}
    base_e = (ent.get("base") or {}).get("mean")
    base_t = (ppl_theirs.get("base") or {}).get("mean")
    base_o = (ppl_ours.get("base") or {}).get("mean")
    for cell in a.cells:
        e = (ent.get(cell) or {}).get("mean")
        t = (ppl_theirs.get(cell) or {}).get("mean")
        o = (ppl_ours.get(cell) or {}).get("mean")
        acc = fb[cell]
        acc_s = f"{sum(acc.values()) / len(ACC_TASKS):.4f}" if len(acc) == len(ACC_TASKS) else "—"
        lines.append(f"| {model} | {cell} | {fmt(e, base_e)} | {fmt(t, base_t)} | {fmt(o, base_o)} | {acc_s} |")
text = "\n".join(lines) + "\n"
print(text)
if a.md_out:
    os.makedirs(os.path.dirname(a.md_out) or ".", exist_ok=True)
    open(a.md_out, "w").write(text)
    print("wrote", a.md_out)
