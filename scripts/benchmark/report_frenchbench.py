"""FrenchBench accuracy per cell, and its distance from the unwatermarked base.

    python scripts/benchmark/report_frenchbench.py --results results --models llama3.1-8b [--cells base kgw_d4 ...]

Reads ``results/frenchbench/<model>/<cell>/**/results_*.json`` written by ``frenchbench.sh`` and
reports the mean accuracy over the nine accuracy-scored subtasks, with a 95% band on the difference
to the base computed from lm-eval's per-subtask standard errors (unpaired, hence conservative).
A cell missing any of the nine subtasks gets no mean.
"""
import argparse
import glob
import json
import math
import os

ACC_TASKS = ["arc_challenge", "boolqa", "fquadv2_bool", "grammar", "hellaswag",
             "reading_comp", "topic_based_nli", "vocab", "xnli"]


def load_cell(path):
    """``{task: acc}`` merged over the results files under ``path``."""
    return {t: v[0] for t, v in load_cell_se(path).items()}


def load_cell_se(path):
    """``{task: (acc, stderr)}``."""
    r = {}
    for f in sorted(glob.glob(os.path.join(path, "**", "results_*.json"), recursive=True)):
        try:
            r.update(json.load(open(f)).get("results", {}))
        except (json.JSONDecodeError, OSError):
            continue
    out = {}
    for t in ACC_TASKS:
        v = r.get(f"french_bench_{t}")
        if isinstance(v, dict) and "acc,none" in v:
            se = v.get("acc_stderr,none")
            out[t] = (v["acc,none"], se if isinstance(se, (int, float)) else float("nan"))
    return out


def mean_and_se(d):
    if len(d) != len(ACC_TASKS):
        return None, None
    return (sum(v for v, _ in d.values()) / len(ACC_TASKS),
            math.sqrt(sum(se ** 2 for _, se in d.values())) / len(ACC_TASKS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--cells", nargs="*", default=None, help="default: every cell directory")
    ap.add_argument("--per_task", action="store_true")
    ap.add_argument("--md_out", default=None)
    a = ap.parse_args()

    md = ["| model | cell | mean acc (9 tasks) | Δ vs base (points) | 95% band |", "|---|---|---:|---:|---:|"]
    for model in a.models:
        root = os.path.join(a.results, "frenchbench", model)
        cells = a.cells or sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p))
        data = {c: load_cell_se(os.path.join(root, c)) for c in cells}
        base_m, base_se = mean_and_se(data.get("base", {}))
        for c in cells:
            m, se = mean_and_se(data[c])
            if m is None:
                md.append(f"| {model} | {c} | ({len(data[c])}/9 tasks) | — | — |")
                continue
            delta = band = "—"
            if base_m is not None and c != "base":
                delta = f"{(m - base_m) * 100:+.1f}"
                band = f"±{1.96 * math.sqrt(se ** 2 + base_se ** 2) * 100:.1f}"
            md.append(f"| {model} | {c} | {m:.4f} | {delta} | {band} |")
            if a.per_task:
                md.append("| | | " + ", ".join(f"{t}={data[c][t][0]:.3f}" for t in ACC_TASKS) + " | | |")
    text = "\n".join(md) + "\n"
    print(text)
    if a.md_out:
        os.makedirs(os.path.dirname(a.md_out) or ".", exist_ok=True)
        open(a.md_out, "w").write(text)
        print("wrote", a.md_out)


if __name__ == "__main__":
    main()
