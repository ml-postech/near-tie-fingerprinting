# Copyright (c) 2026 the authors.
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Achievable cost as a function of the query budget.

    python scripts/figures/query_budget.py --results results --models llama3.2-3b qwen2.5-3b llama3.1-8b --no_chord

For every method M and budget N the curve shows the lowest cost of any configuration of M that is
detectable within N queries,

    D_M(N) = min { cost(cell) : cell of M, N_min(cell) <= N },

a non-increasing step function: a larger budget can only unlock cheaper configurations. Cost is
PPL_fp / PPL_base on a log-scaled axis (``--axis ratio``) or the LLM-judge drop (``--axis judge``).
The dashed chord joins a baseline's frontier points; ``--no_chord`` draws the measured cells alone.
The shaded band is the gap between our curve and the best baseline at each budget.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from style import LEGEND, ORDER, PANEL, STYLE, scaled_font, setup  # noqa: E402
from provenance import write_cmd  # noqa: E402
from robust_fp.results import JudgeRun, UtilityTable, family_of, is_designed, load_qmin, model_title  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", default="results")
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--axis", default="ratio", choices=["ratio", "judge"])
ap.add_argument("--max_budget", type=int, default=1000, help="largest budget drawn")
ap.add_argument("--no_chord", action="store_true")
ap.add_argument("--judge_prefix", default="judge_")
ap.add_argument("--report", action="store_true", help="print the margin to the best baseline at every budget")
ap.add_argument("--panel", nargs=2, type=float, default=None, metavar=("W", "H"))
ap.add_argument("--bbox", default="tight", choices=["tight", "fixed"])
ap.add_argument("--layout_pad", type=float, default=None)
ap.add_argument("--w_pad", type=float, default=None)
ap.add_argument("--h_pad", type=float, default=None)
ap.add_argument("--font_scale", type=float, default=1.0)
ap.add_argument("--ylim", nargs=2, type=float, default=None)
ap.add_argument("--no_legend", action="store_true")
ap.add_argument("--legend_panel", type=int, default=-1)
ap.add_argument("--legend_outside", default=None, choices=["top", "bottom"])
ap.add_argument("--legend_fontsize", type=float, default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--out_dir", default="figures")
A = ap.parse_args()

if A.legend_outside and A.bbox == "fixed":
    sys.exit("--legend_outside needs --bbox tight; a fixed canvas clips the strip")

FONT = scaled_font(A.font_scale)
setup(FONT)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

STY = {k: dict(c=v["color"], mk=v["marker"], ms=6.4 * v["size"], lbl=v["label"], lw=v["lw"], z=v["z"])
       for k, v in STYLE.items()}
GRID = list(range(100, 1001, 100))
QMIN = load_qmin(A.results)
BASELINES = [f for f in ORDER if f != "kgwnt"]


def cells(model, family):
    """Sorted ``[(N_min, cost)]`` of the family's designed cells that clear the gate within ``--max_budget``."""
    util = UtilityTable(A.results, model)
    judge = JudgeRun(A.results, model, prefix=A.judge_prefix) if A.axis == "judge" else None
    out = []
    for cell, n_min in QMIN.get(model, {}).items():
        fam, _ = family_of(cell)
        if fam != family or n_min > A.max_budget or not is_designed(cell):
            continue
        if judge is not None:
            base, m = judge.mean("base"), judge.mean(cell)
            d = None if base is None or m is None else base - m
        else:
            d = util.ratio(cell)
        if d is not None:
            out.append((n_min, d))
    return sorted(out)


def pareto(P):
    out, best = [], float("inf")
    for q, d in P:
        if d < best:
            best = d
            out.append((q, d))
    return out


def stepfn(P, grid):
    return [min([d for q, d in P if q <= g], default=None) for g in grid]


PW, PH = A.panel or (PANEL["w"], PANEL["h"])
fig, axes = plt.subplots(1, len(A.models), figsize=(PW * len(A.models), PH), sharey=True, squeeze=False)
axes = axes[0]
for ax, model in zip(axes, A.models):
    ours = stepfn(cells(model, "kgwnt"), GRID)
    per = {f: stepfn(cells(model, f), GRID) for f in BASELINES}
    bx, blo, bhi = [], [], []
    for i, g in enumerate(GRID):
        cand = [per[f][i] for f in BASELINES if per[f][i] is not None]
        if ours[i] is not None and cand:
            bx.append(g)
            blo.append(ours[i])
            bhi.append(min(cand))
    if bx:
        ax.fill_between(bx, blo, bhi, step="post", color="#0F7B6C", alpha=0.085, lw=0, zorder=0)
    if A.report:
        print(f"\n=== {model}  axis={A.axis} (lower is better)")
        for i, g in enumerate(GRID):
            cand = {f: per[f][i] for f in BASELINES if per[f][i] is not None}
            if ours[i] is None or not cand:
                print(f"  N={g:5d}  no feasible cell")
                continue
            bf = min(cand, key=cand.get)
            print(f"  N={g:5d}  ours {ours[i]:.3f}  best baseline {cand[bf]:.3f} ({bf})  margin {cand[bf] - ours[i]:+.3f}")

    for fam in ORDER:
        st = STY[fam]
        P = cells(model, fam)
        if not P:
            continue
        F = pareto(P)
        step = stepfn(P, GRID)
        xs = [g for g, v in zip(GRID, step) if v is not None]
        ys = [v for v in step if v is not None]
        if xs:
            ax.step(xs, ys, where="post", color=st["c"], lw=st["lw"], zorder=st["z"], solid_capstyle="round")
        if fam != "kgwnt" and len(F) > 1 and not A.no_chord:
            ax.plot([q for q, _ in F], [d for _, d in F], ls=(0, (4, 2.5)), lw=1.0, color=st["c"], alpha=0.75,
                    zorder=st["z"])
        ax.plot([q for q, _ in F], [d for _, d in F], st["mk"], ms=st["ms"], color=st["c"], mec="white", mew=0.8,
                zorder=st["z"] + 1, ls="none", label=st["lbl"])
        off = [p for p in P if p not in F]
        if off:
            ax.plot([q for q, _ in off], [d for _, d in off], st["mk"], ms=st["ms"] * 0.78, color=st["c"],
                    alpha=0.55, mec="white", mew=0.5, zorder=2, ls="none")

    ax.axhline(1.0 if A.axis == "ratio" else 0.0, color="0.35", lw=0.8, ls=":", zorder=1)
    if A.axis == "ratio":
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if A.ylim:
            ax.set_ylim(*A.ylim)
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in (0.8, 1, 1.5, 2, 3, 4, 6, 8) if lo <= t <= hi])
    elif A.ylim:
        ax.set_ylim(*A.ylim)
    ax.set_title(model_title(model), fontsize=FONT["title"], pad=9)
    ax.set_xlabel(r"query budget $N$", fontsize=FONT["axis"])
    ax.set_xlim(70, 1050)
    ax.set_xticks([100, 300, 500, 700, 1000])
    if ax is axes[0]:
        ax.set_ylabel({"judge": r"LLM-judge drop $\Delta J$",
                       "ratio": r"$\mathrm{PPL}_{\mathrm{fp}}/\mathrm{PPL}_{\mathrm{base}}$"}[A.axis], fontsize=FONT["axis"])

present = set(axes[0].get_legend_handles_labels()[1])
handles = [Line2D([0], [0], color=STY[f]["c"], lw=STY[f]["lw"], marker=STY[f]["mk"], ms=STY[f]["ms"], mec="white", mew=0.8)
           for f in ORDER if STY[f]["lbl"] in present]
labels = [STY[f]["lbl"] for f in ORDER if STY[f]["lbl"] in present]
if not A.no_legend:
    if A.legend_outside:
        top = A.legend_outside == "top"
        fig.legend(handles, labels, fontsize=A.legend_fontsize or FONT["legend"], ncol=len(labels), frameon=False,
                   loc="lower center" if top else "upper center", bbox_to_anchor=(0.5, 1.0 if top else 0.0),
                   handlelength=1.5, handletextpad=0.5, columnspacing=1.4, borderaxespad=0.2)
    else:
        axes[A.legend_panel].legend(handles, labels, fontsize=A.legend_fontsize or FONT["legend"], **LEGEND)
fig.tight_layout(**{k: v for k, v in (("pad", A.layout_pad), ("w_pad", A.w_pad), ("h_pad", A.h_pad)) if v is not None})

os.makedirs(A.out_dir, exist_ok=True)
out = A.out or os.path.join(A.out_dir, f"query_budget_{A.axis}_{'-'.join(A.models)}{'_nochord' if A.no_chord else ''}.pdf")
save = dict(bbox_inches="tight") if A.bbox == "tight" else {}
fig.savefig(out, **save)
fig.savefig(out.replace(".pdf", ".png"), dpi=210, **save)
write_cmd(out)
print("wrote", out)
