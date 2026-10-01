"""Near-tie restriction applied to other watermarks: a watermark against its near-tie variant.

    python scripts/figures/nt_ablation.py --results results --model llama3.1-8b --pair sweet morphmark --axis both
    python scripts/figures/nt_ablation.py --results results --model llama3.1-8b --pair nt3

``--pair sweet`` draws SWEET against SWEET+NT, ``--pair morphmark`` MorphMark against MorphMark+NT,
on the same prompts, base generation, oracle and judge as the main frontier; ``--pair nt3`` draws
the three near-tie variants (KGW-NT, SWEET+NT, MorphMark+NT) alone. Every series is read with its
own detector. The near-tie cells are every ``sweetnt_*`` / ``morphmarknt_*`` cell with a utility
measurement, or the ones named with ``--nt_cells``.
"""
import argparse
import glob
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from style import GATE_LABEL, GATE_Z, LEGEND, PANEL, STYLE, scaled_font, setup  # noqa: E402
from provenance import write_cmd  # noqa: E402
from robust_fp.results import DESIGNED_GRID, JudgeRun, UtilityTable, family_of, worst_z  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", default="results")
ap.add_argument("--model", default="llama3.1-8b")
ap.add_argument("--pair", nargs="+", default=["morphmark"], choices=["morphmark", "sweet", "nt3"])
ap.add_argument("--axis", default="ratio", choices=["ratio", "judge", "both"])
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--judge_prefix", default="judge_")
ap.add_argument("--nt_cells", nargs="+", default=None, help="restrict the near-tie series to these cells")
ap.add_argument("--legend_loc", default="in_band", choices=["in_band", "above_gate"])
ap.add_argument("--legend_fontsize", type=float, default=None)
ap.add_argument("--no_legend", action="store_true")
ap.add_argument("--no_errorbar", action="store_true")
ap.add_argument("--point_labels", action="store_true")
ap.add_argument("--font_scale", type=float, default=1.0)
ap.add_argument("--ylim", nargs=2, type=float, default=None)
ap.add_argument("--xlim", nargs=2, type=float, default=None)
ap.add_argument("--layout", default="wide", choices=["wide", "tall"])
ap.add_argument("--panel_w", type=float, default=None)
ap.add_argument("--panel_h", type=float, default=None)
ap.add_argument("--dump", action="store_true", help="print the collected points and exit")
ap.add_argument("--out", default=None)
ap.add_argument("--out_dir", default="figures")
A = ap.parse_args()

FONT = scaled_font(A.font_scale)
setup(FONT)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

_NT = dict(c=STYLE["kgwnt"]["color"], mk=STYLE["kgwnt"]["marker"], ms=7.8 * STYLE["kgwnt"]["size"])
STY = {
    "morphmark":    dict(c=STYLE["morphmark"]["color"], mk=STYLE["morphmark"]["marker"], ms=7.8, lbl="MorphMark"),
    "morphmark_nt": dict(_NT, lbl="MorphMark-NT"),
    "sweet":        dict(c=STYLE["sweet"]["color"], mk=STYLE["sweet"]["marker"], ms=7.8, lbl="SWEET"),
    "sweet_nt":     dict(_NT, lbl="SWEET-NT"),
    # In the nt3 panel each near-tie variant keeps the colour of the watermark it restricts.
    "nt_kgw":       dict(_NT, lbl="KGW-NT"),
    "nt_sweet":     dict(c=STYLE["sweet"]["color"], mk=STYLE["sweet"]["marker"], ms=7.8, lbl="SWEET-NT"),
    "nt_morphmark": dict(c=STYLE["morphmark"]["color"], mk=STYLE["morphmark"]["marker"], ms=7.8, lbl="MorphMark-NT"),
}
SERIES = {"morphmark": ["morphmark", "morphmark_nt"], "sweet": ["sweet", "sweet_nt"],
          "nt3": ["nt_kgw", "nt_sweet", "nt_morphmark"]}
FAMILY = {"morphmark": "morphmark", "morphmark_nt": "morphmarknt", "sweet": "sweet", "sweet_nt": "sweetnt",
          "nt_kgw": "kgwnt", "nt_sweet": "sweetnt", "nt_morphmark": "morphmarknt"}

util = UtilityTable(A.results, A.model)
judge = JudgeRun(A.results, A.model, prefix=A.judge_prefix)
JBASE = judge.mean("base")
if A.axis in ("judge", "both") and JBASE is None:
    sys.exit(f"no judge score for the base model under results/judge/{A.model}")


def tag(family, setting):
    m = re.search(r"_t(\d{3})$", setting)
    if m:
        return f"τ{int(m.group(1)) / 100:.1f}"
    return setting.replace("_", "")


def candidates(family):
    if family in DESIGNED_GRID:
        settings = list(DESIGNED_GRID[family])
        if family == "morphmark":
            settings += ["k30_p15", "k40_p05", "k40_p70"]      # the stronger MorphMark settings of the ablation
        return [f"{family}_{s}" for s in settings]
    found = sorted(os.path.basename(p)[len("comps_"):-len(".json")]
                   for p in glob.glob(os.path.join(util.dir, f"comps_{family}_*.json")))
    return found


def collect(series):
    pts = []
    for cell in candidates(FAMILY[series]):
        if not util.has(cell):
            continue
        z = worst_z(A.results, A.model, cell)
        if z is None:
            continue
        lo, hi = util.paired_ci(cell, axis="log", boot=A.boot)
        pts.append(dict(cell=cell, tag=tag(*family_of(cell)), x=util.ratio(cell), xerr=(hi - lo) / 2, z=z,
                        j=judge.mean(cell)))
    return pts


DATA = {pair: {s: collect(s) for s in SERIES[pair]} for pair in A.pair}
if A.nt_cells is not None:
    nt_keys = [k for k in STY if k.endswith("_nt") or k.startswith("nt_")]
    known = {p["cell"] for D in DATA.values() for k, v in D.items() if k in nt_keys for p in v}
    unknown = set(A.nt_cells) - known
    if unknown:
        sys.exit(f"--nt_cells names {sorted(unknown)}, which were not collected; available: {sorted(known)}")
    for D in DATA.values():
        for k in [k for k in D if k in nt_keys]:
            hit = [p for p in D[k] if p["cell"] in A.nt_cells]
            if hit:
                D[k] = hit

ROW_X = {"ratio": lambda p: p["x"], "judge": lambda p: None if p["j"] is None else JBASE - p["j"]}


def pareto(pts, row):
    getx = ROW_X[row]
    s = sorted([p for p in pts if p["z"] >= GATE_Z and getx(p) is not None], key=lambda p: (getx(p), -p["z"]))
    out, best = [], -1e9
    for p in s:
        if p["z"] > best:
            out.append(p)
            best = p["z"]
    return out


if A.dump:
    for pair, D in DATA.items():
        for key, pts in D.items():
            front = {id(p) for p in pareto(pts, "ratio")}
            print(f"\n== {pair} / {key} ==")
            for p in sorted(pts, key=lambda q: q["x"]):
                flag = "FRONTIER" if id(p) in front else ("below-gate" if p["z"] < GATE_Z else "dominated")
                jj = "  --  " if p["j"] is None else f"{p['j']:6.3f}"
                print(f"{p['cell']:30s} {p['tag']:8s} ratio={p['x']:7.3f} z={p['z']:6.2f} judge={jj}  {flag}")
    sys.exit(0)

allp = [p for D in DATA.values() for v in D.values() for p in v]
if not allp:
    sys.exit("no cells with both a utility measurement and a complete detection record")
ymin = min(min(p["z"] for p in allp) - 1.5, GATE_Z - 1.5)
ymax = max(p["z"] for p in allp) + 1.8
ROWS = ["ratio", "judge"] if A.axis == "both" else [A.axis]
XLIM = {}
for row in ROWS:
    if row == "judge":
        v = [JBASE - p["j"] for p in allp if p["j"] is not None] + [0.0]
        pad = 0.06 * (max(v) - min(v))
        XLIM[row] = (min(v) - pad, max(v) + pad)
    else:
        lo = math.log(min(p["x"] * math.exp(-p["xerr"]) for p in allp))
        hi = math.log(max(p["x"] * math.exp(p["xerr"]) for p in allp))
        XLIM[row] = (math.exp(lo - 0.05 * (hi - lo)), math.exp(hi + 0.12 * (hi - lo)))

NR = len(A.pair) if A.layout == "tall" else len(ROWS)
NC = len(ROWS) if A.layout == "tall" else len(A.pair)
fig, grid = plt.subplots(NR, NC, figsize=((A.panel_w or PANEL["w"]) * NC, (A.panel_h or PANEL["h"]) * NR),
                         sharey=True, squeeze=False)
for ri, row in enumerate(ROWS):
    for ci, pair in enumerate(A.pair):
        ax = grid[ci][ri] if A.layout == "tall" else grid[ri][ci]
        D = DATA[pair]
        getx = ROW_X[row]
        xmin, xmax = A.xlim or XLIM[row]
        lo_y, hi_y = A.ylim or (ymin, ymax)
        ax.axhspan(lo_y, GATE_Z, color="0.5", alpha=0.07, lw=0, zorder=0)
        ax.axhline(GATE_Z, ls=(0, (5, 4)), color="0.45", lw=1.0, zorder=1)
        ax.axvline(0.0 if row == "judge" else 1.0, ls=":", color="0.45", lw=1.0, zorder=1)
        if row != "judge":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks([t for t in (1, 1.5, 2, 3, 5) if xmin <= t <= xmax])
        for key in SERIES[pair]:
            s, pts = STY[key], [q for q in D.get(key, []) if getx(q) is not None]
            if not pts:
                continue
            ax.scatter([getx(p) for p in pts], [p["z"] for p in pts], marker=s["mk"], s=s["ms"] ** 2 * 0.75,
                       facecolor=s["c"], edgecolor="white", linewidth=0.4, alpha=0.30, zorder=2)
            pf = pareto(pts, row)
            if not pf:
                continue
            if row != "judge" and not A.no_errorbar:
                lo = [p["x"] - p["x"] * math.exp(-p["xerr"]) for p in pf]
                hi = [p["x"] * math.exp(p["xerr"]) - p["x"] for p in pf]
                ax.errorbar([p["x"] for p in pf], [p["z"] for p in pf], xerr=[lo, hi], fmt="none", ecolor=s["c"],
                            elinewidth=1.1, alpha=0.75, capsize=2.5, zorder=5)
            ax.plot([getx(p) for p in pf], [p["z"] for p in pf], marker=s["mk"], color=s["c"], ms=s["ms"] + 1.0,
                    lw=2.2, label=s["lbl"], markerfacecolor=s["c"], markeredgecolor="white", markeredgewidth=0.8,
                    zorder=6)
            is_nt = key.endswith("_nt") or key.startswith("nt_")
            for i, p in enumerate(pf if A.point_labels else []):
                edge = i in (0, len(pf) - 1)
                if (not is_nt or row == "judge") and not edge:
                    continue
                if i == 0:
                    off, ha, va = (0, -19), "center", "top"
                elif i == len(pf) - 1:
                    off, ha, va = ((0, 12), "center", "bottom") if is_nt else ((10, -3), "left", "center")
                else:
                    dy, va = (6, "bottom") if i % 2 else (-11, "top")
                    off, ha = ((-10 if is_nt else 10), dy), ("right" if is_nt else "left")
                ax.annotate(p["tag"], (getx(p), p["z"]), fontsize=9.0, color=s["c"], ha=ha, va=va, xytext=off,
                            textcoords="offset points", zorder=7)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(lo_y, hi_y)
        bottom = (ci == len(A.pair) - 1) if A.layout == "tall" else True
        if not bottom:
            ax.tick_params(labelbottom=False)
        ax.set_xlabel({"judge": r"LLM-judge drop   $\Delta J$",
                       "ratio": r"$\mathrm{PPL}_{\mathrm{fp}}\,/\,\mathrm{PPL}_{\mathrm{base}}$   (log scale)"}[row]
                      if bottom else "", fontsize=FONT["axis"])
        if (ri == 0 if A.layout == "tall" else ci == 0):
            ax.set_ylabel(r"Worst-case $z$-score", fontsize=FONT["axis"])
            ax.text(xmax - 0.02 * (xmax - xmin), GATE_Z - 0.45, GATE_LABEL, fontsize=FONT["note"], ha="right",
                    va="top", color="0.4")
        if ri == 0 and not A.no_legend:
            gy = (GATE_Z - lo_y) / float(hi_y - lo_y) + 0.03 if A.legend_loc == "above_gate" else 0.0
            lg = dict(LEGEND, loc="lower right", bbox_to_anchor=(1.0, gy), frameon=False)
            for k in ("framealpha", "facecolor", "edgecolor"):
                lg.pop(k, None)
            ax.legend(fontsize=A.legend_fontsize or FONT["legend"], **lg)

fig.tight_layout()
os.makedirs(A.out_dir, exist_ok=True)
out = A.out or os.path.join(A.out_dir, f"nt_ablation_{'_'.join(A.pair)}_{A.axis}{'' if A.layout == 'wide' else '_tall'}.pdf")
for ext in ("pdf", "png"):
    fig.savefig(out.replace(".pdf", f".{ext}"), dpi=300, bbox_inches="tight")
write_cmd(out)
print("wrote", out)
for pair in A.pair:
    for k, v in DATA[pair].items():
        print(f"  {k:14s} {len(v)} cells, {len(pareto(v, 'ratio'))} on the frontier")
