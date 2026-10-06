# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""The detection / utility frontier: worst-case z against the cost of the fingerprint.

    python scripts/figures/frontier.py --results results --models qwen2.5-3b llama3.1-8b --axis ratio
    python scripts/figures/frontier.py --results results --models llama3.1-8b --axis judge

Every cell of the designed grid is drawn faded; the non-dominated cells among those that clear the
detection gate are joined into the method's frontier. ``--axis ratio`` puts PPL_fp / PPL_base on a
log-scaled x axis, ``--axis judge`` the drop in LLM-judge score, ``--axis both`` one row of each.
The paired-bootstrap confidence interval of the cost is drawn unless ``--no_errorbar``.
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from style import GATE_LABEL, GATE_Z, LEGEND, ORDER, PANEL, STYLE, scaled_font, setup  # noqa: E402
from provenance import write_cmd  # noqa: E402
from robust_fp.results import DESIGNED_GRID, JudgeRun, UtilityTable, model_title, worst_z  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", default="results")
ap.add_argument("--models", nargs="+", required=True, help="panel order")
ap.add_argument("--axis", default="ratio", choices=["ratio", "judge", "both"])
ap.add_argument("--stat", default="mean", choices=["mean", "median"])
ap.add_argument("--methods", nargs="+", default=None, choices=ORDER, help="restrict to these families")
ap.add_argument("--out", default=None, help="file stem (default frontier_<axis>_<models>)")
ap.add_argument("--out_dir", default="figures")
# layout
ap.add_argument("--panel", nargs=2, type=float, default=None, metavar=("W", "H"), help="panel size in inches")
ap.add_argument("--bbox", default="tight", choices=["tight", "fixed"],
                help="tight crops to the content; fixed saves exactly the panel grid size")
ap.add_argument("--pad", type=float, default=0.1, help="border kept around a tight bounding box, inches")
ap.add_argument("--layout_pad", type=float, default=None, help="tight_layout pad, in font-size units")
ap.add_argument("--w_pad", type=float, default=None)
ap.add_argument("--h_pad", type=float, default=None)
ap.add_argument("--font_scale", type=float, default=1.0)
ap.add_argument("--marker_scale", type=float, default=1.0)
ap.add_argument("--line_scale", type=float, default=1.0)
ap.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("LO", "HI"))
ap.add_argument("--yticks", nargs="+", type=float, default=None)
ap.add_argument("--xticks", nargs="+", type=float, default=None, help="ticks of the ratio axis")
ap.add_argument("--xscale", default="log", choices=["log", "linear"], help="scale of the ratio axis")
ap.add_argument("--ylabel", default=None)
ap.add_argument("--scale_in_label", action="store_true", help="append '(log scale)' to the ratio axis label")
ap.add_argument("--no_title", action="store_true")
ap.add_argument("--no_errorbar", action="store_true")
ap.add_argument("--errorbar_on_top", action="store_true")
# legend
ap.add_argument("--no_legend", action="store_true")
ap.add_argument("--legend_panel", type=int, default=-1, help="panel that carries the legend")
ap.add_argument("--legend_loc", default="upper left")
ap.add_argument("--legend_bbox", nargs=2, type=float, default=None, metavar=("X", "Y"))
ap.add_argument("--legend_fontsize", type=float, default=None)
ap.add_argument("--legend_ncol", type=int, default=1)
ap.add_argument("--legend_outside", default=None, choices=["top", "bottom"], help="one legend strip outside the panels")
ap.add_argument("--dump_cells", action="store_true", help="print the collected cells and exit")
A = ap.parse_args()

if A.legend_outside and A.bbox == "fixed":
    sys.exit("--legend_outside needs --bbox tight; a fixed canvas clips the strip")

FONT = scaled_font(A.font_scale)
setup(FONT)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

FAMILIES = [f for f in ORDER if not A.methods or f in A.methods]
STY = {k: dict(c=v["color"], mk=v["marker"], ms=7.8 * v["size"], lbl=v["label"]) for k, v in STYLE.items()}
XLAB = {"ratio": r"$\mathrm{PPL}_{\mathrm{fp}}\,/\,\mathrm{PPL}_{\mathrm{base}}$",
        "judge": r"LLM-judge drop ($\Delta J$)"}


def pretty(family, setting):
    if family == "kgw":
        v = float(setting[1:])
        return rf"$\delta\!=\!{v:g}$"
    if family == "morphmark":
        return setting.replace("_p", ",p")
    q, d = setting.split("_d")
    return f"{q},δ{d}"


def load(model, axis):
    """``{family: [point]}`` with ``point = {x, xerr, z, tag, cell}``."""
    D = {f: [] for f in FAMILIES}
    util = UtilityTable(A.results, model)
    if axis == "judge":
        judge = JudgeRun(A.results, model)
        if not judge.has_base:
            return D

        def cost(cell):
            return judge.drop(cell)
    else:
        if not util.exists:
            return D

        def cost(cell):
            r = util.ratio(cell, A.stat)
            ci = util.paired_ci(cell, A.stat, axis="ratio")
            if r is None or ci is None:
                return None
            return r, max(r - ci[0], ci[1] - r)

    for family in FAMILIES:
        for setting in DESIGNED_GRID[family]:
            cell = f"{family}_{setting}"
            if not os.path.exists(util.comps_path(cell)):
                continue
            z = worst_z(A.results, model, cell)
            c = cost(cell)
            if z is None or c is None:
                continue
            D[family].append(dict(x=c[0], xerr=c[1], z=z, tag=pretty(family, setting), cell=cell))
    return D


def pareto(pts):
    """Non-dominated cells among those that clear the gate."""
    s = sorted([p for p in pts if p["z"] >= GATE_Z], key=lambda p: (p["x"], -p["z"]))
    out, best = [], -1e9
    for p in s:
        if p["z"] > best:
            out.append(p)
            best = p["z"]
    return out


ROWS = ["ratio", "judge"] if A.axis == "both" else [A.axis]
DATA = {row: {m: load(m, row) for m in A.models} for row in ROWS}

if A.dump_cells:
    for row in ROWS:
        for m in A.models:
            pts = [p for v in DATA[row][m].values() for p in v]
            front = {id(p) for p in pareto(pts)}
            print(f"\n== {m} / {row} ==")
            for p in sorted(pts, key=lambda p: p["x"]):
                flag = "FRONTIER" if id(p) in front else ("below-gate" if p["z"] < GATE_Z else "dominated")
                print(f"   {p['cell']:26s} x={p['x']:8.4f} ±{p['xerr']:.4f}  z={p['z']:6.2f}  {flag}")
    sys.exit(0)

PW, PH = A.panel or (PANEL["w"], PANEL["h"])
fig, grid = plt.subplots(len(ROWS), len(A.models), figsize=(PW * len(A.models), PH * len(ROWS)),
                         sharey=True, squeeze=False)
for ri, row in enumerate(ROWS):
    pts_all = [p for m in A.models for v in DATA[row][m].values() for p in v]
    if not pts_all:
        continue
    allx = [p["x"] + p["xerr"] for p in pts_all]
    allxl = [p["x"] - p["xerr"] for p in pts_all]
    allz = [p["z"] for p in pts_all]
    ymin, ymax = min(min(allz) - 1.5, GATE_Z - 1.5), max(allz) + 1.8
    if A.ylim:
        ymin, ymax = A.ylim
    if row == "ratio" and A.xscale == "log":
        lo, hi = math.log(max(min(allxl), 1e-3)), math.log(max(allx))
        span = hi - lo
        xmin, xmax = min(math.exp(lo - 0.05 * span), 1.0), math.exp(hi + 0.10 * span)
    else:
        span = max(allx) - min(allxl)
        xmin, xmax = min(allxl) - 0.05 * span, max(allx) + 0.10 * span
        if row == "ratio":
            xmin = min(xmin, 1.0)

    for ci, model in enumerate(A.models):
        ax, D = grid[ri][ci], DATA[row][model]
        ax.axhspan(ymin, GATE_Z, color="0.5", alpha=0.07, lw=0, zorder=0)
        ax.axhline(GATE_Z, ls=(0, (5, 4)), color="0.45", lw=1.0, zorder=1)
        ax.axvline(1.0 if row == "ratio" else 0.0, ls=":", color="0.45", lw=1.0, zorder=1)
        if row == "ratio":
            ax.set_xscale(A.xscale)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks(A.xticks if A.xticks else [t for t in (1, 1.5, 2, 3, 4, 6, 8, 12) if xmin <= t <= xmax])
        for family in FAMILIES:
            s, pts = STY[family], D[family]
            if not pts:
                continue
            ax.scatter([p["x"] for p in pts], [p["z"] for p in pts], marker=s["mk"],
                       s=(s["ms"] * A.line_scale * A.marker_scale) ** 2 * 0.75, facecolor=s["c"],
                       edgecolor="white", linewidth=0.4 * A.line_scale, alpha=0.30, zorder=2)
            pf = pareto(pts)
            if not pf:
                continue
            if not A.no_errorbar:
                ax.errorbar([p["x"] for p in pf], [p["z"] for p in pf], xerr=[p["xerr"] for p in pf],
                            fmt="none", ecolor=s["c"], elinewidth=1.1 * A.line_scale, alpha=0.75,
                            capsize=2.5 * A.line_scale, zorder=7 if A.errorbar_on_top else 5)
            ax.plot([p["x"] for p in pf], [p["z"] for p in pf], marker=s["mk"], color=s["c"],
                    ms=(s["ms"] + 1.0) * A.line_scale * A.marker_scale, lw=2.2 * A.line_scale,
                    solid_capstyle="round", solid_joinstyle="round",
                    label=s["lbl"] if (ri == 0 and ci == 0) else None,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if A.yticks:
            ax.set_yticks(A.yticks)
        ax.set_xlabel(XLAB[row] + ("  (log scale)" if A.scale_in_label and row == "ratio" and A.xscale == "log" else ""),
                      fontsize=FONT["axis"])
        if ri == 0 and not A.no_title:
            ax.set_title(model_title(model), fontsize=FONT["title"], pad=6)
        if ci == 0:
            ax.set_ylabel(A.ylabel or r"worst-case $z$ @ $N=1000$", fontsize=FONT["axis"])
            ax.text(xmax - 0.02 * span, GATE_Z - 0.45, GATE_LABEL, fontsize=FONT["note"], color="0.4",
                    ha="right", va="top")

handles, labels = grid[0][0].get_legend_handles_labels()
if labels and not A.no_legend:
    if A.legend_outside:
        top = A.legend_outside == "top"
        fig.legend(handles, labels, fontsize=FONT["legend"], ncol=len(labels), frameon=False,
                   loc="lower center" if top else "upper center", bbox_to_anchor=(0.5, 1.0 if top else 0.0),
                   handlelength=1.5, handletextpad=0.5, columnspacing=1.4, borderaxespad=0.2)
    else:
        panels = [(r, c) for r in range(len(ROWS)) for c in range(len(A.models))]
        lax = grid[panels[A.legend_panel][0]][panels[A.legend_panel][1]]
        anchor = A.legend_bbox or {"upper left": (0.02, 1.0), "upper right": (0.98, 1.0)}.get(A.legend_loc)
        kw = {**LEGEND, "loc": A.legend_loc, "ncol": A.legend_ncol}
        if anchor:
            kw["bbox_to_anchor"] = anchor
        lax.legend(handles, labels, fontsize=A.legend_fontsize or FONT["legend"], **kw)
fig.tight_layout(**{k: v for k, v in (("pad", A.layout_pad), ("w_pad", A.w_pad), ("h_pad", A.h_pad)) if v is not None})

os.makedirs(A.out_dir, exist_ok=True)
name = A.out or (f"frontier_{A.axis}" + (f"_{'-'.join(FAMILIES)}" if A.methods else "") + f"_{'-'.join(A.models)}")
save = dict(bbox_inches="tight", pad_inches=A.pad) if A.bbox == "tight" else {}
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(A.out_dir, f"{name}.{ext}"), dpi=300, **save)
write_cmd(os.path.join(A.out_dir, name))
print(f"wrote {A.out_dir}/{name}.{{pdf,png}}")
for row in ROWS:
    for model in A.models:
        print(f"\n--- {model} frontier ({row}) ---")
        for family in FAMILIES:
            pts = DATA[row][model][family]
            below = sum(1 for p in pts if p["z"] < GATE_Z)
            print(f"  {STY[family]['lbl']:16s} [{len(pts)} cells, {below} below gate]  " +
                  "  ".join(f"{p['tag']}: z={p['z']:.2f} cost={p['x']:.3f}±{p['xerr']:.3f}" for p in pareto(pts)))
