# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Presentation shared by every figure: one colour and marker per method, one type scale."""
import matplotlib

__all__ = ["STYLE", "ORDER", "LABEL", "FONT", "GATE_Z", "GATE_LABEL", "PANEL", "LEGEND", "setup", "scaled_font"]

# KGW and its two restrictions are achromatic and red; the two other published methods take the
# cool colours. Marker shapes stay distinct so the figures read in black and white.
STYLE = {
    "kgw":       dict(color="#000000", marker="^", label="KGW",           size=1.00, lw=1.6, z=3),
    "sweet":     dict(color="#1F6FB2", marker="s", label="SWEET",         size=1.00, lw=1.6, z=3),
    "morphmark": dict(color="#2E8B57", marker="D", label="MorphMark",     size=1.00, lw=1.6, z=3),
    "kgwtk":     dict(color="#808080", marker="P", label="KGW-TK",        size=1.00, lw=1.6, z=4),
    "kgwnt":     dict(color="#D62728", marker="o", label="KGW-NT (ours)", size=1.00, lw=1.6, z=5),
}
ORDER = ["kgw", "sweet", "morphmark", "kgwtk", "kgwnt"]
LABEL = {k: v["label"] for k, v in STYLE.items()}

FONT = dict(base=13, title=13.5, axis=13.5, legend=11.5, annot=9.5, note=9.5)

GATE_Z = 4.0
GATE_LABEL = r"$z\!=\!4$   $(p\!\approx\!3\times10^{-5})$"

PANEL = dict(w=5.2, h=4.5)

LEGEND = dict(loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0,
              frameon=True, framealpha=1.0, facecolor="white", edgecolor="none", borderpad=0.35,
              handlelength=1.5, handletextpad=0.5, labelspacing=0.32)


def scaled_font(scale: float) -> dict:
    return {k: v * scale for k, v in FONT.items()}


def setup(font: dict) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": font["base"],
        "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
