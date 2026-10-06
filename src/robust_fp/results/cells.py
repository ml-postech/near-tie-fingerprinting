# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Cell naming.

A cell is one fingerprinted model, named ``<family>_<setting>``:

    kgw_d4            KGW,       delta = 4
    kgwnt_q50_d4      KGW-NT,    tau_gap at the 50th percentile of the base model's logit gap, delta = 4
    sweet_q50_d4      SWEET,     entropy threshold such that 50% of the base model's positions are high-entropy
    kgwtk_k5_d4       KGW-TK,    top-5 window, delta = 4
    morphmark_k13_p15 MorphMark, k_exp = 1.3, p_0 = 0.15
    sweetnt_q50_d4_t150       SWEET + near-tie wrapper, tau_gap = 1.50
    morphmarknt_k40_p05_t150  MorphMark + near-tie wrapper, tau_gap = 1.50

``DESIGNED_GRID`` lists the cells of each family that the paper's frontier is drawn from.
"""
FAMILIES = ("kgw", "sweet", "morphmark", "kgwtk", "kgwnt", "sweetnt", "morphmarknt")

DESIGNED_GRID = {
    "kgw":       [f"d{d}" for d in (1, 2, 4)],
    "kgwnt":     [f"q{q}_d{d}" for q in (10, 30, 50, 70) for d in (1, 2, 4)],
    "sweet":     [f"q{q}_d{d}" for q in (10, 30, 50, 70) for d in (1, 2, 4)],
    "kgwtk":     [f"k{k}_d{d}" for k in (2, 5, 10, 50) for d in (1, 2, 4)],
    "morphmark": [f"k{k}_p{p}" for k in ("05", "10", "13", "20") for p in ("15", "30", "50")],
}


def family_of(cell: str):
    """``'kgwnt_q50_d4' -> ('kgwnt', 'q50_d4')``. Longer family names are matched first."""
    for fam in sorted(FAMILIES, key=len, reverse=True):
        if cell.startswith(fam + "_"):
            return fam, cell[len(fam) + 1:]
    raise ValueError(f"{cell!r} does not start with a known family name {FAMILIES}")


def is_designed(cell: str) -> bool:
    fam, setting = family_of(cell)
    return setting in DESIGNED_GRID.get(fam, ())
