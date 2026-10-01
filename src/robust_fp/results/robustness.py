# Copyright (c) 2026 the authors.
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""The detection axis: worst-case z over the robustness protocol, and the query budget N_min.

The robustness protocol generates ``N`` responses from the fingerprinted model under eleven
deployment conditions (``ARMS``), five independent times each, and scores every (arm, run) with the
cell's own detector. The cell's detection statistic is

    worst-z = min over arms of ( mean over the 5 runs of z )

The runs are replicates of one condition and are averaged; the arms are different threats and the
worst one counts.

``N_min`` is the smallest query budget at which the worst-case z clears the detection gate and
stays above it at every larger budget on the grid.
"""
import glob
import json
import os
import statistics
from typing import Dict, Optional

#: Detection gate on the pooled z-score.
GATE_Z = 4.0

#: The eleven deployment conditions of the robustness protocol.
ARMS = ["clean", "T04", "T10", "fp8", "int4", "sys0", "sys1", "sys2", "Wanda", "SparseGPT", "ft"]
RUNS = 5


def _z_key(row: dict) -> str:
    """The statistic a detection row is read with.

    ``detect.py`` and ``query_budget.py`` record it as ``z_key``. For a row written without it, the
    rule is derived from the recorded ``teacher`` name: the near-tie teachers and SWEET restrict the
    scored positions (``elig_z``), every other teacher is read over all positions (``all_z``).
    """
    if row.get("z_key") in ("elig_z", "all_z"):
        return row["z_key"]
    if "teacher" not in row:
        raise KeyError(f"detection row {row.get('label', row.get('cell'))!r} records neither z_key nor teacher")
    name = row["teacher"]
    return "elig_z" if ("neartie" in name or "sweet" in name) else "all_z"


def detect_path(results: str, model: str, cell: str) -> str:
    return os.path.join(results, "detection", model, cell, "detect.jsonl")


def _rows(results, model, cell) -> Optional[Dict[str, dict]]:
    p = detect_path(results, model, cell)
    if not os.path.exists(p):
        return None
    rec = {}
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rec[(r["arm"], int(r["run"]))] = r
    return rec or None


def cell_arms(results: str, model: str, cell: str, arms=None, runs: int = RUNS) -> Optional[Dict[str, float]]:
    """``{arm: mean z over the runs}`` under the cell's own detector, or ``None`` unless every arm
    has all of its runs. A minimum over a subset of the arms would be an optimistic statistic."""
    rec = _rows(results, model, cell)
    if rec is None:
        return None
    out = {}
    for arm in arms or ARMS:
        zs = []
        for run in range(runs):
            r = rec.get((arm, run))
            if r is None:
                return None
            z = r.get(_z_key(r))
            if not isinstance(z, (int, float)):
                return None
            zs.append(z)
        out[arm] = statistics.mean(zs)
    return out


def worst_z(results: str, model: str, cell: str, arms=None, runs: int = RUNS) -> Optional[float]:
    per = cell_arms(results, model, cell, arms=arms, runs=runs)
    return None if per is None else min(per.values())


# ---------------------------------------------------------------------------- query budget
def query_budget_curves(results: str, model: str) -> Dict[tuple, list]:
    """``{(cell, arm, N): [z per run]}`` from every ``query_budget/<model>/curves*.jsonl`` file.

    Rows are de-duplicated on (cell, arm, run, N), keeping the last, so a budget scored twice is
    counted once.
    """
    seen = {}
    for p in sorted(glob.glob(os.path.join(results, "query_budget", model, "curves*.jsonl"))):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                seen[(r["cell"], r["arm"], r["run"], r["N"])] = r
    out = {}
    for r in seen.values():
        out.setdefault((r["cell"], r["arm"], r["N"]), []).append(r["z"])
    return out


def worst_z_at(curves: Dict[tuple, list], cell: str, N: int, arms=None, runs: int = RUNS) -> float:
    """Worst-case z at budget ``N``: mean over runs, then min over arms. Raises if a cell is not
    fully scored at that budget."""
    vals = []
    for arm in arms or ARMS:
        zs = [z for z in curves.get((cell, arm, N), []) if isinstance(z, (int, float))]
        if len(zs) != runs:
            raise ValueError(f"{cell} {arm} at N={N}: {len(zs)} runs, expected {runs}")
        vals.append(statistics.mean(zs))
    return min(vals)


def qmin(curves: Dict[tuple, list], cell: str, gate: float = GATE_Z, arms=None, runs: int = RUNS) -> Optional[int]:
    """Smallest budget on the grid at which worst-case z >= ``gate`` at that budget and at every
    larger one. ``None`` when the cell never clears the gate."""
    budgets = sorted({n for (c, _, n) in curves if c == cell})
    if not budgets:
        return None
    curve = {n: worst_z_at(curves, cell, n, arms=arms, runs=runs) for n in budgets}
    ok = None
    for n in reversed(budgets):
        if curve[n] >= gate:
            ok = n
        else:
            break
    return ok


def load_qmin(results: str) -> Dict[str, Dict[str, int]]:
    """``{model: {cell: N_min}}`` written by ``scripts/detection/reduce_query_budget.py``."""
    with open(os.path.join(results, "query_budget", "qmin.json")) as fh:
        return json.load(fh)
