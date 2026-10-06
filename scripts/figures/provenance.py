# Copyright (c) 2026 the authors.
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Record the command that produced a figure next to the figure (``<name>.cmd.txt``)."""
import os
import subprocess
import sys


def _git_head():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=5).stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _quote(a):
    return a if a and all(c.isalnum() or c in "-_./=:," for c in a) else "'" + a.replace("'", "'\\''") + "'"


def write_cmd(out_path: str) -> str:
    stem = os.path.splitext(out_path)[0]
    line = " ".join(_quote(x) for x in ["python", os.path.relpath(sys.argv[0])] + sys.argv[1:])
    with open(stem + ".cmd.txt", "w") as fh:
        fh.write(f"# regenerates {os.path.basename(stem)}.{{pdf,png}}\n# commit: {_git_head()}\n{line}\n")
    return stem + ".cmd.txt"
