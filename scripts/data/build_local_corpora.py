# Copyright (c) 2026 the authors.
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Materialise the training corpora as parquet files for nodes without network access.

    python scripts/data/build_local_corpora.py --out /data/fp_corpora --luciefr_docs 200000 --openwebtext_docs 200000

Writes ``luciefr.parquet``, ``alpaca_gpt4.parquet`` and ``openwebtext.parquet`` with the Hub
column names. Point ``FP_LOCAL_DATA`` at the output directory and the training loaders read these
files instead of streaming from the Hub. The LucieFr and OpenWebText subsets are the first
documents of each stream, in stream order.
"""
import argparse
import itertools
import os

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--luciefr_docs", type=int, default=200_000)
ap.add_argument("--openwebtext_docs", type=int, default=200_000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def write(rows, path):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    print(f"wrote {path}: {table.num_rows} rows, columns {table.column_names}")


lucie = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "RedPajama-fr", split="train", streaming=True)
write([{"text": r["text"]} for r in itertools.islice(lucie, a.luciefr_docs)], os.path.join(a.out, "luciefr.parquet"))

alpaca = load_dataset("vicgalle/alpaca-gpt4", split="train")
write([{"instruction": r["instruction"], "input": r["input"], "output": r["output"]} for r in alpaca],
      os.path.join(a.out, "alpaca_gpt4.parquet"))

owt = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
write([{"text": r["text"]} for r in itertools.islice(owt, a.openwebtext_docs)], os.path.join(a.out, "openwebtext.parquet"))
