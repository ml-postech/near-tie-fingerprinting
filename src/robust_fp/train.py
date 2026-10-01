# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Embed a fingerprint: distil a watermark into an instruction-tuned model.

    python -m robust_fp.train --config configs/llama3.1-8b/kgwnt_q50_d4.yaml

The student is written to ``<FP_MODELS_DIR>/<base model name>_<custom_name>`` (``FP_MODELS_DIR``
defaults to ``models``), or to ``--output_dir`` when given.
"""
import argparse
import os

from robust_fp.config import MainConfiguration
from robust_fp.finetuning.finetune import finetune_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="experiment configuration (YAML)")
    ap.add_argument("--output_dir", default=None, help="where to save the student")
    args = ap.parse_args()

    cfg = MainConfiguration.from_yaml(args.config)
    if cfg.finetuning is None:
        raise SystemExit(f"{args.config} has no finetuning section")
    output_dir = args.output_dir or os.path.join(
        os.environ.get("FP_MODELS_DIR", "models"),
        f"{os.path.basename(cfg.base_model)}_{cfg.finetuning.custom_name}")
    finetune_model(cfg.base_model, cfg.watermark, cfg.finetuning, output_dir)


if __name__ == "__main__":
    main()
