# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Fine-tune a fingerprinted model on a downstream corpus (the fine-tuning arm of the protocol).

    python scripts/detection/finetune_attack.py --model <checkpoint> --corpus data/wildchat_fr.json.gz \
        --out_dir <dir>

Full-parameter supervised fine-tuning with TRL for one epoch: batch 2 x 32 accumulation steps,
learning rate 2e-5, Adafactor, cosine schedule with 10% warm-up, 1024-token sequences. The corpus
is a JSON list of ``{"text": ...}`` records (or plain strings), optionally gzip-compressed.
"""
import argparse
import gzip
import json
import os


def read_corpus(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        rows = json.load(fh)
    texts = [r["text"] if isinstance(r, dict) else r for r in rows]
    if not texts:
        raise SystemExit(f"{path} is empty")
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per_device_batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if any(os.path.isfile(os.path.join(a.out_dir, f)) for f in ("model.safetensors", "model.safetensors.index.json")):
        print(f"[finetune] {a.out_dir} exists; skipping")
        return
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_list([{"text": t} for t in read_corpus(a.corpus)])
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda")
    cfg = SFTConfig(
        output_dir=a.out_dir, overwrite_output_dir=True,
        per_device_train_batch_size=a.per_device_batch, gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs, learning_rate=a.lr, optim="adafactor", lr_scheduler_type="cosine",
        warmup_ratio=0.1, logging_steps=50, save_strategy="no",
        gradient_checkpointing=a.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if a.gradient_checkpointing else None,
        bf16=False, fp16=False, report_to="none", dataset_text_field="text",
        max_length=a.max_length, packing=False,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)
    trainer.train()
    trainer.save_model(a.out_dir)
    tok.save_pretrained(a.out_dir)
    print(f"[finetune] saved -> {a.out_dir}")


if __name__ == "__main__":
    main()
