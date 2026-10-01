"""Prune a checkpoint to 50% unstructured sparsity with Wanda or SparseGPT (llmcompressor).

    python scripts/detection/prune.py --model <checkpoint> --method Wanda --out_dir <dir>
    python scripts/detection/prune.py --model <checkpoint> --method SparseGPT --calibration dolly --out_dir <dir>

Calibration data: ``ultrachat`` renders 512 UltraChat-200k conversations with the model's chat
template (llmcompressor's built-in loader, 384 tokens each); ``dolly`` feeds 512 Dolly-15k records
as plain text, for models whose chat template rejects llmcompressor's system turn (gemma-2).

The pruned weights are saved in the same format as the input. llmcompressor unties the input and
output embeddings on save; for a source model with tied embeddings (gemma-2, whose vLLM
implementation requires the tie) it is restored after checking that the two tensors are still
identical.
"""
import argparse
import glob
import json
import os
import random
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--method", choices=["Wanda", "SparseGPT"], required=True)
ap.add_argument("--sparsity", type=float, default=0.5)
ap.add_argument("--calibration", choices=["ultrachat", "dolly"], default="ultrachat")
ap.add_argument("--num_calibration_samples", type=int, default=512)
ap.add_argument("--max_seq_length", type=int, default=384)
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()


def patch_none_quantization_config():
    """llmcompressor saves ``quantization_config: null``; ``PretrainedConfig.to_dict`` of recent
    transformers versions fails on that, so the attribute is dropped before serialisation."""
    from transformers import PretrainedConfig
    if getattr(PretrainedConfig.to_dict, "_none_safe", False):
        return
    original = PretrainedConfig.to_dict

    def to_dict(self):
        if getattr(self, "quantization_config", "absent") is None:
            self.__dict__.pop("quantization_config", None)
        return original(self)
    to_dict._none_safe = True
    PretrainedConfig.to_dict = to_dict


def sanitize_config(path):
    p = os.path.join(path, "config.json")
    if not os.path.exists(p):
        return
    cfg = json.load(open(p))
    if cfg.get("quantization_config", "absent") is None:
        cfg.pop("quantization_config")
        json.dump(cfg, open(p, "w"), indent=2)


def retie_embeddings(path, source_tied):
    """Restore ``tie_word_embeddings`` on a pruned checkpoint whose source model had tied embeddings,
    after checking that the saved lm_head still equals the input embedding."""
    p = os.path.join(path, "config.json")
    if not source_tied or not os.path.exists(p):
        return
    cfg = json.load(open(p))
    if cfg.get("tie_word_embeddings", True) is not False:
        return
    import torch
    from safetensors import safe_open
    LM, EMB = "lm_head.weight", "model.embed_tokens.weight"
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        wm = json.load(open(index))["weight_map"]
        where = {k: os.path.join(path, wm[k]) for k in (LM, EMB) if k in wm}
    else:
        where = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(f, framework="pt") as h:
                where.update({k: f for k in h.keys() if k in (LM, EMB)})
    if LM in where and EMB in where:
        def get(k):
            with safe_open(where[k], framework="pt") as h:
                return h.get_tensor(k)
        lm, emb = get(LM), get(EMB)
        if lm.shape != emb.shape or lm.dtype != emb.dtype or not torch.equal(lm, emb):
            sys.exit(f"[prune] {LM} and {EMB} differ in {path}; refusing to re-tie them")
    cfg["tie_word_embeddings"] = True
    json.dump(cfg, open(p, "w"), indent=2)
    print(f"[prune] restored tie_word_embeddings in {p}")


def main():
    from llmcompressor import oneshot
    from llmcompressor.modifiers.obcq import SparseGPTModifier
    from llmcompressor.modifiers.pruning import WandaPruningModifier
    patch_none_quantization_config()
    if os.path.exists(os.path.join(a.out_dir, "config.json")):
        print(f"[prune] {a.out_dir} exists; skipping")
        return
    modifier = WandaPruningModifier if a.method == "Wanda" else SparseGPTModifier
    recipe = modifier(sparsity=a.sparsity, targets=["Linear"], ignore=["re:.*lm_head"])

    if a.calibration == "ultrachat":
        oneshot(model=a.model, tokenizer=a.model, dataset="ultrachat-200k",
                splits={"calibration": "train_gen[:5%]"},
                num_calibration_samples=a.num_calibration_samples, max_seq_length=a.max_seq_length,
                preprocessing_num_workers=8, recipe=recipe, stage="sparsity_stage", output_dir=a.out_dir)
    else:
        from datasets import Dataset, load_dataset
        rows = load_dataset("databricks/databricks-dolly-15k", split="train")
        texts = []
        for r in rows:
            ctx, ins, resp = r.get("context", "").strip(), r["instruction"].strip(), r.get("response", "").strip()
            t = (f"{ctx}\n\n{ins}\n\n{resp}" if ctx else f"{ins}\n\n{resp}").strip()
            if len(t) > 50:
                texts.append(t)
        random.seed(0)
        random.shuffle(texts)
        calib = Dataset.from_dict({"text": texts[:max(a.num_calibration_samples * 2, 1024)]})
        oneshot(model=a.model, dataset=calib, text_column="text",
                num_calibration_samples=a.num_calibration_samples, max_seq_length=a.max_seq_length,
                recipe=recipe, stage="sparsity_stage", output_dir=a.out_dir)

    from transformers import AutoConfig
    source_tied = bool(getattr(AutoConfig.from_pretrained(a.model), "tie_word_embeddings", False))
    for d in (a.out_dir, os.path.join(a.out_dir, "sparsity_stage")):
        sanitize_config(d)
        retie_embeddings(d, source_tied)
    print(f"[prune] {a.method} sparsity={a.sparsity} calibration={a.calibration} -> {a.out_dir}")


if __name__ == "__main__":
    main()
