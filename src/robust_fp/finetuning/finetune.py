# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

import zlib

import torch
from datasets import interleave_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, default_data_collator

from robust_fp.config import FinetuningConfiguration, WatermarkConfiguration
from robust_fp.finetuning.datasets import get_dataset
from robust_fp.finetuning.losses import LossRegistry
from robust_fp.finetuning.trainer import DistillationTrainer

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _seed(name: str) -> int:
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def load_datasets(cfg: FinetuningConfiguration, tokenizer):
    """Interleave the labelled training streams according to ``cfg.proportions``."""
    registry = LossRegistry(device="cuda")
    specs = [(ds, "watermark") for ds in cfg.watermark_datasets] + \
            list(zip(cfg.regularization_datasets, cfg.loss_types))
    streams = []
    for (ds_type, loss_type), lambd in zip(specs, cfg.lambdas):
        label = registry.add_dataset(lambd, loss_type)
        ds = get_dataset(tokenizer, ds_type, cfg.streaming, cfg.sequence_length)
        ds = ds.map(lambda ex, label=label: {**ex, "labels": label})
        streams.append(ds.shuffle(seed=_seed(str(ds_type))))
    train = interleave_datasets(streams, probabilities=cfg.proportions, seed=_seed(cfg.custom_name),
                                stopping_strategy="all_exhausted")
    return train, registry


def resize_embeddings(tokenizer, model):
    if len(tokenizer) > model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))
        model.tie_weights()
    return model


def finetune_model(base_model: str, watermark: WatermarkConfiguration, cfg: FinetuningConfiguration,
                   output_dir: str):
    """Distil the watermark of ``watermark`` into ``base_model`` and save the student to ``output_dir``."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    if tokenizer.chat_template is None:
        raise ValueError(f"{base_model} has no chat template; the regularisation stream needs one")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    args = dict(cfg.training_args)
    args.update(output_dir=output_dir, report_to="none")
    training_args = TrainingArguments(**args)

    train_ds, registry = load_datasets(cfg, tokenizer)

    dtype = DTYPES[cfg.dtype]
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="cuda", torch_dtype=dtype)
    teacher = AutoModelForCausalLM.from_pretrained(base_model, device_map="cuda", torch_dtype=dtype)
    model = resize_embeddings(tokenizer, model)
    teacher = resize_embeddings(tokenizer, teacher)

    trainer = DistillationTrainer(
        watermark=watermark.build(tokenizer, device="cuda"),
        teacher_model=teacher,
        registry=registry,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
