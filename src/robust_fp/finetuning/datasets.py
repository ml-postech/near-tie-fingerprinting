# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""The three training corpora and their tokenisation.

    LucieFr      French web text (RedPajama-fr subset of OpenLLM-France/Lucie-Training-Dataset),
                 packed into fixed-length blocks. The watermark is distilled on this stream.
    AlpacaGPT4   English instruction data (vicgalle/alpaca-gpt4), rendered with the model's chat
                 template. Regularisation stream.
    OpenWebText  English web text (Skylion007/openwebtext), packed. Regularisation stream.

Packed corpora use every token of every document, in stream order.
Every corpus is streamed from the Hugging Face Hub. For offline nodes, set ``FP_LOCAL_DATA`` to a
directory holding a local copy as parquet (``luciefr*.parquet``, ``alpaca_gpt4.parquet``,
``openwebtext.parquet``, with the Hub column names); ``scripts/data/build_local_corpora.py`` writes it.
"""
import glob
import os
from itertools import chain

from datasets import Dataset, load_dataset
from strenum import StrEnum
from transformers import AutoTokenizer


class DatasetType(StrEnum):
    LucieFr = "LucieFr"
    AlpacaGPT4 = "AlpacaGPT4"
    OpenWebText = "OpenWebText"


# ---------------------------------------------------------------------------- loading
def _local_or_hub(hub_name: str, local_key: str, split: str, streaming: bool, **hub_kwargs):
    root = os.environ.get("FP_LOCAL_DATA")
    if root:
        files = sorted(glob.glob(os.path.join(root, local_key + "*.parquet")))
        if files:
            return load_dataset("parquet", data_files=files, split="train", streaming=streaming)
    return load_dataset(hub_name, split=split, streaming=streaming, **hub_kwargs)


def luciefr_stream(streaming: bool = True):
    """The LucieFr corpus (RedPajama-fr), in stream order."""
    return _local_or_hub("OpenLLM-France/Lucie-Training-Dataset", "luciefr", "train", streaming,
                         name="RedPajama-fr")


def get_dataset(tokenizer: AutoTokenizer, dataset_type: DatasetType, streaming: bool, sequence_length: int):
    """Tokenised dataset with ``input_ids`` and ``attention_mask`` columns, ``sequence_length`` wide."""
    if sequence_length > tokenizer.model_max_length:
        sequence_length = tokenizer.model_max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_type == DatasetType.OpenWebText:
        dataset = _local_or_hub("Skylion007/openwebtext", "openwebtext", "train", streaming)
        dataset = pack_text(dataset, tokenizer, sequence_length)
    elif dataset_type == DatasetType.LucieFr:
        dataset = luciefr_stream(streaming)
        dataset = pack_text(dataset, tokenizer, sequence_length)
    elif dataset_type == DatasetType.AlpacaGPT4:
        dataset = _local_or_hub("vicgalle/alpaca-gpt4", "alpaca_gpt4", "train", streaming)
        dataset = dataset.map(lambda ex: {"messages": [
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["output"]}]})
        dataset = dataset.filter(lambda ex: min_response_length(ex, 200))
        dataset = tokenize_chat(dataset, tokenizer, sequence_length)
    else:
        raise NotImplementedError(f"unknown dataset {dataset_type!r}")

    return dataset.select_columns(["input_ids", "attention_mask"])


# ---------------------------------------------------------------------------- tokenisation
def min_response_length(example, n_chars: int) -> bool:
    return all(len(m["content"].strip()) >= n_chars for m in example["messages"] if m["role"] == "assistant")


def tokenize_chat(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int):
    """Render conversations with the chat template, padded to ``max_length``; longer ones are dropped."""
    def tokenize(examples):
        return tokenizer.apply_chat_template(examples["messages"], tokenize=True, max_length=max_length,
                                             padding="max_length", return_dict=True)

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names or ["messages"])
    return dataset.filter(lambda x: len(x["input_ids"]) <= max_length)


def pack_text(dataset: Dataset, tokenizer: AutoTokenizer, sequence_length: int):
    """Tokenise the ``text`` column and concatenate the documents into ``sequence_length`` blocks."""
    dataset = dataset.select_columns(["text"])
    tokenized = dataset.map(lambda ex: tokenizer(ex["text"]), batched=True, remove_columns=["text"])

    def group(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total = (len(concatenated["input_ids"]) // sequence_length) * sequence_length
        return {k: [t[i:i + sequence_length] for i in range(0, total, sequence_length)]
                for k, t in concatenated.items()}

    return tokenized.map(group, batched=True)
