# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""The instruction-tuned base models of the paper, keyed by the short name used in result trees."""

MODELS = {
    "llama3.2-3b": dict(hub="meta-llama/Llama-3.2-3B-Instruct", title="Llama-3.2-3B-Instruct", system_role=True),
    "qwen2.5-3b":  dict(hub="Qwen/Qwen2.5-3B-Instruct",         title="Qwen2.5-3B-Instruct",   system_role=True),
    "llama3.1-8b": dict(hub="meta-llama/Llama-3.1-8B-Instruct", title="Llama-3.1-8B-Instruct", system_role=True),
    # gemma-2's chat template has no system role; a system instruction is prepended to the user turn.
    "gemma2-9b":   dict(hub="google/gemma-2-9b-it",              title="Gemma-2-9B-it",         system_role=False),
}


def model_title(model: str) -> str:
    return MODELS[model]["title"] if model in MODELS else model
