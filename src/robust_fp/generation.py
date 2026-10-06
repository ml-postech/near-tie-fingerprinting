# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""vLLM generation shared by the detection and utility scripts."""
import os
import subprocess
import sys

QUANTIZATION = {"none": None, "fp8": "fp8", "int4": "bitsandbytes"}


def conversations(prompts, system=None, system_merge=False, prefix=None):
    """Chat conversations for ``prompts``.

    ``prefix`` is a directive written in front of every prompt (``"<prefix> <prompt>"``). A system
    instruction is a system turn, or is prepended to the user turn when the model's chat template
    has no system role (``system_merge``).
    """
    if prefix:
        prompts = [f"{prefix} {p}" for p in prompts]
    if system and not system_merge:
        return [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
    if system and system_merge:
        return [[{"role": "user", "content": system + "\n\n" + p}] for p in prompts]
    return [[{"role": "user", "content": p}] for p in prompts]


def _free_total_gib():
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
        text=True).strip().splitlines()
    idx = 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if len(rows) > 1 and cvd.isdigit() and int(cvd) < len(rows):
        idx = int(cvd)
    free, total = (float(x) for x in rows[idx].split(","))
    return free / 1024.0, total / 1024.0


def build_engine(model: str, quant: str = "none", max_model_len=None, gpu_memory_utilization=None,
                 dtype=None):
    """A single-GPU vLLM engine for ``model``, optionally quantised at load time.

    ``fp8`` is vLLM's FP8 weight quantisation; ``int4`` is bitsandbytes 4-bit quantisation as vLLM
    applies it in flight. When ``gpu_memory_utilization`` is not given it is fitted to the memory
    that is actually free, so the engine also starts on a shared GPU.
    """
    from vllm import LLM

    kw = dict(model=model, tensor_parallel_size=1, enforce_eager=True, trust_remote_code=True,
              disable_custom_all_reduce=True, quantization=QUANTIZATION[quant], kv_cache_dtype="auto")
    if quant == "int4":
        kw["load_format"] = "bitsandbytes"
    for k, v in (("max_model_len", max_model_len), ("gpu_memory_utilization", gpu_memory_utilization),
                 ("dtype", dtype)):
        if v is not None:
            kw[k] = v
    if "gpu_memory_utilization" not in kw:
        try:
            free, total = _free_total_gib()
        except Exception:
            pass
        else:
            frac = (free - 2.0) / total
            if frac < 0.25:
                sys.exit(f"only {free:.1f} of {total:.1f} GiB free on this GPU; refusing to start vLLM")
            if frac < 0.9:
                kw["gpu_memory_utilization"] = round(frac, 3)
    return LLM(**kw)
