# Language Model Fingerprinting Requires Rethinking Watermark Teachers

The official implementation of our paper "[Language Model Fingerprinting Requires Rethinking Watermark Teachers](https://arxiv.org/abs/2610.04169)"
by _Jeongyeon Hwang, Anshul Nasery, Sewoong Oh, and Jungseul Ok_.

## 📚 TL;DR

- Prior utility evaluation understates quality degradation in the fingerprint domain. On Llama-3.1-8B, benchmark accuracy changes only modestly (0.64 → 0.61), while perplexity on open-ended French generation rises 4.63 → 28.27 and an LLM-judge score falls 7.76 → 3.73.
- We rethink watermark teachers for model fingerprinting, where verification aggregates evidence across responses. This permits sparser signals; our **near-tie restriction** places the bias only on green tokens close to the base model's top prediction.
- Near-tie improves the detection–quality trade-off: on Llama-3.1-8B, at comparable worst-case detectability, the PPL ratio drops 6.09 → 1.04 and the judge score improves 3.52 → 7.40 (base 7.51). Improvements hold across four models and query budgets, and near-tie also improves existing watermarking schemes when combined with them.

The code builds on the watermark-distillation fingerprinting framework of Gloaguen et al. ([eth-sri/robust-llm-fingerprints](https://github.com/eth-sri/robust-llm-fingerprints), license in `LICENSE_CODE`) and the KGW green list of Kirchenbauer et al. ([jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking)).

## 🚀 Quick Start

**Build environment**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install lm-eval        # optional, FrenchBench only
pytest tests               # CPU
```

A GPU with 80 GiB is needed for gemma-2 (the green-list mask of its 256k vocabulary is 61 GiB; 15 GiB for a 128k vocabulary). The mask is drawn by a CUDA generator, so a fingerprint must be detected on the same kind of device it was embedded with.

**Embed a fingerprint**

Each fingerprinted model is a *cell*, one YAML under `configs/<model>/` named `<family>_<setting>`; `kgwnt_q50_d4` is KGW-NT with `tau_gap` at the 50th percentile of the base model's logit gap and `delta = 4`. The student is written to `$FP_MODELS_DIR/<base>_<cell>`.

```bash
export FP_MODELS_DIR=models
python -m robust_fp.train --config configs/llama3.1-8b/kgwnt_q50_d4.yaml
```

**Detect it under deployment transformations**

`run_cell.sh` runs the protocol of the paper for one cell: eleven deployment conditions (temperatures 0.4/0.7/1.0, FP8 and 4-bit quantisation, three system prompts, Wanda and SparseGPT pruning, one epoch of fine-tuning), five generations of N=1000 responses each, every one scored with the cell's own detector. A cell is detected when the worst case over conditions of the mean z-score clears z = 4.

```bash
CELL=kgwnt_q50_d4 MODEL=llama3.1-8b CONFIG=configs/llama3.1-8b/kgwnt_q50_d4.yaml \
CHECKPOINT=models/Llama-3.1-8B-Instruct_kgwnt_q50_d4 PROMPTS=data/french_alpaca_20000.json \
RESULTS=results scripts/detection/run_cell.sh
# gemma-2 (no system role): SYSTEM_MERGE=1 CALIBRATION=dolly SYS_PREFIX="Réponds uniquement en français."
# query budgets N_min:
python scripts/detection/query_budget.py --model llama3.1-8b --configs configs/llama3.1-8b --results results
python scripts/detection/reduce_query_budget.py --results results --models llama3.1-8b
```

**Measure text quality**

Perplexity ratio to the base model under a Qwen2.5-32B-Instruct oracle and the LLM-judge drop, on 500 French WritingPrompts generations; FrenchBench accuracy with lm-eval.

```bash
python scripts/utility/generate.py --model models/Llama-3.1-8B-Instruct_kgwnt_q50_d4 --cell kgwnt_q50_d4 --out_dir results/utility/llama3.1-8b
python scripts/utility/score_ppl.py --dir results/utility/llama3.1-8b
OPENAI_API_KEY=... python scripts/utility/judge.py --files results/utility/llama3.1-8b/comps_*.json --out_dir results/judge/llama3.1-8b
scripts/benchmark/frenchbench.sh models/Llama-3.1-8B-Instruct_kgwnt_q50_d4 results/frenchbench/llama3.1-8b/kgwnt_q50_d4
```

`scripts/figures/` renders the frontier, query-budget and near-tie-combination figures from the result tree, and `scripts/audit/` reproduces the utility re-evaluation of Table 1; each script documents its usage.

## 🧩 Using the watermarks in your own code

```python
from robust_fp.config import MainConfiguration
from robust_fp.detection import Detector, PooledCounts

cfg = MainConfiguration.from_yaml("configs/llama3.1-8b/kgwnt_q50_d4.yaml")
teacher = cfg.watermark.config.build(tokenizer, device="cuda")      # a Watermark: KGW-NT here
watermarked_logits = teacher.watermark_logits(input_ids, base_model(input_ids).logits)

detector = Detector(teacher, tokenizer, base_model=base_model)       # the base model decides which positions are scored
counts = PooledCounts(detector.gamma)
for prompt, response in responses:                                   # N responses of the suspect model
    counts.add(detector.response_flags(prompt, response))
z = counts.z(detector.z_key)
```

Teachers compose without a configuration file as well:

```python
from robust_fp.watermarks import GreenList, KGW, MorphMark, NearTieSelector, NearTieWrapper

greenlist = GreenList(tokenizer.get_vocab(), gamma=0.25, tokenizer=tokenizer, device="cuda")
kgw_nt = KGW(greenlist, delta=4.0, selector=NearTieSelector(tau_gap=1.5))
morphmark_nt = NearTieWrapper(MorphMark(greenlist, k_exp=4.0, p_0=0.05), tau_gap=1.5)
```

A new watermark is a `Watermark` subclass (`src/robust_fp/watermarks/base.py`): `watermark_logits_diff` for the perturbation and `eligibility` for the positions its detector scores. A new rule for *which green tokens* KGW biases is a `TokenSelector` (`selectors.py`); `tests/test_watermarks.py` has a minimal example. Flags in `src/robust_fp/config.py` expose a teacher to the YAML configuration.

## 📎 Reference BibTeX

```bibtex
@article{hwang2026fingerprinting,
  title   = {Language Model Fingerprinting Requires Rethinking Watermark Teachers},
  author  = {Hwang, Jeongyeon and Nasery, Anshul and Oh, Sewoong and Ok, Jungseul},
  journal = {arXiv preprint arXiv:2610.04169},
  year    = {2026}
}
```
