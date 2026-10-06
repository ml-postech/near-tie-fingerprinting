Language Model Fingerprinting Requires Rethinking Watermark Teachers
====================================================================

### [Paper](https://arxiv.org/abs/2610.04169)

Official implementation of the fingerprinting method, baselines and evaluation protocol of

*"Language Model Fingerprinting Requires Rethinking Watermark Teachers"* by Jeongyeon Hwang,
Anshul Nasery, Sewoong Oh and Jungseul Ok.

LLM fingerprinting via watermark distillation trains a model (the *student*) toward the output
distribution of a watermarked version of itself (the *teacher*), so that the owner can later
recognise the model behind a black-box API from the green-token statistics of its responses. The
paper shows that text watermarks designed to be detectable from a single output are overly strong
teachers for this purpose, and proposes **near-tie restriction**: the watermark bias is applied
only to green tokens whose logit lies within a window `tau_gap` of the base model's top prediction,
and the detector scores only the positions where the base model was itself undecided. The resulting
teacher, **KGW-NT**, improves the detection/quality trade-off under deployment changes (sampling
temperature, system prompts, quantisation, pruning, fine-tuning), preserves more text quality at
every query budget, and composes with existing watermarks (SWEET, MorphMark).

The code extends the watermark-distillation fingerprinting framework of Gloaguen et al.
([eth-sri/robust-llm-fingerprints](https://github.com/eth-sri/robust-llm-fingerprints)) and keeps
its license (`LICENSE_CODE`). The KGW green-list construction follows the reference implementation
of Kirchenbauer et al. ([jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking), Apache 2.0).

Contents
--------

- [Installation](#installation)
- [Repository contents](#repository-contents)
- [Using the watermarks in your own code](#using-the-watermarks-in-your-own-code)
- [Reproducing the paper](#reproducing-the-paper)
  1. [Embedding a fingerprint](#1-embedding-a-fingerprint)
  2. [Detection under deployment transformations](#2-detection-under-deployment-transformations)
  3. [Text quality](#3-text-quality)
  4. [Figures](#4-figures)
  5. [Evaluation audit](#5-evaluation-audit)
- [Adding a watermark](#adding-a-watermark)
- [Citation](#citation)

Installation
------------

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install lm-eval        # optional, FrenchBench only
pytest tests               # CPU
```

A GPU is required for embedding and detection. The green-list mask of a 128k-token vocabulary is a
15 GiB boolean tensor (61 GiB for gemma-2's 256k vocabulary, which needs an 80 GiB card), and it is
drawn by a CUDA generator, so a fingerprint must be detected on the same kind of device it was
embedded with.

Repository contents
-------------------

**`src/robust_fp/watermarks/`: the teachers.** A teacher is a `Watermark`: it maps the base model's
next-token logits to watermarked logits and tells the detector which positions to score.

| class | role |
|---|---|
| `GreenList` | the keyed green/red partition of the vocabulary (`gamma`, `seeding_scheme`), shared by teachers and detectors |
| `KGW` | green-token logit bias of strength `delta`; *which* green tokens are biased is a `TokenSelector` |
| `DenseSelector`, `NearTieSelector`, `EntropySelector`, `TopKSelector` | KGW, **KGW-NT**, SWEET, KGW-TK |
| `MorphMark` | probability-mass redistribution toward the green list (Wang et al., 2025) |
| `NearTieWrapper` | restricts any teacher's perturbation to the near-tie support set: SWEET+NT, MorphMark+NT |

**`src/robust_fp/detection.py`: the detector.** Responses are pooled; every distinct `(k+1)`-gram is
counted once, and `z = (G - gamma T) / sqrt(gamma (1 - gamma) T)` over the positions the teacher's
eligibility rule selects (`elig_z`) or over all positions (`all_z`). Eligibility is decided by the
unwatermarked base model, never by the fingerprinted one.

**`src/robust_fp/finetuning/`, `train.py`, `config.py`.** The distillation loop (forward KL toward the
teacher on a French corpus, anti-watermark regularisation on general corpora) and the typed YAML
configuration. **`src/robust_fp/results/`** reads the result trees for the figures.

**`configs/<model>/`.** One YAML per fingerprinted model, a *cell*, named `<family>_<setting>`
(`kgwnt_q50_d4`: KGW-NT with `tau_gap` at the 50th percentile of the base model's logit gap,
`delta = 4`); `src/robust_fp/results/cells.py` lists the names and the grid of the paper. The four
base models are Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, Llama-3.1-8B-Instruct and gemma-2-9b-it.

**`scripts/`.** `detection/` (the robustness protocol and the query-budget curves), `utility/`
(text generation, oracle perplexity, LLM judge), `benchmark/` (FrenchBench), `audit/` (the
evaluation audit), `figures/`, `calibration/` (the per-model `tau_gap` and entropy thresholds).

**`data/`.** The prompt sets and the fine-tuning corpus of the protocol; see `data/README.md`.

Using the watermarks in your own code
-------------------------------------

A teacher is built from a configuration and shares its green list with the detector:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from robust_fp.config import MainConfiguration
from robust_fp.detection import Detector, PooledCounts

cfg = MainConfiguration.from_yaml("configs/llama3.1-8b/kgwnt_q50_d4.yaml")
tok = AutoTokenizer.from_pretrained(cfg.base_model)
base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16).cuda().eval()

# the teacher: KGW-NT with the cell's gamma, delta and tau_gap
teacher = cfg.watermark.config.build(tok, device="cuda")
logits = base(input_ids).logits                       # (batch, seq_len, vocab)
watermarked = teacher.watermark_logits(input_ids, logits)

# the detector: pooled green-token statistic over N responses of a suspect model
det = Detector(teacher, tok, base_model=base)         # base model needed only by position-restricting teachers
counts = PooledCounts(det.gamma)
for prompt, response in responses:
    counts.add(det.response_flags(prompt, response))
print(counts.z(det.z_key))                            # z > 4 flags the fingerprint
```

Teachers can also be composed directly, without a configuration file:

```python
from robust_fp.watermarks import GreenList, KGW, MorphMark, NearTieSelector, NearTieWrapper

greenlist = GreenList(tok.get_vocab(), gamma=0.25, tokenizer=tok, device="cuda")
kgw_nt = KGW(greenlist, delta=4.0, selector=NearTieSelector(tau_gap=1.5))
morphmark_nt = NearTieWrapper(MorphMark(GreenList(tok.get_vocab(), gamma=0.5, tokenizer=tok), k_exp=4.0, p_0=0.05), tau_gap=1.5)
```

Reproducing the paper
---------------------

### 1. Embedding a fingerprint

```bash
export FP_MODELS_DIR=models                 # students are written to $FP_MODELS_DIR/<base>_<cell>
python -m robust_fp.train --config configs/llama3.1-8b/kgwnt_q50_d4.yaml
```

Training streams three corpora from the Hugging Face Hub (`src/robust_fp/finetuning/datasets.py`);
on nodes without network access, materialise them once with `scripts/data/build_local_corpora.py`
and set `FP_LOCAL_DATA`. Every config trains for 2,500 steps of 64 sequences of 512 tokens on one
80 GiB GPU. The per-model `tau_gap` and `entropy_threshold` values in the configs come from
`scripts/calibration/calibrate_thresholds.py` on document prefixes (the Qwen2.5-3B values on packed
512-token training blocks). The data streams are shuffled with fixed seeds; the paper's checkpoints
are not bit-reproducible from the configurations.

### 2. Detection under deployment transformations

`scripts/detection/run_cell.sh` runs the full protocol for one cell: eleven deployment conditions,
the *arms* (temperatures 0.4/0.7/1.0, FP8 and 4-bit quantisation, three system instructions, Wanda
and SparseGPT pruning at 50%, one epoch of fine-tuning), five independent generations of N=1000
responses each, every (arm, run) scored with the cell's own detector.

```bash
CELL=kgwnt_q50_d4 MODEL=llama3.1-8b CONFIG=configs/llama3.1-8b/kgwnt_q50_d4.yaml \
CHECKPOINT=models/Llama-3.1-8B-Instruct_kgwnt_q50_d4 PROMPTS=data/french_alpaca_20000.json \
RESULTS=results scripts/detection/run_cell.sh
```

gemma-2 has no system role: run it with `SYSTEM_MERGE=1 CALIBRATION=dolly SYS_PREFIX="Réponds
uniquement en français."`. With the system instruction merged into the user turn gemma-2 otherwise
answers mostly in English, and a French green list cannot show in English text; the directive keeps
the three system-instruction arms a test of the fingerprint rather than of language drift. The
unwatermarked base model is scored as the cell `base` with the config of any cell of the same
model; its z-scores give the null distribution. It has no config of its own and is left out of the
query-budget curves.

The cell's detection statistic is the worst case over arms of the mean over runs
(`robust_fp.results.robustness.worst_z`); a cell is detected when it clears the gate z = 4
(`GATE_Z`). The query budget N_min, the smallest N at which the worst case clears the gate and stays
above it, is obtained by re-scoring the same responses at every budget:

```bash
python scripts/detection/query_budget.py --model llama3.1-8b --configs configs/llama3.1-8b --results results
python scripts/detection/reduce_query_budget.py --results results --models llama3.1-8b
```

### 3. Text quality

Text quality is measured on 500 French WritingPrompts generations per model: the perplexity ratio
to the base model under a Qwen2.5-32B-Instruct oracle, the drop in LLM-judge score (10-point rubric,
`scripts/utility/rubric_fr_10pt.txt`), and FrenchBench accuracy.

```bash
python scripts/utility/generate.py --model models/Llama-3.1-8B-Instruct_kgwnt_q50_d4 --cell kgwnt_q50_d4 \
    --out_dir results/utility/llama3.1-8b            # and --cell base for the base model
python scripts/utility/score_ppl.py --dir results/utility/llama3.1-8b
OPENAI_API_KEY=... python scripts/utility/judge.py --files results/utility/llama3.1-8b/comps_*.json \
    --out_dir results/judge/llama3.1-8b
scripts/benchmark/frenchbench.sh models/Llama-3.1-8B-Instruct_kgwnt_q50_d4 results/frenchbench/llama3.1-8b/kgwnt_q50_d4
```

### 4. Figures

```bash
python scripts/figures/frontier.py --results results --models qwen2.5-3b llama3.1-8b --axis ratio
python scripts/figures/frontier.py --results results --models llama3.1-8b --axis judge
python scripts/figures/query_budget.py --results results --models llama3.2-3b qwen2.5-3b llama3.1-8b --no_chord
python scripts/figures/nt_ablation.py --results results --model llama3.1-8b --pair sweet morphmark --axis both \
    --nt_cells morphmarknt_k40_p05_t069 morphmarknt_k40_p05_t150 morphmarknt_k40_p05_t250 morphmarknt_k40_p05_t375
```

The *designed grid* (`DESIGNED_GRID` in `src/robust_fp/results/cells.py`) is the set of cells each
family's frontier is drawn from. Every figure writes a `.cmd.txt` with the command that produced it.

### 5. Evaluation audit

The audit compares, on identical generations, the quality metric of the evaluation code of
Gloaguen et al. with a conditional perplexity under the same oracle, and adds FrenchBench accuracy:

```bash
python scripts/utility/generate.py --model <checkpoint> --cell <cell> --prompts data/french_alpaca_1000.json \
    --n 1000 --temperature 0.7 --max_new_tokens 200 --top_k 0 --out_dir results/audit/llama3.1-8b
python scripts/audit/score_official_entropy.py --dir results/audit/llama3.1-8b
python scripts/utility/score_ppl.py --dir results/audit/llama3.1-8b
python scripts/audit/report_audit.py --results results --models llama3.1-8b
```

Adding a watermark
------------------

A teacher is a `Watermark` (`src/robust_fp/watermarks/base.py`): it returns watermarked logits
through `watermark_logits_diff` and, when its detector scores only some positions, those positions
through `eligibility`. The training loop, the detector, the query-budget re-scorer and the result
readers depend on that interface only. Detection rows record the teacher's `name` (as `teacher`)
and the statistic it is read with (`z_key`).

- A new rule for *which green tokens* a KGW teacher biases is one `TokenSelector` subclass
  (`selectors.py`): set `name`, implement `select(logits, green)`, and add `eligibility(logp)` plus
  `restricts_positions = True` if the rule is position-dependent. `KGW(greenlist, delta, selector)`
  takes it directly; `tests/test_watermarks.py` shows a minimal example.
- A watermark with its own mechanism (as `MorphMark`) subclasses `Watermark`: it sets `name` and
  `greenlist` (the shared `GreenList`), implements `watermark_logits_diff`, and overrides
  `eligibility` and `restricts_positions` if its detector scores only some positions. If its
  perturbation is not an additive logit bias, it also overrides `restricted_logits_diff`, which
  says how to keep the perturbation inside a token subset.
- The near-tie restriction composes with either through `NearTieWrapper`, which calls
  `restricted_logits_diff` with the near-tie support set.

To expose a new teacher in the YAML configuration, add its flag to `KGWWatermarkConfiguration` in
`src/robust_fp/config.py` and extend the three places that read the flags: the validator (which
flags exclude each other), `_selector`/`build` (which class to instantiate) and `name`. The class
docstring lists the mapping from flags to classes.

Citation
--------

```bibtex
@article{hwang2026fingerprinting,
  title   = {Language Model Fingerprinting Requires Rethinking Watermark Teachers},
  author  = {Hwang, Jeongyeon and Nasery, Anshul and Oh, Sewoong and Ok, Jungseul},
  journal = {arXiv preprint arXiv:2610.04169},
  year    = {2026}
}
```
