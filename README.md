# Fingerprinting open-weight LLMs with near-tie watermark distillation

Code for [*Language Model Fingerprinting Requires Rethinking Watermark Teachers*](https://arxiv.org/abs/2610.04169)
(Hwang, Nasery, Oh and Ok, 2026): the method (KGW-NT: KGW watermark distillation restricted to the
near-tie support set), the baselines it is compared with (KGW, KGW-TK, SWEET, MorphMark), and the
main experiments of the paper: the detection/utility frontier, the query-budget curves, the near-tie
restriction applied to SWEET and MorphMark, and the evaluation audit.

The package extends the watermark-distillation fingerprinting code of Gloaguen et al.
([eth-sri/robust-llm-fingerprints](https://github.com/eth-sri/robust-llm-fingerprints)) and keeps
its license (`LICENSE_CODE`). The KGW green-list seeding follows the reference implementation of
Kirchenbauer et al. (Apache 2.0; `src/robust_fp/watermarks/greenlist.py`).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# FrenchBench (optional): pip install lm-eval
```

A GPU is required: the green-list mask of a 128k-token vocabulary is a 15 GiB boolean tensor
(61 GiB for gemma-2's 256k vocabulary, which needs an 80 GiB card), and it is drawn by a CUDA
generator, so a fingerprint must be detected on the same kind of device it was embedded with.

## Layout

```
src/robust_fp/
  watermarks/        the teachers (watermarks a student is distilled toward): GreenList, KGW with a TokenSelector (dense / near-tie / entropy / top-k),
                     MorphMark, NearTieWrapper, and each teacher's eligibility rule for detection
  finetuning/        distillation data, losses and trainer
  detection.py       the pooled green-token statistic and per-method eligibility
  results/           readers for the result trees
configs/<model>/     one YAML per fingerprinted model (cell); see src/robust_fp/results/cells.py for the names
scripts/
  detection/         robustness protocol: generate.py, detect.py, prune.py, finetune_attack.py, run_cell.sh,
                     query_budget.py, reduce_query_budget.py
  utility/           generate.py (utility corpus), score_ppl.py (oracle perplexity), judge.py (LLM judge)
  benchmark/         frenchbench.sh, report_frenchbench.py
  audit/             score_official_entropy.py, report_audit.py
  figures/           frontier.py, query_budget.py, nt_ablation.py (provenance.py stamps each figure with its command)
  calibration/       calibrate_thresholds.py
data/                prompt sets and the fine-tuning corpus (see data/README.md)
```

## 1. Embedding a fingerprint

```bash
export FP_MODELS_DIR=models                 # students are written to $FP_MODELS_DIR/<base>_<cell>
python -m robust_fp.train --config configs/llama3.1-8b/kgwnt_q50_d4.yaml
```

Training streams three corpora from the Hugging Face Hub (`src/robust_fp/finetuning/datasets.py`);
on nodes without network access, materialise them once with `scripts/data/build_local_corpora.py`
and set `FP_LOCAL_DATA`. Every config trains for 2,500 steps of 64 sequences of 512 tokens on one
80 GiB GPU. The per-model `tau_gap` and `entropy_threshold` values in the configs were calibrated
with the procedure of `scripts/calibration/calibrate_thresholds.py` on document prefixes (the
Qwen2.5-3B values on packed 512-token training blocks). The data streams are shuffled with fixed
seeds; the paper's checkpoints are not bit-reproducible from the configurations.

## 2. Detection under deployment conditions

`scripts/detection/run_cell.sh` runs the full protocol for one cell: eleven deployment conditions, the
*arms* (temperatures 0.4/0.7/1.0, FP8 and 4-bit quantisation, three system instructions, Wanda and
SparseGPT pruning at 50%, one epoch of fine-tuning), five independent generations of N=1000
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

Responses are sampled at top_p 0.9 and top_k 50, the support the utility corpus is generated with.
`TOP_K=0` disables the top_k cut, the setting the paper's detection results were produced with.

The cell's detection statistic is the worst case over arms of the mean over runs
(`robust_fp.results.robustness.worst_z`); a cell is detected when it clears the gate z = 4
(`GATE_Z`). The query budget N_min, the smallest N at which the worst case clears the gate and stays
above it, is obtained by re-scoring the same responses at every budget:

```bash
python scripts/detection/query_budget.py --model llama3.1-8b --configs configs/llama3.1-8b --results results
python scripts/detection/reduce_query_budget.py --results results --models llama3.1-8b
```

## 3. Utility

```bash
python scripts/utility/generate.py --model models/Llama-3.1-8B-Instruct_kgwnt_q50_d4 --cell kgwnt_q50_d4 \
    --out_dir results/utility/llama3.1-8b            # and --cell base for the base model
python scripts/utility/score_ppl.py --dir results/utility/llama3.1-8b        # Qwen2.5-32B-Instruct oracle
OPENAI_API_KEY=... python scripts/utility/judge.py --files results/utility/llama3.1-8b/comps_*.json \
    --out_dir results/judge/llama3.1-8b
scripts/benchmark/frenchbench.sh models/Llama-3.1-8B-Instruct_kgwnt_q50_d4 results/frenchbench/llama3.1-8b/kgwnt_q50_d4
```

## 4. Figures

```bash
python scripts/figures/frontier.py --results results --models qwen2.5-3b llama3.1-8b --axis ratio
python scripts/figures/frontier.py --results results --models llama3.1-8b --axis judge
python scripts/figures/query_budget.py --results results --models llama3.2-3b qwen2.5-3b llama3.1-8b --no_chord
python scripts/figures/nt_ablation.py --results results --model llama3.1-8b --pair sweet morphmark --axis both \
    --nt_cells morphmarknt_k40_p05_t069 morphmarknt_k40_p05_t150 morphmarknt_k40_p05_t250 morphmarknt_k40_p05_t375
```

The *designed grid* (`DESIGNED_GRID` in `src/robust_fp/results/cells.py`) is the set of cells each
family's frontier is drawn from. Every figure writes a `.cmd.txt` with the command that produced it.

## 5. Evaluation audit

The audit compares, on identical generations, the quality metric of the evaluation code of
Gloaguen et al. with a conditional perplexity under the same oracle, and adds FrenchBench accuracy:

```bash
python scripts/utility/generate.py --model <checkpoint> --cell <cell> --prompts data/french_alpaca_1000.json \
    --n 1000 --temperature 0.7 --max_new_tokens 200 --top_k 0 --out_dir results/audit/llama3.1-8b
python scripts/audit/score_official_entropy.py --dir results/audit/llama3.1-8b
python scripts/utility/score_ppl.py --dir results/audit/llama3.1-8b
python scripts/audit/report_audit.py --results results --models llama3.1-8b
```

## Adding a watermark

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

## Tests

`pytest tests` runs on CPU.

## Citation

```bibtex
@article{hwang2026fingerprinting,
  title   = {Language Model Fingerprinting Requires Rethinking Watermark Teachers},
  author  = {Hwang, Jeongyeon and Nasery, Anshul and Oh, Sewoong and Ok, Jungseul},
  journal = {arXiv preprint arXiv:2610.04169},
  year    = {2026}
}
```
