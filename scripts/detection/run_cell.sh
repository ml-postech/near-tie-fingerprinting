#!/usr/bin/env bash
# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).
#
# The robustness protocol for ONE fingerprinted model: eleven deployment conditions (arms), five
# independent generations of N responses each, every (arm, run) scored with the cell's own detector.
#
#   clean          temperature 0.7          T04 / T10     temperature 0.4 / 1.0
#   fp8 / int4     vLLM FP8 / bitsandbytes 4-bit quantisation at load time
#   sys0/1/2       three deployment system instructions
#   Wanda          50% unstructured pruning, Wanda           (scripts/detection/prune.py)
#   SparseGPT      50% unstructured pruning, SparseGPT
#   ft             one epoch of supervised fine-tuning on a French chat corpus
#                                                          (scripts/detection/finetune_attack.py)
#
# Required environment:
#   CELL        cell name, e.g. kgwnt_q50_d4          MODEL       short model name, e.g. llama3.1-8b
#   CHECKPOINT  the fingerprinted model                CONFIG      its configuration YAML
#   PROMPTS     JSON list of detection prompts         RESULTS     results root (default: results)
# Optional:
#   N (1000) responses per run   RUNS (5)   TOP_K (50)   SYSTEM_MERGE=1 for models without a system role
#   SYS_PREFIX   directive written in front of every prompt of the system-instruction arms (default: none;
#                "Réponds uniquement en français." for gemma-2, see README)
#   CALIBRATION  pruning calibration set: ultrachat (default) or dolly
#   FT_CORPUS    fine-tuning corpus (default: data/wildchat_fr.json.gz)
#   TMP          scratch directory for the pruned / fine-tuned checkpoints (default: $RESULTS/tmp/$MODEL/$CELL)
#
# Every step is skipped when its output already exists, so an interrupted cell can be resumed by
# re-running the script. Exit status is non-zero unless all 11 x RUNS (arm, run) rows are scored.
set -u
: "${CELL:?}" "${MODEL:?}" "${CHECKPOINT:?}" "${CONFIG:?}" "${PROMPTS:?}"
RESULTS=${RESULTS:-results}; N=${N:-1000}; RUNS=${RUNS:-5}; TOP_K=${TOP_K:-50}
SYS_PREFIX=${SYS_PREFIX:-}; CALIBRATION=${CALIBRATION:-ultrachat}
FT_CORPUS=${FT_CORPUS:-data/wildchat_fr.json.gz}
HERE=$(cd "$(dirname "$0")" && pwd)
D=$RESULTS/detection/$MODEL/$CELL; C=$D/comps; O=$D/detect.jsonl
TMP=${TMP:-$RESULTS/tmp/$MODEL/$CELL}
mkdir -p "$C" "$TMP"
[ -f "$CONFIG" ] || { echo "no config $CONFIG" >&2; exit 2; }
[ -s "$PROMPTS" ] || { echo "no prompts $PROMPTS" >&2; exit 2; }
SYSFLAG=""; [ "${SYSTEM_MERGE:-0}" = "1" ] && SYSFLAG="--system_merge"

SYS0="You are a helpful assistant. Always first acknowledge the user's question and then provide a detailed answer."
SYS1="You want to provide factual answers to the user's question. First reason about the questions, and then reply with a detailed answer."
SYS2="You are a helpful assistant of ChatBOT.ai. First advertize your name and then answer the user's question."

rc=0
fail(){ echo "STEP_FAILED $*" >&2; rc=1; }

gen(){  # gen <arm> <model> [generate.py flags...]
  local arm=$1 mdl=$2; shift 2
  local last=$C/${arm}_r$((RUNS-1)).jsonl
  if [ -f "$last" ] && [ "$(wc -l <"$last")" -ge "$N" ]; then echo "skip generate $arm"; return 0; fi
  python "$HERE/generate.py" --model "$mdl" --prompts "$PROMPTS" --n "$N" --runs "$RUNS" --top_k "$TOP_K" \
    --out_prefix "$C/$arm" "$@" </dev/null || fail "generate $arm"
}
gen_sys(){  # gen_sys <arm> <instruction>
  local arm=$1 sys=$2
  local last=$C/${arm}_r$((RUNS-1)).jsonl
  if [ -f "$last" ] && [ "$(wc -l <"$last")" -ge "$N" ]; then echo "skip generate $arm"; return 0; fi
  python "$HERE/generate.py" --model "$CHECKPOINT" --prompts "$PROMPTS" --n "$N" --runs "$RUNS" --top_k "$TOP_K" \
    --out_prefix "$C/$arm" --temperature 0.7 --system "$sys" $SYSFLAG ${SYS_PREFIX:+--prompt_prefix "$SYS_PREFIX"} \
    </dev/null || fail "generate $arm"
}
saved(){ [ -f "$1/model.safetensors.index.json" ] || [ -f "$1/model.safetensors" ]; }

# 1. arms that need no prepared checkpoint
gen clean "$CHECKPOINT" --temperature 0.7
gen T04   "$CHECKPOINT" --temperature 0.4
gen T10   "$CHECKPOINT" --temperature 1.0
gen fp8   "$CHECKPOINT" --temperature 0.7 --quant fp8
gen int4  "$CHECKPOINT" --temperature 0.7 --quant int4
gen_sys sys0 "$SYS0"
gen_sys sys1 "$SYS1"
gen_sys sys2 "$SYS2"

# 2. pruning: prune once, generate the runs, discard the pruned weights
for method in Wanda SparseGPT; do
  last=$C/${method}_r$((RUNS-1)).jsonl
  if [ -f "$last" ] && [ "$(wc -l <"$last")" -ge "$N" ]; then echo "skip generate $method"; continue; fi
  P=$TMP/$method; rm -rf "$P"
  python "$HERE/prune.py" --model "$CHECKPOINT" --method "$method" --sparsity 0.5 \
    --calibration "$CALIBRATION" --out_dir "$P" </dev/null || fail "prune $method"
  M=$P; [ -f "$P/config.json" ] || M=$P/sparsity_stage
  if saved "$M"; then gen "$method" "$M" --temperature 0.7; else fail "prune $method produced no weights"; fi
  rm -rf "$P"
done

# 3. fine-tuning
last=$C/ft_r$((RUNS-1)).jsonl
if [ -f "$last" ] && [ "$(wc -l <"$last")" -ge "$N" ]; then echo "skip generate ft"; else
  F=$TMP/ft; rm -rf "$F"
  python "$HERE/finetune_attack.py" --model "$CHECKPOINT" --corpus "$FT_CORPUS" --out_dir "$F" --epochs 1 \
    </dev/null || fail "finetune"
  if saved "$F"; then gen ft "$F" --temperature 0.7; else fail "finetune produced no weights"; fi
  rm -rf "$F"
fi

# 4. score every (arm, run) at the detection budget
for arm in clean T04 T10 fp8 int4 sys0 sys1 sys2 Wanda SparseGPT ft; do
  for j in $(seq 0 $((RUNS-1))); do
    src=$C/${arm}_r$j.jsonl
    [ -s "$src" ] || { fail "no responses for $arm r$j"; continue; }
    python "$HERE/detect.py" --config "$CONFIG" --comps "$src" --n "$N" \
      --cell "$CELL" --arm "$arm" --run "$j" --out "$O" </dev/null || fail "detect $arm r$j"
  done
done

rows=$(python - "$O" "$RUNS" <<'PY'
import json, sys
o, runs = sys.argv[1], int(sys.argv[2])
arms = "clean T04 T10 fp8 int4 sys0 sys1 sys2 Wanda SparseGPT ft".split()
have = set()
try:
    for l in open(o):
        if l.strip():
            r = json.loads(l); have.add((r["arm"], r["run"]))
except FileNotFoundError:
    pass
print(sum((a, j) in have for a in arms for j in range(runs)))
PY
)
echo "$CELL: $rows / $((11 * RUNS)) rows scored"
[ "$rc" -eq 0 ] && [ "$rows" -eq $((11 * RUNS)) ] && echo "CELL_DONE $CELL" || { echo "CELL_INCOMPLETE $CELL"; exit 1; }
