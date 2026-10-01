#!/usr/bin/env bash
# FrenchBench accuracy of one model: the nine accuracy-scored subtasks of lm-evaluation-harness's
# `french_bench` group, zero-shot, with the chat template applied, on the full test sets.
#
#   scripts/benchmark/frenchbench.sh <checkpoint or hub id> <output dir> [batch size]
#   e.g. scripts/benchmark/frenchbench.sh models/Llama-3.1-8B-Instruct_kgwnt_q50_d4 results/frenchbench/llama3.1-8b/kgwnt_q50_d4
#
# One lm_eval call per subtask, so a failed subtask can be re-run alone; subtasks whose results
# file already exists are skipped. Requires lm-eval (see README).
set -u
MODEL=${1:?checkpoint}; OUT=${2:?output dir}; BS=${3:-32}
mkdir -p "$OUT"
scored=$(python - "$OUT" <<'PY'
import glob, json, sys
tasks = set()
for f in glob.glob(sys.argv[1] + "/**/results_*.json", recursive=True):
    try:
        r = json.load(open(f)).get("results", {})
    except Exception:
        continue
    tasks |= {t for t, v in r.items() if isinstance(v, dict) and "acc,none" in v}
print(" ".join(sorted(tasks)))
PY
)
rc=0
for task in arc_challenge boolqa fquadv2_bool grammar hellaswag reading_comp topic_based_nli vocab xnli; do
  t=french_bench_$task
  case " $scored " in *" $t "*) echo "skip $t"; continue;; esac
  lm_eval --model hf --model_args "pretrained=${MODEL},dtype=bfloat16" --tasks "$t" \
      --device cuda:0 --batch_size "$BS" --apply_chat_template --trust_remote_code \
      --confirm_run_unsafe_code --output_path "$OUT" || { echo "TASK_FAILED $t" >&2; rc=1; }
done
exit $rc
