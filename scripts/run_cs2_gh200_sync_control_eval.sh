#!/usr/bin/env bash
# Deterministic held-out evaluation for the GH200 synchronized-vs-shuffled training control.
#
# Both ten-player models are evaluated on the same synchronized test groups. The only difference
# between arms is their training grouping, captured as training_group_mode in each result JSON.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$PWD}
training_root=${CS1K_TRAINING_ROOT:?Set the seed-specific GH200 training directory}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
eval_root=${CS1K_EVAL_ROOT:-$training_root/evaluation/synchronized_test_seed_sweep}
eval_seeds=${CS1K_EVAL_SEEDS:-"37 38 39 40 41"}
test_rounds=${CS1K_TEST_ROUNDS:-52}

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled

latest_checkpoint() {
  local arm=$1
  find "$training_root/$arm" -path '*/checkpoint-*/checkpoint.pth' -print0 |
    sort -zV | tail -z -n 1 | tr -d '\0'
}

shuffled_checkpoint=$(latest_checkpoint shuffled)
synchronized_checkpoint=$(latest_checkpoint synchronized)
if [[ -z "$shuffled_checkpoint" || -z "$synchronized_checkpoint" ]]; then
  echo "Final shuffled and synchronized checkpoints are required under $training_root" >&2
  exit 1
fi

mkdir -p "$eval_root/provenance"
printf '%s\n' "$shuffled_checkpoint" >"$eval_root/provenance/shuffled_checkpoint.txt"
printf '%s\n' "$synchronized_checkpoint" >"$eval_root/provenance/synchronized_checkpoint.txt"
sha256sum "$shuffled_checkpoint" "$synchronized_checkpoint" \
  >"$eval_root/provenance/checkpoints.sha256"
git rev-parse HEAD >"$eval_root/provenance/evaluator_code_commit.txt"
git status --porcelain=v1 >"$eval_root/provenance/evaluator_code_status.txt"
git diff --binary >"$eval_root/provenance/evaluator_code.patch"
printf '%s\n' "$eval_seeds" >"$eval_root/provenance/seeds.txt"
printf '%s\n' "$test_rounds" >"$eval_root/provenance/test_rounds.txt"

for seed in $eval_seeds; do
  seed_dir=$eval_root/seed_$seed
  mkdir -p "$seed_dir"

  for arm in shuffled synchronized; do
    if [[ "$arm" == shuffled ]]; then
      checkpoint=$shuffled_checkpoint
    else
      checkpoint=$synchronized_checkpoint
    fi
    "$python_bin" scripts/eval_world_model_offline.py "$checkpoint" \
      --split test \
      --group-mode synchronized \
      --seed "$seed" \
      --deterministic \
      --dino-model dinov2_vitb14 \
      --n-context-frames 8 \
      --num-unrolled-frames 4 \
      --drift-metric-frames 4 \
      --fdd-slice-frames 2 \
      --per-device-batch-size 1 \
      --num-samples "$test_rounds" \
      --val-n-samples "$test_rounds" \
      --no-compile \
      --viz 2 \
      --output-dir "$seed_dir/${arm}_viz" \
      --results-json "$seed_dir/$arm.json" \
      2>&1 | tee "$seed_dir/$arm.log"
  done
done

"$python_bin" scripts/summarize_cs2_rebuttal_eval.py "$eval_root" \
  --arm-a shuffled \
  --arm-b synchronized \
  --arm-a-eval-group-mode synchronized \
  --arm-b-eval-group-mode synchronized \
  --arm-a-n-players 10 \
  --arm-b-n-players 10 \
  --arm-a-training-group-mode shuffled \
  --arm-b-training-group-mode synchronized
