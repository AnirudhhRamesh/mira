#!/usr/bin/env bash
# One-GPU engineering gate for the Dust2 synchronized-vs-shuffled control.
#
# This is deliberately not the publication endpoint. It verifies on an AWS RTX PRO 6000 that the
# exact ten-player architecture, spatial action route, loader grouping intervention, timed training,
# synchronized held-out evaluation, and action-ablation summary all execute before the preregistered
# multi-seed GH200 experiment is submitted.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$PWD}
dataset_dir=${CS1K_DATASET_DIR:?Set CS1K_DATASET_DIR}
manifest_path=${CS1K_MANIFEST_PATH:?Set CS1K_MANIFEST_PATH}
split_provenance=${CS1K_CONFIRMATORY_SPLIT_PROVENANCE:?Set CS1K_CONFIRMATORY_SPLIT_PROVENANCE}
codec_checkpoint=${CS1K_CODEC_CHECKPOINT:?Set CS1K_CODEC_CHECKPOINT}
output_root=${CS1K_OUTPUT_ROOT:?Set CS1K_OUTPUT_ROOT to a new directory}
arm_hours=${CS1K_ARM_HOURS:-0.75}
seed=${CS1K_SEED:-28}
eval_seeds=${CS1K_EVAL_SEEDS:-37}
action_modes=${CS1K_ACTION_MODES:-"true batch-shifted time-shifted zero"}
dataloader_workers=${CS1K_DATALOADER_WORKERS:-4}
dataloader_prefetch_factor=${CS1K_DATALOADER_PREFETCH_FACTOR:-2}
dataloader_persistent_workers=${CS1K_DATALOADER_PERSISTENT_WORKERS:-true}
dataloader_pin_memory=${CS1K_DATALOADER_PIN_MEMORY:-true}
expected_gpu=${CS1K_EXPECTED_GPU_SUBSTRING:-"RTX PRO 6000"}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}

if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "CS1K_SEED must be a non-negative integer" >&2
  exit 1
fi
if (( seed % 2 == 0 )); then
  default_arm_order=synchronized,shuffled
else
  default_arm_order=shuffled,synchronized
fi
arm_order=${CS1K_ARM_ORDER:-$default_arm_order}
IFS=, read -r -a arms <<<"$arm_order"
if [[ ${#arms[@]} -ne 2 || "${arms[0]}" == "${arms[1]}" ]]; then
  echo "CS1K_ARM_ORDER must contain synchronized and shuffled exactly once" >&2
  exit 1
fi
for arm in "${arms[@]}"; do
  if [[ "$arm" != synchronized && "$arm" != shuffled ]]; then
    echo "Unsupported arm $arm" >&2
    exit 1
  fi
done
if [[ "$action_modes" != "true batch-shifted time-shifted zero" ]]; then
  echo "The audited preflight requires action modes: true batch-shifted time-shifted zero" >&2
  exit 1
fi

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled

if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "The G7e preflight requires a clean source tree" >&2
  exit 1
fi
for path in "$manifest_path" "$split_provenance" "$codec_checkpoint"; do
  if [[ ! -f "$path" ]]; then
    echo "Required input is absent: $path" >&2
    exit 1
  fi
done
if [[ "$(dirname "$(realpath "$manifest_path")")" != "$(realpath "$dataset_dir")" ]]; then
  echo "CS1K_MANIFEST_PATH must live directly inside CS1K_DATASET_DIR" >&2
  exit 1
fi
if [[ -e "$output_root" ]]; then
  echo "Refusing to resume or overwrite existing output root: $output_root" >&2
  exit 1
fi

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/[[:space:]]*$//')
if [[ ${#gpu_names[@]} -ne 1 ]]; then
  echo "Exactly one visible GPU is required, found ${#gpu_names[@]}" >&2
  exit 1
fi
if [[ "${gpu_names[0],,}" != *"${expected_gpu,,}"* ]]; then
  echo "Expected GPU containing '$expected_gpu', found '${gpu_names[0]}'" >&2
  exit 1
fi

mkdir -p "$output_root/provenance"
status_file=$output_root/status.tsv
code_commit=$(git rev-parse HEAD)
"$python_bin" scripts/prepare_cs2_confirmatory_split.py \
  --source-manifest "$dataset_dir/manifest.parquet" \
  --output-manifest "$manifest_path" \
  --provenance-output "$split_provenance" \
  --verify-only \
  >"$output_root/provenance/confirmatory_split_verification.json"
test_rounds=$(
  "$python_bin" - "$split_provenance" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["statistics"]["splits"]["test"]["rounds"])
PY
)
"$python_bin" scripts/prepare_counterstrike1k.py \
  --data-root "$dataset_dir" \
  --manifest "$manifest_path" \
  --map-slug dust2 \
  --provenance-output "$output_root/provenance/dataset.json"
git rev-parse HEAD >"$output_root/provenance/code_commit.txt"
git status --porcelain=v1 >"$output_root/provenance/code_status.txt"
git diff --binary >"$output_root/provenance/code.patch"
sha256sum "$codec_checkpoint" >"$output_root/provenance/codec_checkpoint.sha256"
sha256sum "$manifest_path" "$split_provenance" \
  >"$output_root/provenance/confirmatory_split_files.sha256"
sha256sum pixi.lock pyproject.toml >"$output_root/provenance/environment_files.sha256"
"$python_bin" - <<'PY' >"$output_root/provenance/installed_packages.txt"
from importlib.metadata import distributions

for name, version in sorted(
    (distribution.metadata["Name"], distribution.version)
    for distribution in distributions()
    if distribution.metadata["Name"]
):
    print(f"{name}=={version}")
PY
nvidia-smi -q >"$output_root/provenance/nvidia_smi_q.txt"
uname -a >"$output_root/provenance/uname.txt"
printf '%s\n' \
  "scope=aws_engineering_preflight_not_publication_endpoint" \
  "code_commit=$code_commit" \
  "seed=$seed" \
  "arm_hours=$arm_hours" \
  "arm_order=$arm_order" \
  "eval_seeds=$eval_seeds" \
  "action_modes=$action_modes" \
  "test_rounds=$test_rounds" \
  "gpu_name=${gpu_names[0]}" \
  "manifest_path=$manifest_path" \
  "manifest_sha256=$(sha256sum "$manifest_path" | cut -d' ' -f1)" \
  "split_provenance=$split_provenance" \
  "split_provenance_sha256=$(sha256sum "$split_provenance" | cut -d' ' -f1)" \
  "codec_checkpoint=$codec_checkpoint" \
  "codec_checkpoint_sha256=$(sha256sum "$codec_checkpoint" | cut -d' ' -f1)" \
  "action_routing=spatial" \
  "dataloader_workers=$dataloader_workers" \
  "dataloader_prefetch_factor=$dataloader_prefetch_factor" \
  "dataloader_persistent_workers=$dataloader_persistent_workers" \
  "dataloader_pin_memory=$dataloader_pin_memory" \
  >"$output_root/provenance/launcher.env"

telemetry_pid=
stop_telemetry() {
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
}
trap stop_telemetry EXIT
nvidia-smi \
  --query-gpu=timestamp,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv \
  --loop=5 \
  >"$output_root/provenance/gpu_timeseries.csv" \
  2>"$output_root/provenance/gpu_timeseries.log" &
telemetry_pid=$!

write_status() {
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >>"$status_file"
}

for arm in "${arms[@]}"; do
  write_status "$arm" running
  "$python_bin" scripts/train_world_model.py \
    model=multi_wrapper_world_model_cs2_small \
    dataset=counterstrike1k_dust2 \
    dataset.train_index="$manifest_path" \
    dataset.test_index="$manifest_path" \
    dataset.n_players=10 \
    dataset.group_mode="$arm" \
    dataset.validation_group_mode=synchronized \
    model.architecture.config.action_routing=spatial \
    model.architecture.config.wm_config.codec_checkpoint="$codec_checkpoint" \
    run.seed="$seed" \
    run.steps=100000000 \
    run.batch_size=1 \
    run.deterministic=true \
    run.compile=false \
    run.max_duration_hours="$arm_hours" \
    run.log_every=25 \
    run.checkpoint_every=1000 \
    run.checkpoint_keep_recent=10 \
    run.output_dir="$output_root/$arm" \
    dataloader.num_workers="$dataloader_workers" \
    dataloader.shuffle_buffer_size=100 \
    dataloader.prefetch_factor="$dataloader_prefetch_factor" \
    dataloader.persistent_workers="$dataloader_persistent_workers" \
    dataloader.pin_memory="$dataloader_pin_memory" \
    validation.val_first=true \
    validation.val_every=1000 \
    validation.val_n_samples=40 \
    validation.local_rollout_every=1000 \
    validation.local_rollout_seed=37 \
    validation.downstream_val_every=100000000 \
    optim.scheduler.warmup_steps=500 \
    optim.model_ema_decay=0.999 \
    world_model_metrics.n_context_frames=8 \
    world_model_metrics.num_unrolled_frames=4 \
    world_model_metrics.drift_metric_frames=4 \
    world_model_metrics.fdd_slice_frames=2 \
    world_model_metrics.dino_model=dinov2_vitb14 \
    world_model_metrics.num_samples=40 \
    world_model_metrics.per_device_batch_size=1 \
    world_model_metrics.num_viz_samples=2 \
    wandb.mode=disabled \
    2>&1 | tee "$output_root/$arm.log"
  write_status "$arm" complete
done

latest_checkpoint() {
  local arm=$1
  find "$output_root/$arm" -path '*/checkpoint-*/checkpoint.pth' -print0 |
    sort -zV | tail -z -n 1 | tr -d '\0'
}

eval_root=$output_root/evaluation/synchronized_test_action_loss
mkdir -p "$eval_root/provenance"
for arm in shuffled synchronized; do
  checkpoint=$(latest_checkpoint "$arm")
  if [[ -z "$checkpoint" ]]; then
    echo "No checkpoint found for $arm" >&2
    exit 1
  fi
  printf '%s\n' "$checkpoint" >"$eval_root/provenance/${arm}_checkpoint.txt"
  sha256sum "$checkpoint" >"$eval_root/provenance/${arm}_checkpoint.sha256"
done
git rev-parse HEAD >"$eval_root/provenance/evaluator_code_commit.txt"
git status --porcelain=v1 >"$eval_root/provenance/evaluator_code_status.txt"
git diff --binary >"$eval_root/provenance/evaluator_code.patch"
printf '%s\n' "$eval_seeds" >"$eval_root/provenance/seeds.txt"
printf '%s\n' "$action_modes" >"$eval_root/provenance/action_modes.txt"

write_status evaluation running
for eval_seed in $eval_seeds; do
  for arm in shuffled synchronized; do
    checkpoint=$(cat "$eval_root/provenance/${arm}_checkpoint.txt")
    arm_dir=$eval_root/seed_$eval_seed/$arm
    mkdir -p "$arm_dir"
    for action_mode in $action_modes; do
      "$python_bin" scripts/eval_world_model_offline.py "$checkpoint" \
        --split test \
        --group-mode synchronized \
        --action-mode "$action_mode" \
        --seed "$eval_seed" \
        --deterministic \
        --dino-model dinov2_vitb14 \
        --val-n-samples "$test_rounds" \
        --skip-metrics \
        --no-compile \
        --results-json "$arm_dir/$action_mode.json" \
        2>&1 | tee "$arm_dir/$action_mode.log"
    done
  done
done

"$python_bin" scripts/summarize_cs2_action_ablation.py "$eval_root" \
  --arm-a shuffled \
  --arm-b synchronized \
  --arm-a-eval-group-mode synchronized \
  --arm-b-eval-group-mode synchronized \
  --arm-a-n-players 10 \
  --arm-b-n-players 10 \
  --arm-a-training-group-mode shuffled \
  --arm-b-training-group-mode synchronized \
  --expected-action-routing spatial
write_status evaluation complete
write_status preflight complete
stop_telemetry
telemetry_pid=
echo "$output_root"
