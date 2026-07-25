#!/usr/bin/env bash
# Four-node GH200 synchronized-vs-shuffled matched-information ablation.
#
# Launch this same script on every node with NODE_RANK=0..NNODES-1 and a shared MASTER_ADDR,
# CS1K_DATASET_DIR, CS1K_CODEC_CHECKPOINT, and CS1K_OUTPUT_ROOT. Each arm uses the identical
# ten-player architecture, global batch, action/video volume, seed, GPU topology, and wall-clock
# budget; only whether the ten POVs come from the same synchronized round changes.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$PWD}
dataset_dir=${CS1K_DATASET_DIR:?Set CS1K_DATASET_DIR on every node}
manifest_path=${CS1K_MANIFEST_PATH:?Set the frozen confirmatory manifest path on every node}
split_provenance=${CS1K_CONFIRMATORY_SPLIT_PROVENANCE:?Set the frozen split provenance on every node}
codec_checkpoint=${CS1K_CODEC_CHECKPOINT:?Set CS1K_CODEC_CHECKPOINT on every node}
output_root=${CS1K_OUTPUT_ROOT:?Set CS1K_OUTPUT_ROOT to a shared result directory}
arm_hours=${CS1K_ARM_HOURS:?Set the per-arm wall-clock budget}
seed=${CS1K_SEED:-28}
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

nnodes=${NNODES:-4}
node_rank=${NODE_RANK:?Set NODE_RANK=0..NNODES-1}
nproc_per_node=${NPROC_PER_NODE:-1}
master_addr=${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 hostname or IP}
master_port=${MASTER_PORT:-29500}
dataloader_workers=${CS1K_DATALOADER_WORKERS:?Freeze workers from the GH200 loader benchmark}
dataloader_prefetch_factor=${CS1K_DATALOADER_PREFETCH_FACTOR:?Freeze prefetch from the GH200 benchmark}
dataloader_persistent_workers=${CS1K_DATALOADER_PERSISTENT_WORKERS:?Freeze persistence from the benchmark}
dataloader_pin_memory=${CS1K_DATALOADER_PIN_MEMORY:?Freeze pin-memory from the GH200 benchmark}
loader_benchmark_jsons=${CS1K_LOADER_BENCHMARK_JSONS:?Provide at least three GH200 benchmark JSONs}

if [[ "$nnodes" != 4 || "$nproc_per_node" != 1 ]]; then
  echo "The preregistered GH200 topology is exactly four nodes with one process/GPU per node" >&2
  exit 1
fi
if ! [[ "$node_rank" =~ ^[0-3]$ ]]; then
  echo "NODE_RANK must be one of 0, 1, 2, or 3" >&2
  exit 1
fi

python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
torchrun_bin=${TORCHRUN_BIN:-$(dirname "$python_bin")/torchrun}

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export WANDB_MODE=disabled

code_commit=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "GH200 publication runs require a clean source tree" >&2
  exit 1
fi
if [[ ! -f "$codec_checkpoint" ]]; then
  echo "Codec checkpoint not found: $codec_checkpoint" >&2
  exit 1
fi
if [[ ! -f "$manifest_path" ]]; then
  echo "Confirmatory manifest not found: $manifest_path" >&2
  exit 1
fi
if [[ ! -f "$split_provenance" ]]; then
  echo "Confirmatory split provenance not found: $split_provenance" >&2
  exit 1
fi
if [[ "$(dirname "$(realpath "$manifest_path")")" != "$(realpath "$dataset_dir")" ]]; then
  echo "CS1K_MANIFEST_PATH must live directly inside CS1K_DATASET_DIR" >&2
  exit 1
fi
IFS=, read -r -a arms <<<"$arm_order"
if [[ ${#arms[@]} -ne 2 ]]; then
  echo "CS1K_ARM_ORDER must contain exactly two comma-separated arms" >&2
  exit 1
fi
for arm in "${arms[@]}"; do
  if [[ "$arm" != synchronized && "$arm" != shuffled ]]; then
    echo "Unsupported arm $arm; expected synchronized or shuffled" >&2
    exit 1
  fi
done
if [[ "${arms[0]}" == "${arms[1]}" ]]; then
  echo "CS1K_ARM_ORDER must contain synchronized and shuffled once each" >&2
  exit 1
fi

experiment_root=$output_root/seed_$seed
node_provenance=$experiment_root/provenance/node_$node_rank
mkdir -p "$node_provenance"
"$python_bin" scripts/prepare_cs2_confirmatory_split.py \
  --source-manifest "$dataset_dir/manifest.parquet" \
  --output-manifest "$manifest_path" \
  --provenance-output "$split_provenance" \
  --verify-only \
  >"$node_provenance/confirmatory_split_verification.json"
read -r -a loader_benchmark_paths <<<"$loader_benchmark_jsons"
"$python_bin" scripts/validate_cs2_loader_selection.py "${loader_benchmark_paths[@]}" \
  --num-workers "$dataloader_workers" \
  --prefetch-factor "$dataloader_prefetch_factor" \
  --persistent-workers "$dataloader_persistent_workers" \
  --pin-memory "$dataloader_pin_memory" \
  --expected-git-commit "$code_commit" \
  --expected-gpu-substring GH200 \
  --output "$node_provenance/frozen_loader_selection.json"
"$python_bin" scripts/prepare_counterstrike1k.py \
  --data-root "$dataset_dir" \
  --manifest "$manifest_path" \
  --map-slug dust2 \
  --provenance-output "$node_provenance/dataset.json"
git rev-parse HEAD >"$node_provenance/code_commit.txt"
git status --porcelain=v1 >"$node_provenance/code_status.txt"
git diff --binary >"$node_provenance/code.patch"
sha256sum "$codec_checkpoint" >"$node_provenance/codec_checkpoint.sha256"
sha256sum "$manifest_path" "$split_provenance" \
  >"$node_provenance/confirmatory_split_files.sha256"
sha256sum pixi.lock pyproject.toml >"$node_provenance/environment_files.sha256"
"$python_bin" - <<'PY' >"$node_provenance/installed_packages.txt"
from importlib.metadata import distributions

packages = sorted(
    (distribution.metadata["Name"], distribution.version)
    for distribution in distributions()
    if distribution.metadata["Name"]
)
for name, version in packages:
    print(f"{name}=={version}")
PY
nvidia-smi -q >"$node_provenance/nvidia_smi_q.txt"
uname -a >"$node_provenance/uname.txt"
printf '%s\n' \
  "seed=$seed" \
  "arm_hours=$arm_hours" \
  "arm_order=$arm_order" \
  "nnodes=$nnodes" \
  "nproc_per_node=$nproc_per_node" \
  "master_addr=$master_addr" \
  "master_port=$master_port" \
  "manifest_path=$manifest_path" \
  "manifest_sha256=$(sha256sum "$manifest_path" | cut -d' ' -f1)" \
  "split_provenance=$split_provenance" \
  "split_provenance_sha256=$(sha256sum "$split_provenance" | cut -d' ' -f1)" \
  "action_routing=spatial" \
  "dataloader_workers=$dataloader_workers" \
  "dataloader_prefetch_factor=$dataloader_prefetch_factor" \
  "dataloader_persistent_workers=$dataloader_persistent_workers" \
  "dataloader_pin_memory=$dataloader_pin_memory" \
  "frozen_loader_selection_sha256=$(sha256sum "$node_provenance/frozen_loader_selection.json" | cut -d' ' -f1)" \
  >"$node_provenance/launcher.env"

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
  >"$node_provenance/gpu_timeseries.csv" \
  2>"$node_provenance/gpu_timeseries.log" &
telemetry_pid=$!

torchrun_args=(
  --nnodes "$nnodes"
  --nproc-per-node "$nproc_per_node"
  --node-rank "$node_rank"
  --master-addr "$master_addr"
  --master-port "$master_port"
)

write_status() {
  if [[ "$node_rank" == 0 ]]; then
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" \
      >>"$experiment_root/status.tsv"
  fi
}

for arm in "${arms[@]}"; do
  write_status "$arm" running
  "$torchrun_bin" "${torchrun_args[@]}" scripts/train_world_model.py \
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
    run.output_dir="$experiment_root/$arm" \
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
    wandb.mode=disabled
  write_status "$arm" complete
done

write_status ablation complete
