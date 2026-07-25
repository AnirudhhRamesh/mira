#!/usr/bin/env bash
# Slurm-native orchestration for one audited four-node GH200 training seed.
#
# Submit this file with site-specific account/partition/time flags, for example:
#   sbatch --nodes=4 --ntasks-per-node=1 --gpus-per-node=1 ... \
#     scripts/run_cs2_gh200_slurm_seed.sh
#
# Required CS1K_* paths and the per-arm wall-clock budget must be exported with --export or by the
# batch environment. The script benchmarks every allocated node, freezes one deterministic loader
# configuration across all 12+ repeats, trains both arms, evaluates the untouched test, and audits.
set -euo pipefail

project_dir=${MIRA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${MIRA_PYTHON:-$project_dir/.pixi/envs/default/bin/python}
dataset_dir=${CS1K_DATASET_DIR:?Set CS1K_DATASET_DIR}
manifest_path=${CS1K_MANIFEST_PATH:?Set CS1K_MANIFEST_PATH}
split_provenance=${CS1K_CONFIRMATORY_SPLIT_PROVENANCE:?Set split provenance}
codec_checkpoint=${CS1K_CODEC_CHECKPOINT:?Set CS1K_CODEC_CHECKPOINT}
output_root=${CS1K_OUTPUT_ROOT:?Set CS1K_OUTPUT_ROOT}
arm_hours=${CS1K_ARM_HOURS:?Set CS1K_ARM_HOURS}
seed=${CS1K_SEED:-28}
prefetch_factor=${CS1K_DATALOADER_PREFETCH_FACTOR:-2}
persistent_workers=${CS1K_DATALOADER_PERSISTENT_WORKERS:-true}
pin_memory=${CS1K_DATALOADER_PIN_MEMORY:-true}
expected_manifest_sha256=33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e

: "${SLURM_JOB_ID:?Run through sbatch or inside a four-node Slurm allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
slurm_nodes=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}
if [[ "$slurm_nodes" != 4 ]]; then
  echo "The publication control requires exactly four Slurm nodes, found ${slurm_nodes:-unset}" >&2
  exit 1
fi
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "CS1K_SEED must be a non-negative integer" >&2
  exit 1
fi
for command_name in scontrol srun sha256sum; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "GH200 publication runs require a clean source tree" >&2
  exit 1
fi
for path in "$manifest_path" "$split_provenance" "$codec_checkpoint"; do
  if [[ ! -f "$path" ]]; then
    echo "Required frozen input not found: $path" >&2
    exit 1
  fi
done
if [[ "$(sha256sum "$manifest_path" | cut -d' ' -f1)" != "$expected_manifest_sha256" ]]; then
  echo "Confirmatory manifest SHA-256 drifted" >&2
  exit 1
fi

mapfile -t allocated_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#allocated_hosts[@]} -ne 4 ]]; then
  echo "Expected four hostnames from Slurm, found ${allocated_hosts[*]:-none}" >&2
  exit 1
fi
if [[ $(printf '%s\n' "${allocated_hosts[@]}" | sort -u | wc -l) -ne 4 ]]; then
  echo "Slurm allocation did not resolve to four distinct hosts" >&2
  exit 1
fi
master_addr=${allocated_hosts[0]}
master_port=${MASTER_PORT:-29500}

read -r -a srun_gpu_args <<<"${CS1K_SRUN_GPU_ARGS:---gpus-per-task=1}"
common_srun=(
  --nodes=4
  --ntasks=4
  --ntasks-per-node=1
  --kill-on-bad-exit=1
  "${srun_gpu_args[@]}"
)

benchmark_root=$output_root/loader_benchmarks/slurm_$SLURM_JOB_ID
selection_path=$benchmark_root/frozen_loader_selection_global.json
if [[ -e "$benchmark_root" ]]; then
  echo "Refusing to overwrite existing Slurm loader preflight: $benchmark_root" >&2
  exit 1
fi
mkdir -p "$benchmark_root"

export MIRA_PROJECT_DIR="$project_dir"
export MIRA_PYTHON="$python_bin"
export CS1K_DATASET_DIR="$dataset_dir"
export CS1K_MANIFEST_PATH="$manifest_path"
export CS1K_LOADER_BENCHMARK_ROOT="$benchmark_root"
export CS1K_DATALOADER_PREFETCH_FACTOR="$prefetch_factor"
export CS1K_DATALOADER_PERSISTENT_WORKERS="$persistent_workers"
export CS1K_DATALOADER_PIN_MEMORY="$pin_memory"
export CS1K_EXPECTED_MANIFEST_SHA256="$expected_manifest_sha256"

srun "${common_srun[@]}" "$project_dir/scripts/run_cs2_gh200_loader_preflight.sh"

mapfile -d '' -t benchmark_paths < <(
  find "$benchmark_root" -mindepth 2 -maxdepth 2 -type f -name 'repeat_*.json' -print0 | sort -z
)
minimum_expected=$((4 * ${CS1K_LOADER_BENCHMARK_REPEATS:-3}))
if [[ ${#benchmark_paths[@]} -lt "$minimum_expected" ]]; then
  echo "Expected at least $minimum_expected node-local benchmark JSONs, found ${#benchmark_paths[@]}" >&2
  exit 1
fi
"$python_bin" scripts/validate_cs2_loader_selection.py "${benchmark_paths[@]}" \
  --num-workers auto \
  --prefetch-factor "$prefetch_factor" \
  --persistent-workers "$persistent_workers" \
  --pin-memory "$pin_memory" \
  --expected-git-commit "$(git rev-parse HEAD)" \
  --expected-gpu-substring GH200 \
  --expected-manifest-sha256 "$expected_manifest_sha256" \
  --expected-host-count 4 \
  --minimum-repeats-per-hostname 3 \
  --output "$selection_path"

read -r selected_workers selected_prefetch selected_persistent selected_pin < <(
  "$python_bin" - "$selection_path" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))["selected_config"]
print(
    config["num_workers"],
    config["prefetch_factor"],
    str(config["persistent_workers"]).lower(),
    str(config["pin_memory"]).lower(),
)
PY
)
export CS1K_DATALOADER_WORKERS="$selected_workers"
export CS1K_DATALOADER_PREFETCH_FACTOR="$selected_prefetch"
export CS1K_DATALOADER_PERSISTENT_WORKERS="$selected_persistent"
export CS1K_DATALOADER_PIN_MEMORY="$selected_pin"
export CS1K_GLOBAL_LOADER_SELECTION="$selection_path"
export CS1K_CONFIRMATORY_SPLIT_PROVENANCE="$split_provenance"
export CS1K_CODEC_CHECKPOINT="$codec_checkpoint"
export CS1K_OUTPUT_ROOT="$output_root"
export CS1K_ARM_HOURS="$arm_hours"
export CS1K_SEED="$seed"
export CS1K_REQUIRE_SLURM=true
export NNODES=4
export NPROC_PER_NODE=1
export MASTER_ADDR="$master_addr"
export MASTER_PORT="$master_port"

srun "${common_srun[@]}" "$project_dir/scripts/run_cs2_gh200_sync_control.sh"

training_root=$output_root/seed_$seed
export CS1K_TRAINING_ROOT="$training_root"
srun \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --exclusive \
  "${srun_gpu_args[@]}" \
  "$project_dir/scripts/run_cs2_gh200_sync_control_eval.sh"

test -s "$training_root/audit.json"
printf 'completed and audited %s\n' "$training_root"
