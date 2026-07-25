"""Static and fail-closed checks for the four-node GH200 Slurm entrypoints."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gh200_shell_entrypoints_parse() -> None:
    scripts = [
        ROOT / "scripts" / "run_cs2_gh200_loader_preflight.sh",
        ROOT / "scripts" / "run_cs2_gh200_slurm_seed.sh",
        ROOT / "scripts" / "run_cs2_gh200_sync_control.sh",
        ROOT / "scripts" / "run_cs2_gh200_sync_control_eval.sh",
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)


def test_publication_launcher_rejects_non_slurm_invocation() -> None:
    env = {
        **os.environ,
        "CS1K_DATASET_DIR": "/nonexistent/data",
        "CS1K_MANIFEST_PATH": "/nonexistent/manifest.parquet",
        "CS1K_CONFIRMATORY_SPLIT_PROVENANCE": "/nonexistent/split.json",
        "CS1K_CODEC_CHECKPOINT": "/nonexistent/codec.pth",
        "CS1K_OUTPUT_ROOT": "/nonexistent/output",
        "CS1K_ARM_HOURS": "12",
        "CS1K_DATALOADER_WORKERS": "8",
        "CS1K_DATALOADER_PREFETCH_FACTOR": "2",
        "CS1K_DATALOADER_PERSISTENT_WORKERS": "true",
        "CS1K_DATALOADER_PIN_MEMORY": "true",
        "CS1K_LOADER_BENCHMARK_ROOT": "/nonexistent/benchmarks",
        "CS1K_GLOBAL_LOADER_SELECTION": "/nonexistent/global-selection.json",
        "NODE_RANK": "0",
        "MASTER_ADDR": "nid00000",
    }
    for key in tuple(env):
        if key.startswith("SLURM_"):
            env.pop(key)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_cs2_gh200_sync_control.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Slurm allocation" in result.stderr


def test_slurm_orchestrator_requests_exact_four_by_one_topology() -> None:
    text = (ROOT / "scripts" / "run_cs2_gh200_slurm_seed.sh").read_text()
    assert "--nodes=4" in text
    assert "--ntasks=4" in text
    assert "--ntasks-per-node=1" in text
    assert "--num-workers auto" in text
    assert "--expected-host-count 4" in text
    assert "run_cs2_gh200_sync_control_eval.sh" in text
