#!/usr/bin/env python3
"""Freeze a target-hardware CS2 loader configuration from repeated exact-contract benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

REQUIRED_GROUP_MODES = ("synchronized", "shuffled")
MINIMUM_REPEATS = 3
MINIMUM_TIMED_BATCHES = 100


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _summary(values: list[float]) -> dict[str, float | int]:
    _require(values and all(math.isfinite(value) and value > 0 for value in values), "invalid throughput")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def validate_selection(
    benchmark_paths: list[Path],
    *,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    expected_git_commit: str,
    expected_gpu_substring: str = "GH200",
) -> dict[str, Any]:
    _require(
        len(benchmark_paths) >= MINIMUM_REPEATS,
        f"at least {MINIMUM_REPEATS} independent benchmark repeats are required",
    )
    _require(num_workers > 0, "publication loader selection requires num_workers > 0")
    _require(prefetch_factor > 0, "prefetch_factor must be positive")
    _require(len(expected_git_commit) == 40, "expected_git_commit must be a full Git SHA")

    observed_throughput: dict[str, list[float]] = {mode: [] for mode in REQUIRED_GROUP_MODES}
    inputs: list[dict[str, Any]] = []
    for path in benchmark_paths:
        payload = _read_json(path)
        source = str(path)
        _require(payload.get("schema") == "mira-cs2-dataloader-benchmark-v1", f"{source}: schema drift")
        _require(payload.get("status") == "pass", f"{source}: benchmark did not pass")
        _require(payload.get("parity", {}).get("status") == "pass", f"{source}: tensor parity failed")

        provenance = payload.get("provenance", {})
        _require(provenance.get("git_clean") is True, f"{source}: benchmark source was dirty")
        _require(
            provenance.get("git_commit") == expected_git_commit,
            f"{source}: benchmark source commit drifted",
        )
        gpu = str(provenance.get("gpu", ""))
        _require(expected_gpu_substring.lower() in gpu.lower(), f"{source}: expected target GPU not found")

        config = payload.get("config", {})
        expected_contract = {
            "map_slug": "dust2",
            "clip_len": 16,
            "target_fps": 8,
            "frame_size": [168, 308],
        }
        for field, expected in expected_contract.items():
            _require(config.get(field) == expected, f"{source}: {field} must be {expected!r}")
        _require(config.get("transfer_device") in {"cuda", "cuda:0"}, f"{source}: CUDA transfer missing")
        _require(
            set(config.get("group_modes", [])) >= set(REQUIRED_GROUP_MODES),
            f"{source}: synchronized and shuffled benchmark cases are required",
        )

        selected: dict[str, dict[str, Any]] = {}
        for result in payload.get("results", []):
            if (
                result.get("group_mode") in REQUIRED_GROUP_MODES
                and result.get("num_workers") == num_workers
                and result.get("prefetch_factor") == prefetch_factor
                and result.get("persistent_workers") is persistent_workers
                and result.get("pin_memory") is pin_memory
            ):
                selected[str(result["group_mode"])] = result
        _require(
            set(selected) == set(REQUIRED_GROUP_MODES),
            f"{source}: selected loader configuration is missing a required group mode",
        )
        for mode, result in selected.items():
            _require(result.get("status") == "pass", f"{source}: {mode} selected case failed")
            _require(
                int(result.get("timed_batches", 0)) >= MINIMUM_TIMED_BATCHES,
                f"{source}: {mode} requires at least {MINIMUM_TIMED_BATCHES} timed batches",
            )
            _require(
                result.get("transfer_device") in {"cuda", "cuda:0"},
                f"{source}: {mode} selected case omitted CUDA transfer",
            )
            observed_throughput[mode].append(float(result["input_frames_per_second"]))

        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "gpu": gpu,
                "hostname": provenance.get("hostname"),
            }
        )

    return {
        "schema": "mira-cs2-frozen-loader-selection-v1",
        "status": "pass",
        "expected_git_commit": expected_git_commit,
        "expected_gpu_substring": expected_gpu_substring,
        "selected_config": {
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "pin_memory": pin_memory,
        },
        "required_contract": {
            "group_modes": list(REQUIRED_GROUP_MODES),
            "map_slug": "dust2",
            "clip_len": 16,
            "target_fps": 8,
            "frame_size": [168, 308],
            "transfer_device": "cuda",
            "minimum_repeats": MINIMUM_REPEATS,
            "minimum_timed_batches": MINIMUM_TIMED_BATCHES,
        },
        "throughput_input_frames_per_second": {
            mode: _summary(values) for mode, values in observed_throughput.items()
        },
        "benchmark_inputs": inputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", type=Path, nargs="+")
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--prefetch-factor", type=int, required=True)
    parser.add_argument("--persistent-workers", type=_parse_bool, required=True)
    parser.add_argument("--pin-memory", type=_parse_bool, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-gpu-substring", default="GH200")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_selection(
        args.benchmarks,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        expected_git_commit=args.expected_git_commit,
        expected_gpu_substring=args.expected_gpu_substring,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
