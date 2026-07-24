"""Validate and summarize paired CounterStrike-1K single/shared evaluation runs.

The input directory must contain ``seed_<N>/single.json`` and ``seed_<N>/shared.json`` files
written by :mod:`scripts.eval_world_model_offline`. The summarizer first enforces the experimental
contract (same split, seed, public metric backbone, and number of raw POV rows), then reports
per-arm mean/sample-standard-deviation and paired shared-minus-single deltas.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

LOWER_IS_BETTER_PREFIXES = (
    "val/loss",
    "test/loss",
    "metrics/dino_",
    "metrics/latent_",
    "metrics/frechet_",
    "metrics/lpips",
    "metrics/denoise_ms_",
)
HIGHER_IS_BETTER_PREFIXES = (
    "metrics/psnr",
    "metrics/ssim",
    "metrics/denoise_latent_fps",
    "metrics/denoise_pov_latent_fps",
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return payload


def _direction(metric: str) -> str:
    if metric.startswith(LOWER_IS_BETTER_PREFIXES):
        return "lower"
    if metric.startswith(HIGHER_IS_BETTER_PREFIXES):
        return "higher"
    return "descriptive"


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected finite metric values, got {values}")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _validate_pair(single: dict[str, Any], shared: dict[str, Any], source: Path) -> None:
    for field in ("split", "seed", "deterministic", "map_slug", "dino_model"):
        if single.get(field) != shared.get(field):
            raise ValueError(
                f"{source}: paired field {field!r} differs: "
                f"single={single.get(field)!r}, shared={shared.get(field)!r}"
            )
    if single.get("group_mode") != "single" or single.get("n_players") != 1:
        raise ValueError(f"{source}: single arm must use group_mode=single and n_players=1")
    if shared.get("group_mode") != "synchronized" or shared.get("n_players") != 10:
        raise ValueError(f"{source}: shared arm must use group_mode=synchronized and n_players=10")
    if not single.get("deterministic"):
        raise ValueError(f"{source}: strict deterministic evaluation was not enabled")
    for phase in ("validation", "metrics"):
        single_rows = single[phase]["total_raw_pov_rows"]
        shared_rows = shared[phase]["total_raw_pov_rows"]
        if single_rows != shared_rows:
            raise ValueError(
                f"{source}: {phase} raw POV rows differ: single={single_rows}, shared={shared_rows}"
            )
    if set(single["results"]) != set(shared["results"]):
        raise ValueError(f"{source}: single/shared result metric keys differ")


def summarize(root: Path) -> dict[str, Any]:
    seed_dirs = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {root}")

    arms: dict[str, list[dict[str, Any]]] = {"single": [], "shared": []}
    for seed_dir in seed_dirs:
        single = _read(seed_dir / "single.json")
        shared = _read(seed_dir / "shared.json")
        _validate_pair(single, shared, seed_dir)
        arms["single"].append(single)
        arms["shared"].append(shared)

    seeds = [int(item["seed"]) for item in arms["single"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Evaluation seeds are not unique: {seeds}")
    for arm in arms.values():
        if len({item["checkpoint_sha256"] for item in arm}) != 1:
            raise ValueError("An arm changed checkpoints between evaluation seeds")

    metric_names = sorted(arms["single"][0]["results"])
    arm_summary: dict[str, Any] = {}
    for arm_name, payloads in arms.items():
        arm_summary[arm_name] = {
            metric: _summary([float(payload["results"][metric]) for payload in payloads])
            for metric in metric_names
        }

    paired: dict[str, Any] = {}
    for metric in metric_names:
        deltas = [
            float(shared["results"][metric]) - float(single["results"][metric])
            for single, shared in zip(arms["single"], arms["shared"], strict=True)
        ]
        direction = _direction(metric)
        improvements = (
            [-delta for delta in deltas] if direction == "lower" else deltas if direction == "higher" else []
        )
        paired[metric] = {
            "direction": direction,
            "shared_minus_single": _summary(deltas),
        }
        if improvements:
            paired[metric]["shared_improvement"] = _summary(improvements)

    first_single = arms["single"][0]
    first_shared = arms["shared"][0]
    return {
        "contract": {
            "split": first_single["split"],
            "map_slug": first_single["map_slug"],
            "dino_model": first_single["dino_model"],
            "deterministic": first_single["deterministic"],
            "seeds": seeds,
            "validation_raw_pov_rows_per_seed": first_single["validation"]["total_raw_pov_rows"],
            "metrics_raw_pov_rows_per_seed": first_single["metrics"]["total_raw_pov_rows"],
            "single_checkpoint_sha256": first_single["checkpoint_sha256"],
            "shared_checkpoint_sha256": first_shared["checkpoint_sha256"],
        },
        "arms": arm_summary,
        "paired": paired,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing seed_<N>/ result pairs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <root>/summary.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.root / "summary.json"
    payload = summarize(args.root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
