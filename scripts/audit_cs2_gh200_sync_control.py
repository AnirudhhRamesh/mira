#!/usr/bin/env python3
"""Fail-closed audit for one completed four-node Dust2 synchronized-vs-shuffled run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ARMS = ("shuffled", "synchronized")
EVAL_SEEDS = (37, 38, 39, 40, 41)
ACTION_MODES = ("true", "batch-shifted", "time-shifted", "zero")
EXPECTED_SOURCE_MANIFEST_SHA256 = "e6d1199595327ccbed7a79760266f1c30fdcf23b717232705c9b11bb6d8707d3"
EXPECTED_MANIFEST_SHA256 = "33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e"
EXPECTED_SELECTION_SHA256 = "056e60b7bb4435e212f43e6a2e4c2ea0f5c1265978180588ff59689a2fbabb0f"
EXPECTED_SPLITS = {
    "train": {"matches": 36, "rounds": 766, "pov_rows": 7660},
    "val": {"matches": 3, "rounds": 54, "pov_rows": 540},
    "test": {"matches": 3, "rounds": 69, "pov_rows": 690},
    "pilot_test": {"matches": 3, "rounds": 52, "pov_rows": 520},
}
GLOBAL_FRAMES_PER_STEP = 4 * 1 * 10 * 16


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a YAML mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise ValueError(f"{path}: malformed launcher line {line!r}")
        result[key] = value
    return result


def _finite_results(payload: dict[str, Any], path: Path) -> None:
    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError(f"{path}: missing results")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in results.values()
    ):
        raise ValueError(f"{path}: results contain non-finite values")


@dataclass
class Auditor:
    root: Path
    manifest: Path
    split_provenance: Path
    arm_hours: float
    expected_training_commit: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)

    def require(self, name: str, condition: bool, evidence: Any) -> None:
        if not condition:
            raise ValueError(f"{name}: {evidence}")
        self.checks[name] = evidence

    def audit_dataset(self) -> None:
        provenance = _read_json(self.split_provenance)
        self.require(
            "dataset.source_manifest_sha256",
            provenance.get("source_manifest_sha256") == EXPECTED_SOURCE_MANIFEST_SHA256,
            provenance.get("source_manifest_sha256"),
        )
        self.require(
            "dataset.manifest_sha256",
            _sha256(self.manifest) == provenance.get("output_manifest_sha256") == EXPECTED_MANIFEST_SHA256,
            {
                "actual": _sha256(self.manifest),
                "recorded": provenance.get("output_manifest_sha256"),
            },
        )
        self.require(
            "dataset.selection_sha256",
            provenance.get("confirmatory_selection_sha256") == EXPECTED_SELECTION_SHA256,
            provenance.get("confirmatory_selection_sha256"),
        )
        self.require("dataset.map", provenance.get("map_slug") == "dust2", provenance.get("map_slug"))
        overlaps = provenance["statistics"]["match_overlap"]
        self.require("dataset.match_disjoint", all(not value for value in overlaps.values()), overlaps)
        for split, expected in EXPECTED_SPLITS.items():
            observed = provenance["statistics"]["splits"][split]
            for key, value in expected.items():
                self.require(f"dataset.{split}.{key}", observed.get(key) == value, observed.get(key))

    def audit_nodes(self) -> tuple[str, dict[str, Any], int, tuple[str, str]]:
        node_roots = sorted((self.root / "provenance").glob("node_*"))
        self.require(
            "topology.node_directories",
            [path.name for path in node_roots] == [f"node_{rank}" for rank in range(4)],
            [path.name for path in node_roots],
        )
        commits: set[str] = set()
        launchers: list[dict[str, str]] = []
        loader_configs: list[dict[str, Any]] = []
        for rank, node in enumerate(node_roots):
            commit = (node / "code_commit.txt").read_text(encoding="utf-8").strip()
            commits.add(commit)
            self.require(
                f"node_{rank}.clean_status",
                (node / "code_status.txt").read_text(encoding="utf-8") == "",
                (node / "code_status.txt").read_text(encoding="utf-8"),
            )
            self.require(
                f"node_{rank}.clean_patch",
                (node / "code.patch").stat().st_size == 0,
                (node / "code.patch").stat().st_size,
            )
            gpu = (node / "nvidia_smi_q.txt").read_text(encoding="utf-8")
            self.require(f"node_{rank}.gh200", "GH200" in gpu, gpu[:200])
            telemetry = (node / "gpu_timeseries.csv").read_text(encoding="utf-8").splitlines()
            self.require(f"node_{rank}.telemetry", len(telemetry) >= 2, len(telemetry))
            launcher = _read_env(node / "launcher.env")
            launchers.append(launcher)
            loader = _read_json(node / "frozen_loader_selection.json")
            self.require(f"node_{rank}.loader_status", loader.get("status") == "pass", loader.get("status"))
            self.require(
                f"node_{rank}.loader_commit",
                loader.get("expected_git_commit") == commit,
                loader.get("expected_git_commit"),
            )
            self.require(
                f"node_{rank}.loader_gpu",
                loader.get("expected_gpu_substring") == "GH200",
                loader.get("expected_gpu_substring"),
            )
            loader_configs.append(loader["selected_config"])

        self.require("code.single_commit", len(commits) == 1, sorted(commits))
        commit = next(iter(commits))
        if self.expected_training_commit is not None:
            self.require(
                "code.expected_commit",
                commit == self.expected_training_commit,
                {"expected": self.expected_training_commit, "observed": commit},
            )
        self.require(
            "loader.same_selection_all_nodes",
            all(config == loader_configs[0] for config in loader_configs),
            loader_configs,
        )
        common_launcher_fields = (
            "seed",
            "arm_hours",
            "arm_order",
            "nnodes",
            "nproc_per_node",
            "manifest_sha256",
            "split_provenance_sha256",
            "action_routing",
            "dataloader_workers",
            "dataloader_prefetch_factor",
            "dataloader_persistent_workers",
            "dataloader_pin_memory",
        )
        self.require(
            "launcher.same_contract_all_nodes",
            all(
                {key: launcher[key] for key in common_launcher_fields}
                == {key: launchers[0][key] for key in common_launcher_fields}
                for launcher in launchers
            ),
            [{key: launcher[key] for key in common_launcher_fields} for launcher in launchers],
        )
        launcher = launchers[0]
        self.require(
            "topology.four_by_one",
            launcher["nnodes"] == "4" and launcher["nproc_per_node"] == "1",
            [launcher["nnodes"], launcher["nproc_per_node"]],
        )
        self.require(
            "launcher.arm_hours",
            math.isclose(float(launcher["arm_hours"]), self.arm_hours),
            launcher["arm_hours"],
        )
        self.require(
            "launcher.manifest_hash",
            launcher["manifest_sha256"] == EXPECTED_MANIFEST_SHA256,
            launcher["manifest_sha256"],
        )
        self.require(
            "launcher.provenance_hash",
            launcher["split_provenance_sha256"] == _sha256(self.split_provenance),
            launcher["split_provenance_sha256"],
        )
        self.require(
            "launcher.spatial_routing",
            launcher["action_routing"] == "spatial",
            launcher["action_routing"],
        )
        order = tuple(launcher["arm_order"].split(","))
        self.require("launcher.arm_order", len(order) == 2 and set(order) == set(ARMS), order)
        return commit, loader_configs[0], int(launcher["seed"]), (order[0], order[1])

    def audit_status(self, order: tuple[str, str]) -> None:
        observed = [
            tuple(line.split("\t")[1:])
            for line in (self.root / "status.tsv").read_text(encoding="utf-8").splitlines()
        ]
        expected = [
            (order[0], "running"),
            (order[0], "complete"),
            (order[1], "running"),
            (order[1], "complete"),
            ("ablation", "complete"),
        ]
        self.require("pipeline.status", observed == expected, observed)

    def audit_training(
        self, seed: int, loader_config: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        arm_configs: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            stage = self.root / arm
            config = _read_yaml(stage / "world_model_config.yaml")
            arm_configs[arm] = config
            run = config["run"]
            dataset = config["dataset"]
            validation = config["validation"]
            routing = config["model"]["architecture"]["config"]["action_routing"]
            expected = {
                "seed": seed,
                "batch_size": 1,
                "deterministic": True,
                "max_duration_hours": self.arm_hours,
            }
            for key, value in expected.items():
                self.require(f"{arm}.run.{key}", run.get(key) == value, run.get(key))
            self.require(f"{arm}.dataset.map", dataset.get("map_slug") == "dust2", dataset)
            self.require(f"{arm}.dataset.n_players", dataset.get("n_players") == 10, dataset)
            self.require(f"{arm}.dataset.group_mode", dataset.get("group_mode") == arm, dataset)
            self.require(
                f"{arm}.dataset.manifest",
                Path(dataset["train_index"]).resolve()
                == Path(dataset["test_index"]).resolve()
                == self.manifest.resolve(),
                [dataset.get("train_index"), dataset.get("test_index")],
            )
            self.require(
                f"{arm}.dataset.splits",
                dataset.get("train_split") == "train" and dataset.get("test_split") == "val",
                [dataset.get("train_split"), dataset.get("test_split")],
            )
            self.require(
                f"{arm}.dataset.validation_group_mode",
                dataset.get("validation_group_mode") == "synchronized",
                dataset,
            )
            self.require(f"{arm}.routing", routing == "spatial", routing)
            observed_loader = {
                "num_workers": config["dataloader"]["num_workers"],
                "prefetch_factor": config["dataloader"]["prefetch_factor"],
                "persistent_workers": config["dataloader"]["persistent_workers"],
                "pin_memory": config["dataloader"]["pin_memory"],
            }
            self.require(f"{arm}.loader", observed_loader == loader_config, observed_loader)
            expected_validation = {
                "val_first": True,
                "val_every": 1000,
                "val_n_samples": 40,
                "local_rollout_every": 1000,
                "local_rollout_seed": 37,
            }
            for key, value in expected_validation.items():
                self.require(f"{arm}.validation.{key}", validation.get(key) == value, validation.get(key))

            rows = [
                json.loads(line)
                for line in (stage / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            train = [row for row in rows if row.get("kind") == "train"]
            validations = [row for row in rows if row.get("kind") == "validation"]
            traces = [row for row in rows if row.get("kind") == "rollout_trace"]
            limits = [row for row in rows if row.get("kind") == "time_limit"]
            self.require(f"{arm}.train_metrics", bool(train), len(train))
            self.require(f"{arm}.time_limit_count", len(limits) == 1, len(limits))
            elapsed = float(limits[0]["elapsed_wall_seconds"])
            self.require(
                f"{arm}.wall_clock",
                self.arm_hours * 3600 <= elapsed <= self.arm_hours * 3600 + 30,
                elapsed,
            )
            expected_trace_steps = {
                int(row["step"]) for row in validations if int(row["step"]) % 1000 == 0
            }
            observed_trace_steps = {int(row["step"]) for row in traces}
            self.require(
                f"{arm}.rollout_cadence",
                expected_trace_steps == observed_trace_steps and 0 in observed_trace_steps,
                {"expected": sorted(expected_trace_steps), "observed": sorted(observed_trace_steps)},
            )
            for step in observed_trace_steps:
                trace_root = stage / "rollout_traces" / f"step-{step:09d}"
                metadata = _read_json(trace_root / "metadata.json")
                self.require(
                    f"{arm}.rollout.{step}.contract",
                    metadata.get("seed") == 37
                    and metadata.get("split") == "val"
                    and metadata.get("action_routing") == "spatial",
                    metadata,
                )
                video = trace_root / "rollout.mp4"
                self.require(f"{arm}.rollout.{step}.video", video.stat().st_size > 0, video.stat().st_size)

            final_step = int(limits[0]["step"])
            checkpoint = stage / f"checkpoint-{final_step}" / "checkpoint.pth"
            self.require(f"{arm}.checkpoint", checkpoint.stat().st_size > 0, str(checkpoint))
            last_train = train[-1]
            expected_frames = (int(last_train["step"]) + 1) * GLOBAL_FRAMES_PER_STEP
            self.require(
                f"{arm}.frame_accounting",
                int(last_train["System/n_frames_processed"]) == expected_frames,
                {
                    "expected": expected_frames,
                    "observed": last_train["System/n_frames_processed"],
                },
            )
            results[arm] = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "time_limit_step": final_step,
                "elapsed_wall_seconds": elapsed,
                "rollout_steps": sorted(observed_trace_steps),
            }
        self.require(
            "arms.identical_model",
            arm_configs["shuffled"]["model"] == arm_configs["synchronized"]["model"],
            "model configs differ",
        )
        self.require(
            "arms.identical_optimizer",
            arm_configs["shuffled"]["optim"] == arm_configs["synchronized"]["optim"],
            "optimizer configs differ",
        )
        return results

    def _audit_eval_provenance(self, root: Path, training_commit: str) -> None:
        provenance = root / "provenance"
        self.require(
            f"{root.name}.evaluator_commit",
            (provenance / "evaluator_code_commit.txt").read_text(encoding="utf-8").strip()
            == training_commit,
            (provenance / "evaluator_code_commit.txt").read_text(encoding="utf-8").strip(),
        )
        self.require(
            f"{root.name}.clean_status",
            (provenance / "evaluator_code_status.txt").read_text(encoding="utf-8") == "",
            (provenance / "evaluator_code_status.txt").read_text(encoding="utf-8"),
        )
        self.require(
            f"{root.name}.clean_patch",
            (provenance / "evaluator_code.patch").stat().st_size == 0,
            (provenance / "evaluator_code.patch").stat().st_size,
        )

    def audit_evaluation(
        self, training_commit: str, checkpoints: dict[str, dict[str, Any]]
    ) -> dict[str, str]:
        primary = self.root / "evaluation" / "synchronized_test_seed_sweep"
        action = self.root / "evaluation" / "synchronized_test_action_loss_seed_sweep"
        checkpoint_hashes = {arm: checkpoints[arm]["checkpoint_sha256"] for arm in ARMS}
        self._audit_eval_provenance(primary, training_commit)
        self._audit_eval_provenance(action, training_commit)

        primary_summary = _read_json(primary / "summary.json")
        expected_primary = {
            "arm_a": "shuffled",
            "arm_b": "synchronized",
            "split": "test",
            "map_slug": "dust2",
            "dino_model": "dinov2_vitb14",
            "window_mode": "midpoint",
            "deterministic": True,
            "seeds": list(EVAL_SEEDS),
            "validation_raw_pov_rows_per_seed": 690,
            "metrics_raw_pov_rows_per_seed": 690,
            "action_routing": "spatial",
            "shuffled_checkpoint_sha256": checkpoint_hashes["shuffled"],
            "synchronized_checkpoint_sha256": checkpoint_hashes["synchronized"],
        }
        for key, value in expected_primary.items():
            self.require(
                f"primary.contract.{key}",
                primary_summary["contract"].get(key) == value,
                primary_summary["contract"].get(key),
            )
        primary_paths = sorted(primary.glob("seed_*/*.json"))
        self.require("primary.result_count", len(primary_paths) == 10, len(primary_paths))
        for path in primary_paths:
            payload = _read_json(path)
            _finite_results(payload, path)
            arm = path.stem
            seed = int(path.parent.name.removeprefix("seed_"))
            self.require(
                f"primary.{path.parent.name}.{arm}.membership",
                arm in ARMS and seed in EVAL_SEEDS,
                [arm, seed],
            )
            identity = (
                payload.get("split"),
                payload.get("seed"),
                payload.get("deterministic"),
                payload.get("map_slug"),
                payload.get("dino_model"),
                payload.get("group_mode"),
                payload.get("training_group_mode"),
                payload.get("n_players"),
                payload.get("action_routing"),
                payload.get("checkpoint_sha256"),
            )
            expected = (
                "test",
                seed,
                True,
                "dust2",
                "dinov2_vitb14",
                "synchronized",
                arm,
                10,
                "spatial",
                checkpoint_hashes[arm],
            )
            self.require(f"primary.{path.parent.name}.{arm}.identity", identity == expected, identity)
            self.require(
                f"primary.{path.parent.name}.{arm}.rows",
                payload["validation"]["total_raw_pov_rows"]
                == payload["metrics"]["total_raw_pov_rows"]
                == 690,
                [payload["validation"], payload["metrics"]],
            )

        action_summary = _read_json(action / "summary.json")
        expected_action = {
            "arm_a": "shuffled",
            "arm_b": "synchronized",
            "split": "test",
            "map_slug": "dust2",
            "deterministic": True,
            "seeds": list(EVAL_SEEDS),
            "action_modes": list(ACTION_MODES),
            "window_mode": "midpoint",
            "action_routing": "spatial",
            "validation_raw_pov_rows_per_arm_per_seed": 690,
            "shuffled_checkpoint_sha256": checkpoint_hashes["shuffled"],
            "synchronized_checkpoint_sha256": checkpoint_hashes["synchronized"],
        }
        for key, value in expected_action.items():
            self.require(
                f"action.contract.{key}",
                action_summary["contract"].get(key) == value,
                action_summary["contract"].get(key),
            )
        action_paths = sorted(action.glob("seed_*/*/*.json"))
        self.require("action.result_count", len(action_paths) == 40, len(action_paths))
        for path in action_paths:
            payload = _read_json(path)
            _finite_results(payload, path)
            arm = path.parent.name
            seed = int(path.parent.parent.name.removeprefix("seed_"))
            mode = path.stem
            self.require(
                f"action.{path.parent.parent.name}.{arm}.{mode}.membership",
                arm in ARMS and seed in EVAL_SEEDS and mode in ACTION_MODES,
                [arm, seed, mode],
            )
            identity = (
                payload.get("split"),
                payload.get("seed"),
                payload.get("deterministic"),
                payload.get("map_slug"),
                payload.get("group_mode"),
                payload.get("training_group_mode"),
                payload.get("n_players"),
                payload.get("action_routing"),
                payload.get("action_mode"),
                payload.get("checkpoint_sha256"),
            )
            expected = (
                "test",
                seed,
                True,
                "dust2",
                "synchronized",
                arm,
                10,
                "spatial",
                mode,
                checkpoint_hashes[arm],
            )
            self.require(
                f"action.{path.parent.parent.name}.{arm}.{mode}.identity",
                identity == expected,
                identity,
            )
            self.require(
                f"action.{path.parent.parent.name}.{arm}.{mode}.rows",
                payload["validation"]["total_raw_pov_rows"] == 690,
                payload["validation"],
            )
        return {
            "primary_summary": str(primary / "summary.json"),
            "action_summary": str(action / "summary.json"),
        }

    def run(self) -> dict[str, Any]:
        self.audit_dataset()
        commit, loader, seed, order = self.audit_nodes()
        self.audit_status(order)
        checkpoints = self.audit_training(seed, loader)
        evaluation = self.audit_evaluation(commit, checkpoints)
        return {
            "schema": "mira-cs2-gh200-sync-control-audit-v1",
            "status": "pass",
            "experiment_root": str(self.root.resolve()),
            "training_commit": commit,
            "seed": seed,
            "arm_order": list(order),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "checkpoints": checkpoints,
            "evaluation": evaluation,
            "checks": self.checks,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-provenance", type=Path, required=True)
    parser.add_argument("--arm-hours", type=float, required=True)
    parser.add_argument("--expected-training-commit")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = Auditor(
        root=args.experiment_root,
        manifest=args.manifest,
        split_provenance=args.split_provenance,
        arm_hours=args.arm_hours,
        expected_training_commit=args.expected_training_commit,
    ).run()
    output = args.output or args.experiment_root / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
