"""Focused fail-closed checks for the GH200 Dust2 auditor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_cs2_gh200_sync_control.py"
    spec = importlib.util.spec_from_file_location("audit_cs2_gh200_sync_control_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_module()
COMMIT = "1" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_provenance(path: Path, manifest_hash: str) -> None:
    payload = {
        "source_manifest_sha256": AUDIT.EXPECTED_SOURCE_MANIFEST_SHA256,
        "output_manifest_sha256": manifest_hash,
        "confirmatory_selection_sha256": AUDIT.EXPECTED_SELECTION_SHA256,
        "map_slug": "dust2",
        "statistics": {
            "match_overlap": {"train:test": []},
            "splits": {
                split: expected | {"aligned_pov_hours": 1.0}
                for split, expected in AUDIT.EXPECTED_SPLITS.items()
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _auditor(tmp_path: Path, monkeypatch) -> AUDIT.Auditor:
    manifest = tmp_path / "manifest.parquet"
    manifest.write_bytes(b"frozen manifest")
    monkeypatch.setattr(AUDIT, "EXPECTED_MANIFEST_SHA256", _sha256(manifest))
    split_provenance = tmp_path / "split.json"
    _split_provenance(split_provenance, _sha256(manifest))
    return AUDIT.Auditor(
        root=tmp_path / "seed_28",
        manifest=manifest,
        split_provenance=split_provenance,
        arm_hours=12.0,
        expected_training_commit=COMMIT,
    )


def test_audit_dataset_accepts_frozen_disjoint_split(tmp_path: Path, monkeypatch) -> None:
    auditor = _auditor(tmp_path, monkeypatch)
    auditor.audit_dataset()

    assert auditor.checks["dataset.test.rounds"] == 69
    assert auditor.checks["dataset.pilot_test.rounds"] == 52


def test_audit_nodes_requires_identical_four_node_contract(tmp_path: Path, monkeypatch) -> None:
    auditor = _auditor(tmp_path, monkeypatch)
    loader = {
        "status": "pass",
        "expected_git_commit": COMMIT,
        "expected_gpu_substring": "GH200",
        "expected_manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
        "selected_config": {
            "num_workers": 8,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "pin_memory": True,
        },
    }
    launcher = {
        "seed": "28",
        "arm_hours": "12.0",
        "arm_order": "synchronized,shuffled",
        "nnodes": "4",
        "nproc_per_node": "1",
        "master_addr": "nid00000",
        "master_port": "29500",
        "scheduler": "slurm",
        "slurm_job_id": "12345",
        "slurm_job_nodelist": "nid[00000-00003]",
        "manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
        "split_provenance_sha256": _sha256(auditor.split_provenance),
        "action_routing": "spatial",
        "dataloader_workers": "8",
        "dataloader_prefetch_factor": "2",
        "dataloader_persistent_workers": "true",
        "dataloader_pin_memory": "true",
    }
    hostnames = [f"nid{rank:05d}" for rank in range(4)]
    global_loader = loader | {
        "schema": "mira-cs2-frozen-loader-selection-v1",
        "expected_hostname": None,
        "benchmark_host_repeat_counts": {hostname: 3 for hostname in hostnames},
        "benchmark_inputs": [
            {
                "hostname": hostname,
                "manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
            }
            for hostname in hostnames
            for _ in range(3)
        ],
        "selection_evidence": {
            "rule": AUDIT.GLOBAL_LOADER_SELECTION_RULE,
            "selected_num_workers": 8,
        },
    }
    global_loader_text = json.dumps(global_loader, sort_keys=True)
    global_loader_sha256 = hashlib.sha256(global_loader_text.encode()).hexdigest()
    for rank in range(4):
        hostname = hostnames[rank]
        node = auditor.root / "provenance" / f"node_{rank}"
        node.mkdir(parents=True)
        (node / "code_commit.txt").write_text(COMMIT + "\n", encoding="utf-8")
        (node / "code_status.txt").write_text("", encoding="utf-8")
        (node / "code.patch").write_text("", encoding="utf-8")
        (node / "nvidia_smi_q.txt").write_text("Product Name : NVIDIA GH200\n", encoding="utf-8")
        (node / "gpu_timeseries.csv").write_text("header\nsample\n", encoding="utf-8")
        node_launcher = launcher | {
            "node_rank": str(rank),
            "hostname": hostname,
            "visible_gpu_count": "1",
            "visible_gpu_name": "NVIDIA GH200 480GB",
            "slurm_procid": str(rank),
            "slurm_nodeid": str(rank),
            "global_loader_selection_sha256": global_loader_sha256,
        }
        (node / "launcher.env").write_text(
            "".join(f"{key}={value}\n" for key, value in node_launcher.items()),
            encoding="utf-8",
        )
        node_loader = loader | {
            "expected_hostname": hostname,
            "benchmark_host_repeat_counts": {hostname: 3},
            "benchmark_inputs": [
                {
                    "hostname": hostname,
                    "manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
                }
                for _ in range(3)
            ],
        }
        (node / "frozen_loader_selection.json").write_text(
            json.dumps(node_loader),
            encoding="utf-8",
        )
        (node / "global_loader_selection.json").write_text(
            global_loader_text,
            encoding="utf-8",
        )

    commit, selected, seed, order = auditor.audit_nodes()

    assert commit == COMMIT
    assert selected["num_workers"] == 8
    assert seed == 28
    assert order == ("synchronized", "shuffled")

    mismatched_global = global_loader | {
        "selected_config": global_loader["selected_config"] | {"num_workers": 12},
    }
    mismatched_text = json.dumps(mismatched_global, sort_keys=True)
    node_3 = auditor.root / "provenance" / "node_3"
    (node_3 / "global_loader_selection.json").write_text(mismatched_text)
    node_3_launcher = (node_3 / "launcher.env").read_text()
    node_3_launcher = node_3_launcher.replace(
        f"global_loader_selection_sha256={global_loader_sha256}",
        f"global_loader_selection_sha256={hashlib.sha256(mismatched_text.encode()).hexdigest()}",
    )
    (node_3 / "launcher.env").write_text(node_3_launcher)
    with pytest.raises(ValueError, match="node_3.global_loader_selected_config"):
        auditor.audit_nodes()

    (node_3 / "global_loader_selection.json").write_text(global_loader_text)
    (node_3 / "launcher.env").write_text(
        node_3_launcher.replace(
            f"global_loader_selection_sha256={hashlib.sha256(mismatched_text.encode()).hexdigest()}",
            f"global_loader_selection_sha256={global_loader_sha256}",
        )
    )
    (auditor.root / "provenance" / "node_3" / "gpu_timeseries.csv").write_text("header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="node_3.telemetry"):
        auditor.audit_nodes()


def test_audits_complete_first_death_action_grid(tmp_path: Path, monkeypatch) -> None:
    auditor = _auditor(tmp_path, monkeypatch)
    root = auditor.root / "evaluation" / "synchronized_test_first_death_action_loss_seed_sweep"
    provenance = root / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "evaluator_code_commit.txt").write_text(COMMIT + "\n")
    (provenance / "evaluator_code_status.txt").write_text("")
    (provenance / "evaluator_code.patch").write_text("")
    (provenance / "window_mode.txt").write_text("first-death\n")
    checkpoint_hashes = {
        "shuffled": "a" * 64,
        "synchronized": "b" * 64,
    }
    summary = {
        "contract": {
            "arm_a": "shuffled",
            "arm_b": "synchronized",
            "split": "test",
            "map_slug": "dust2",
            "deterministic": True,
            "seeds": list(AUDIT.EVAL_SEEDS),
            "action_modes": list(AUDIT.ACTION_MODES),
            "window_mode": "first-death",
            "action_routing": "spatial",
            "validation_raw_pov_rows_per_arm_per_seed": 690,
            "shuffled_checkpoint_sha256": checkpoint_hashes["shuffled"],
            "synchronized_checkpoint_sha256": checkpoint_hashes["synchronized"],
        }
    }
    (root / "summary.json").write_text(json.dumps(summary))
    for seed in AUDIT.EVAL_SEEDS:
        for arm in AUDIT.ARMS:
            arm_root = root / f"seed_{seed}" / arm
            arm_root.mkdir(parents=True)
            for mode in AUDIT.ACTION_MODES:
                payload = {
                    "split": "test",
                    "seed": seed,
                    "deterministic": True,
                    "map_slug": "dust2",
                    "group_mode": "synchronized",
                    "training_group_mode": arm,
                    "n_players": 10,
                    "action_routing": "spatial",
                    "window_mode": "first-death",
                    "action_mode": mode,
                    "checkpoint_sha256": checkpoint_hashes[arm],
                    "validation": {"total_raw_pov_rows": 690},
                    "results": {"test/loss_total": 0.5},
                }
                (arm_root / f"{mode}.json").write_text(json.dumps(payload))

    result = auditor._audit_action_evaluation(
        root,
        label="death_action",
        window_mode="first-death",
        training_commit=COMMIT,
        checkpoint_hashes=checkpoint_hashes,
    )

    assert result.endswith("summary.json")
    (provenance / "window_mode.txt").write_text("midpoint\n")
    with pytest.raises(ValueError, match="death_action.provenance.window_mode"):
        auditor._audit_action_evaluation(
            root,
            label="death_action",
            window_mode="first-death",
            training_commit=COMMIT,
            checkpoint_hashes=checkpoint_hashes,
        )
