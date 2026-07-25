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
        "manifest_sha256": AUDIT.EXPECTED_MANIFEST_SHA256,
        "split_provenance_sha256": _sha256(auditor.split_provenance),
        "action_routing": "spatial",
        "dataloader_workers": "8",
        "dataloader_prefetch_factor": "2",
        "dataloader_persistent_workers": "true",
        "dataloader_pin_memory": "true",
    }
    for rank in range(4):
        node = auditor.root / "provenance" / f"node_{rank}"
        node.mkdir(parents=True)
        (node / "code_commit.txt").write_text(COMMIT + "\n", encoding="utf-8")
        (node / "code_status.txt").write_text("", encoding="utf-8")
        (node / "code.patch").write_text("", encoding="utf-8")
        (node / "nvidia_smi_q.txt").write_text("Product Name : NVIDIA GH200\n", encoding="utf-8")
        (node / "gpu_timeseries.csv").write_text("header\nsample\n", encoding="utf-8")
        (node / "launcher.env").write_text(
            "".join(f"{key}={value}\n" for key, value in launcher.items()),
            encoding="utf-8",
        )
        (node / "frozen_loader_selection.json").write_text(json.dumps(loader), encoding="utf-8")

    commit, selected, seed, order = auditor.audit_nodes()

    assert commit == COMMIT
    assert selected["num_workers"] == 8
    assert seed == 28
    assert order == ("synchronized", "shuffled")

    (auditor.root / "provenance" / "node_3" / "gpu_timeseries.csv").write_text(
        "header\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="node_3.telemetry"):
        auditor.audit_nodes()
