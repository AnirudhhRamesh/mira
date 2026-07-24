"""Tests for freezing a GH200 loader configuration from exact-contract benchmarks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "validate_cs2_loader_selection.py"
    spec = importlib.util.spec_from_file_location("validate_cs2_loader_selection_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATE = _load_module()
COMMIT = "1" * 40


def _benchmark(path: Path, *, repeat: int, include_shuffled: bool = True, clean: bool = True) -> None:
    modes = ["synchronized", "shuffled"] if include_shuffled else ["synchronized"]
    payload = {
        "schema": "mira-cs2-dataloader-benchmark-v1",
        "status": "pass",
        "parity": {"status": "pass"},
        "provenance": {
            "git_clean": clean,
            "git_commit": COMMIT,
            "gpu": "NVIDIA GH200 480GB",
            "hostname": f"node-{repeat}",
        },
        "config": {
            "map_slug": "dust2",
            "group_modes": modes,
            "clip_len": 16,
            "target_fps": 8,
            "frame_size": [168, 308],
            "transfer_device": "cuda",
        },
        "results": [
            {
                "status": "pass",
                "group_mode": mode,
                "num_workers": 8,
                "prefetch_factor": 2,
                "persistent_workers": True,
                "pin_memory": True,
                "timed_batches": 100,
                "transfer_device": "cuda",
                "input_frames_per_second": 1000.0 + repeat,
            }
            for mode in modes
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _repeats(root: Path) -> list[Path]:
    paths = [root / f"repeat_{index}.json" for index in range(3)]
    for index, path in enumerate(paths):
        _benchmark(path, repeat=index)
    return paths


def test_freezes_repeated_target_hardware_selection(tmp_path: Path) -> None:
    result = VALIDATE.validate_selection(
        _repeats(tmp_path),
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        expected_git_commit=COMMIT,
    )

    assert result["status"] == "pass"
    assert result["selected_config"]["num_workers"] == 8
    assert result["throughput_input_frames_per_second"]["synchronized"]["n"] == 3
    assert result["throughput_input_frames_per_second"]["shuffled"]["mean"] == 1001.0
    assert len(result["benchmark_inputs"]) == 3


def test_rejects_fewer_than_three_repeats(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 3"):
        VALIDATE.validate_selection(
            _repeats(tmp_path)[:2],
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            expected_git_commit=COMMIT,
        )


def test_rejects_missing_shuffled_contract(tmp_path: Path) -> None:
    paths = _repeats(tmp_path)
    _benchmark(paths[1], repeat=1, include_shuffled=False)

    with pytest.raises(ValueError, match="synchronized and shuffled"):
        VALIDATE.validate_selection(
            paths,
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            expected_git_commit=COMMIT,
        )


def test_rejects_dirty_benchmark_source(tmp_path: Path) -> None:
    paths = _repeats(tmp_path)
    _benchmark(paths[2], repeat=2, clean=False)

    with pytest.raises(ValueError, match="source was dirty"):
        VALIDATE.validate_selection(
            paths,
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            expected_git_commit=COMMIT,
        )
