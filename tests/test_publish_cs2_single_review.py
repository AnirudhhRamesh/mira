from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "publish_cs2_single_review.py"
    spec = importlib.util.spec_from_file_location("single_review_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PUBLISH = _load_module()


def test_safe_files_include_rollouts_but_never_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "pipeline_status.tsv").write_text("time\tcodec\trunning\n")
    trace = tmp_path / "single" / "rollout_traces" / "step-000001000"
    trace.mkdir(parents=True)
    (trace / "rollout.mp4").write_bytes(b"video")
    (trace / "metadata.json").write_text("{}")
    checkpoint = tmp_path / "single" / "checkpoint-1000"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.pth").write_bytes(b"secret")

    files = PUBLISH.safe_files(tmp_path)
    relative = {item[1] for item in files}

    assert "training-rollouts/step-000001000/rollout.mp4" in relative
    assert "training-rollouts/step-000001000/metadata.json" in relative
    assert not any("checkpoint" in item for item in relative)


def test_stage_progress_uses_frozen_step_endpoint(tmp_path: Path) -> None:
    single = tmp_path / "single"
    single.mkdir()
    (single / "metrics.jsonl").write_text('{"kind":"train","step":7500,"train/loss_total":0.4}\n')

    stage = PUBLISH.stage_manifest(tmp_path, "single", {"single": "running"})

    assert stage["endpoint_step"] == 15_000
    assert stage["progress_percent"] == 50.0
