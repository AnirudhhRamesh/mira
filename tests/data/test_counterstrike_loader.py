import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mira.data.counterstrike import CS2_ACTION_DTYPE, CS2_KEYS
from mira.data.counterstrike_benchmark import (
    batch_signature,
    validate_batch_contract,
    verify_single_synchronized_parity,
)
from mira.data.training_loader import create_loader


def _fixture(root: Path) -> Path:
    (root / "videos" / "360p").mkdir(parents=True)
    (root / "actions").mkdir()
    (root / "events").mkdir()
    manifest = []
    for round_idx in range(12):
        match_id = f"{round_idx:012d}"
        round_id = f"match_{match_id}__r001"
        for pov_idx in range(10):
            sample_key = f"{round_id}__p{pov_idx:02d}"
            manifest.append(
                {
                    "sample_key": sample_key,
                    "match_id": match_id,
                    "round_id": round_id,
                    "pov_idx": pov_idx,
                    "split": "train",
                    "map_slug": "dust2",
                    "frames": 32,
                    "frame0_tick": 100,
                    "fps": 32.0,
                }
            )
            (root / "videos" / "360p" / f"{sample_key}.mp4").touch()
            actions = np.zeros(32, dtype=CS2_ACTION_DTYPE)
            actions["delta_yaw"] = 1.0
            actions["delta_pitch"] = -0.5
            actions["buttons"][::4] = 1 << 0
            actions["buttons"][1::4] = 1 << 1
            actions.tofile(root / "actions" / f"{sample_key}.actions.bin")
            (root / "events" / f"{sample_key}.events.json").write_text(
                json.dumps({"events": [{"type": "player_death", "frame_idx": 10}]})
            )
    pq.write_table(pa.Table.from_pylist(manifest), root / "manifest_dust2.parquet")
    return root


def _loader(root: Path, *, mode: str, n_players: int, window_mode: str = "midpoint"):
    return create_loader(
        root,
        dataset_backend="counterstrike1k",
        split="train",
        map_slug="dust2",
        group_mode=mode,
        window_mode=window_mode,
        clip_len=2,
        target_fps=8,
        n_players=n_players,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        infinite=False,
        frame_size=(16, 28),
        valid_keys=list(CS2_KEYS),
        pin_memory=False,
    )


def test_counterstrike_action_aggregation_and_video_subdir(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path)
    monkeypatch.setattr(
        "mira.data.counterstrike.decode_frames",
        lambda _path, indices, frame_size: torch.zeros(len(indices), 3, *frame_size, dtype=torch.uint8),
    )

    batch, metadata = next(iter(_loader(root, mode="single", n_players=1)))

    assert batch.video.shape == (1, 2, 3, 16, 28)
    assert batch.actions.key_presses.shape == (1, 2, len(CS2_KEYS))
    assert batch.actions.key_presses[0, :, :2].tolist() == [[1, 1], [1, 1]]
    assert batch.actions.mouse_movements[0].tolist() == [[4.0, -2.0], [4.0, -2.0]]
    assert torch.isnan(batch.actions.game_mouse_sensitivity).all()
    assert metadata[0].sample_key.endswith("__p00")


def test_synchronized_and_shuffled_controls(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path)
    monkeypatch.setattr(
        "mira.data.counterstrike.decode_frames",
        lambda _path, indices, frame_size: torch.zeros(len(indices), 3, *frame_size, dtype=torch.uint8),
    )

    synced, synced_meta = next(iter(_loader(root, mode="synchronized", n_players=10)))
    shuffled, shuffled_meta = next(iter(_loader(root, mode="shuffled", n_players=10)))

    assert synced.video.shape[0] == shuffled.video.shape[0] == 10
    assert len({meta.round_id for meta in synced_meta}) == 1
    assert len({meta.source_start_frame for meta in synced_meta}) == 1
    assert len({meta.round_id for meta in shuffled_meta}) == 10
    assert len({meta.match_id for meta in shuffled_meta}) == 10


def test_first_death_windows_are_centered_and_paired_across_arms(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path)
    monkeypatch.setattr(
        "mira.data.counterstrike.decode_frames",
        lambda _path, indices, frame_size: torch.zeros(len(indices), 3, *frame_size, dtype=torch.uint8),
    )

    _single, single_meta = next(iter(_loader(root, mode="single", n_players=1, window_mode="first-death")))
    _shared, shared_meta = next(
        iter(_loader(root, mode="synchronized", n_players=10, window_mode="first-death"))
    )

    # Two target frames at 8 fps require eight source frames; frame 10 is centered at start 6.
    assert single_meta[0].source_start_frame == 6
    assert {meta.source_start_frame for meta in shared_meta} == {6}
    assert single_meta[0].round_id == shared_meta[0].round_id
    assert {meta.window_mode for meta in shared_meta} == {"first-death"}


def test_single_synchronized_tensor_parity_gate(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path)

    def fake_decode(path, indices, frame_size):
        pov_idx = int(Path(path).stem.rsplit("__p", maxsplit=1)[1])
        return torch.full(
            (len(indices), 3, *frame_size),
            pov_idx,
            dtype=torch.uint8,
        )

    monkeypatch.setattr("mira.data.counterstrike.decode_frames", fake_decode)
    result = verify_single_synchronized_parity(
        root,
        clip_len=2,
        target_fps=8,
        frame_size=(16, 28),
    )

    assert result["status"] == "pass"
    assert len(result["sample_keys"]) == 10
    assert len(result["video_sha256"]) == 64
    assert result["source_start_frame"] == 12


def test_benchmark_contract_and_signature_fail_closed(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path)
    monkeypatch.setattr(
        "mira.data.counterstrike.decode_frames",
        lambda _path, indices, frame_size: torch.zeros(len(indices), 3, *frame_size, dtype=torch.uint8),
    )
    batch, metadata = next(iter(_loader(root, mode="synchronized", n_players=10)))

    validate_batch_contract(batch, metadata, group_mode="synchronized", clip_len=2)
    signature = batch_signature(batch, metadata)
    assert len(signature["metadata_sha256"]) == 64
    assert len(signature["sample_keys"]) == 10

    metadata[0].source_start_frame += 1
    with pytest.raises(ValueError, match="one source start"):
        validate_batch_contract(batch, metadata, group_mode="synchronized", clip_len=2)


def test_persistent_worker_loader_configuration(tmp_path) -> None:
    root = _fixture(tmp_path)
    loader = create_loader(
        root,
        dataset_backend="counterstrike1k",
        split="train",
        map_slug="dust2",
        group_mode="single",
        clip_len=2,
        target_fps=8,
        n_players=1,
        batch_size=10,
        num_workers=1,
        shuffle=False,
        infinite=False,
        frame_size=(16, 28),
        valid_keys=list(CS2_KEYS),
        prefetch_factor=3,
        pin_memory=False,
        persistent_workers=True,
    )

    assert loader.num_workers == 1
    assert loader.prefetch_factor == 3
    assert loader.persistent_workers is True

    with pytest.raises(ValueError, match="requires num_workers"):
        create_loader(
            root,
            dataset_backend="counterstrike1k",
            split="train",
            map_slug="dust2",
            group_mode="single",
            clip_len=2,
            target_fps=8,
            n_players=1,
            num_workers=0,
            persistent_workers=True,
        )
