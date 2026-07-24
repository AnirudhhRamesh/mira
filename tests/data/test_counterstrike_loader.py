from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mira.data.counterstrike import CS2_ACTION_DTYPE, CS2_KEYS
from mira.data.training_loader import create_loader


def _fixture(root: Path) -> Path:
    (root / "videos" / "360p").mkdir(parents=True)
    (root / "actions").mkdir()
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
    pq.write_table(pa.Table.from_pylist(manifest), root / "manifest_dust2.parquet")
    return root


def _loader(root: Path, *, mode: str, n_players: int):
    return create_loader(
        root,
        dataset_backend="counterstrike1k",
        split="train",
        map_slug="dust2",
        group_mode=mode,
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
