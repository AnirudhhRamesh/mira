# Entry-point scripts

Hydra applications for training, evaluation, and serving. Each reads its config from `configs/`
(override any key as `key=value`) and runs single-GPU, or multi-GPU under `torchrun`.

- `train_codec.py` — train the video codec.
- `train_world_model.py` — train the world model (single-player, or add
  `model=multi_wrapper_world_model dataset.n_players=4` for 4-player).
- `eval_world_model_offline.py` — offline evaluation of a trained world model (validation loss +
  rollout metrics) from a checkpoint.
- `bench_wm_speed.py` — micro-benchmark world-model rollout speed.
- `bench_cs2_dataloader.py` — benchmark the exact Dust2 MIRA input contract across worker settings
  and fail closed unless the fixed first-round single/synchronized video and action tensors match.
- `render_cs2_rebuttal_report.py` — render deterministic Markdown tables only after the complete
  Dust2 run audit passes and all summaries match the audited checkpoint and seed contracts.

See [`configs/README.md`](../configs/README.md) for the config layout and the top-level
[`README.md`](../README.md) for example commands.
