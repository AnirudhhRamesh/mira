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
- `prepare_cs2_confirmatory_split.py` — freeze or verify the post-pilot, match-atomic Dust2
  confirmatory split by salted hash only, quarantining the already consumed release test.
- `diagnose_cs2_multi_action_routing.py` — quantify how much global or spatial multiplayer routing
  preserves a deterministic cross-POV action intervention before the diffusion transformer.
- `run_cs2_spatial_routing_preflight.sh` — run the frozen one-hour G7e validation-only engineering
  gate for spatial player-aligned action routing.
- `assess_cs2_spatial_routing_preflight.py` — enforce the preregistered routing and action-loss
  thresholds without consulting the new confirmatory test.
- `extract_cs2_future_event_features.py` — read only pre-horizon pixels/actions and export the
  frozen checkpoint's last causal context representation without editing the MIRA architecture.
- `run_cs2_frozen_event_probe.sh` — compare identical synchronized contexts represented by the
  single, synchronized-trained, and matched-information shuffled-trained checkpoints using the
  public CounterStrike-1K future-event probe. Set both `CS1K_SYNCHRONIZED_CHECKPOINT` and
  `CS1K_SHUFFLED_CHECKPOINT` for an exact common-step comparison; otherwise the launcher uses each
  arm's latest checkpoint.
- `render_cs2_rebuttal_report.py` — render deterministic Markdown tables only after the complete
  Dust2 run audit passes and all summaries match the audited checkpoint and seed contracts.
- `validate_cs2_loader_selection.py` — freeze a GH200 loader configuration only from at least three
  clean-source, transfer-inclusive, exact-contract synchronized/shuffled benchmark repeats.
- `watch_cs2_strict_audit_report.sh` — after the pinned pilot audit passes, run the stronger
  versioned identity audit and render a hash-bound Markdown report without manual intervention.

See [`configs/README.md`](../configs/README.md) for the config layout and the top-level
[`README.md`](../README.md) for example commands.
