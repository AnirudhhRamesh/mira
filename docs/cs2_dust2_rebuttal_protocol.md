# CounterStrike-1K Dust2 MIRA rebuttal protocol

This protocol is versioned before inspecting either final world-model result. It distinguishes the
single-GPU pilot from the confirmatory matched-information experiment and fixes the held-out
evaluation contract in advance.

## Scope and data unit

- Map: Dust2 only.
- Source: CounterStrike-1K 360p WebDataset v12, materialized and verified by
  `scripts/prepare_counterstrike1k.py`.
- Training support: 39 matches, 835 complete ten-POV rounds, 8,350 POV rows, 95.386 aligned
  POV-hours (the sum over the ten synchronized views inside their common temporal intersections).
- Validation: 3 matches / 54 rounds. Test: 3 disjoint matches / 52 rounds.
- No match may cross train, validation, or test. All ten POV rows of a round remain in one split.
- Input: 168x308 RGB, 16 frames at 8 fps. The codec has temporal stride 2, so the model receives
  eight latent frames covering two seconds.

The canonical selection digest, full-manifest digest, source shard list, payload counts, split
statistics, and leakage check are written beside every run. Already materialized payloads may be
reused only if the verifier passes; otherwise the preparation script extracts the selected members
from the derived shard list.

For the v12 360p release used here, the canonical selection SHA-256 is
`5b4733ba910c96221b06923699b6595b27b23ec9659cb1b15032bbf7dceb9cf9` and the full manifest
SHA-256 is `e6d1199595327ccbed7a79760266f1c30fdcf23b717232705c9b11bb6d8707d3`.

## Pilot question: single versus shared MIRA

Question: under an equal GPU wall-clock budget, does a ten-POV shared MIRA baseline trained on
synchronized Dust2 rounds outperform a single-POV MIRA baseline?

Fixed controls:

- one shared, frozen codec excluded from the comparison budget;
- identical DINOv2-B/14 codec, spatial/temporal resolution, action vocabulary, inner transformer
  width/depth, optimizer, seed, and GPU;
- 5.5 timed hours per arm on the same RTX PRO 6000 Blackwell Server Edition;
- ten raw POV rows, 160 video frames, and ten action streams presented per optimizer step
  (single batch 10 versus shared group batch 1 x 10);
- identical loader worker, prefetch, persistence, pinning, and shuffle-buffer settings in both arms;
- final time-limit checkpoint with EMA weights; no best-test-checkpoint selection;
- strict deterministic Torch algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Equal wall time is the primary compute match. Because joint spatial attention and independent
batched attention do not have equal FLOP scaling, the report must also disclose optimizer steps,
processed POV frames, throughput, parameter counts, and peak memory. The pilot is evidence about
the released training path, not by itself a causal estimate of synchronization.

## Confirmatory question: synchronization versus matched shuffled information

Question: holding the ten-player architecture and information volume fixed, does training on POVs
from the same synchronized round improve prediction relative to ten POVs drawn from different
rounds?

`group_mode=synchronized` and `group_mode=shuffled` use:

- the same ten-player wrapper, token/action counts, global batch, codec, initialization seed,
  optimizer, four-node GH200 topology, and per-arm wall-clock budget;
- the same fixed splits and complete-round eligibility rule;
- different training grouping only.

Both trained models are evaluated on synchronized test groups. Evaluating the shuffled-trained arm
on shuffled groups would change the estimand and is prohibited. During training,
`dataset.validation_group_mode=synchronized` also forces both periodic validation and rollout
metrics onto the same synchronized held-out task; only the training loader grouping differs. Run at
least three training seeds; counterbalance arm order across seeds.
`scripts/run_cs2_gh200_sync_control.sh` runs one seed and accepts an explicit arm order, while
`scripts/run_cs2_gh200_sync_control_eval.sh` forces synchronized test grouping.

## Held-out endpoints

The test suite evaluates every one of the 52 complete test rounds at its deterministic midpoint,
which is 520 raw POV clips per model and seed.

Primary endpoints:

1. held-out diffusion `loss_total`;
2. Frechet DINO distance versus the codec reconstruction after four unrolled latent frames
   (one generated second).

Secondary endpoints:

- raw Frechet DINO and Inception distances;
- DINO cosine/L2 and latent drift;
- PSNR, LPIPS, and SSIM;
- codec reconstruction floor;
- denoising latency and raw-POV latent throughput;
- train/validation curves at equal wall time and matched processed frames.

The metric backbone is public `dinov2_vitb14` for all arms. Test rollout seeds are fixed to
37, 38, and 39 for the pilot and 37 through 41 for the GH200 control. Validation, rollout-metric,
and speed phases reseed independently so enabling or skipping one phase cannot change another.
Sample counts must divide batch size exactly; silent truncation is an error.

## Action-conditioning diagnostic

For each final pilot checkpoint, validation diffusion loss is recomputed on all 520 test POV rows
with the same videos, windows, and diffusion RNG under:

- true actions;
- actions cyclically shifted across POV rows;
- actions shifted by half a clip in time;
- zero keyboard/mouse actions.

The paired quantity is `ablated loss - true-action loss`; positive values indicate that the model
uses the correctly aligned actions. Zero actions are an out-of-distribution diagnostic, not the
primary comparison. This diagnostic does not replace event-level evaluation and must not be
described as proof of causal event fidelity.

As a preregistered event-focused secondary diagnostic, repeat that exact intervention on two-second
windows centered at the first in-range `player_death` event in each held-out round. All 52 Dust2
test rounds are eligible. The source-frame start is `death_frame - required_source_frames / 2`,
clamped only at round boundaries, and is identical for all ten POVs and both model arms. Evaluate
all 520 raw POV rows with seeds 37 through 41. This tests whether aligned actions become more
important in immediate combat/death context; it remains a held-out diffusion-loss diagnostic and
must not be reported as generated death-classification accuracy.

## Reporting and interpretation

- Preserve individual seed JSON files; never report only the best seed.
- Report per-arm mean and sample standard deviation, plus paired arm deltas.
- For the confirmatory result, training-seed variation is the inferential unit. Multiple rollout
  seeds measure sampler variation and are not independent training replicates.
- Report failures and restarts. A run that fails before completing an optimizer step is a preflight
  failure, not a zero-valued result.
- Do not claim a synchronization benefit from the pilot alone. The synchronized-vs-shuffled shared
  architecture is the matched-information test.
- Do not generalize beyond Dust2, the two-second training window, one-second rollout, public
  DINOv2-based codec, or the tested compute range.

## Loader systems gate

The G7e pilot remains fixed at four TorchCodec CPU workers in both arms. It is not restarted or
mutated after observing partial training curves. Before confirmatory GH200 runs, execute
`scripts/bench_cs2_dataloader.py` on the target node type using the exact model-facing batch:
ten POV rows, 16 frames per row, 168x308 RGB, 8 fps, and the released MIRA action reduction.

The benchmark must:

- compare the candidate worker/prefetch settings for both single and synchronized grouping;
- verify byte-identical decoded video, key, and mouse tensors for the first deterministic
  single/synchronized round before timing;
- record the dataset manifest digest, source commit/status, command, Torch/TorchCodec/CUDA
  versions, hardware, first-batch hashes, and per-case throughput; and
- write an atomic JSON result and fail nonzero on any semantic or timing-case error.

Changing only the number of CPU workers or queue settings is allowed after this gate because it
does not change decoded samples. A GPU/DALI backend additionally requires a preregistered decoded
pixel tolerance and a model-quality control because different H.264 decoders are not assumed to be
pixel-identical. The selected loader settings are frozen identically for all confirmatory arms and
training seeds, preserved in resolved configs and node launcher provenance, and audited separately
from model quality.

For an isolated benchmark node that does not hold the full training materialization,
`scripts/stage_cs2_loader_benchmark_val.sh` downloads the three validation matches from the pinned
Hugging Face dataset revision, verifies the manifest and six shard SHA-256 digests, extracts only
the `val`/Dust2 payloads, and writes the exact selection/download provenance. This avoids reading
or copying from a live training volume.

## Reproduction entry points

- Data selection/materialization: `scripts/prepare_counterstrike1k.py`
- Isolated loader-benchmark staging: `scripts/stage_cs2_loader_benchmark_val.sh`
- Exact-contract data-loader benchmark: `scripts/bench_cs2_dataloader.py`
- G7e pilot: `scripts/run_cs2_rebuttal_pipeline.sh`
- Paired pilot evaluation: `scripts/run_cs2_rebuttal_eval.sh`
- Action loss diagnostic: `scripts/run_cs2_action_loss_ablation.sh`
- First-death-centered action diagnostic: `scripts/run_cs2_death_action_ablation.sh`
- Unattended paired-evaluation guard: `scripts/watch_cs2_rebuttal_eval.sh`
- Unattended midpoint-action guard: `scripts/watch_cs2_action_loss_ablation.sh`
- Unattended event-diagnostic guard: `scripts/watch_cs2_death_action_ablation.sh`
- Unattended final certification guard: `scripts/watch_cs2_rebuttal_audit.sh`
- GH200 matched control: `scripts/run_cs2_gh200_sync_control.sh`
- GH200 held-out evaluation: `scripts/run_cs2_gh200_sync_control_eval.sh`
- Completed pilot audit: `scripts/audit_cs2_rebuttal_run.py`

Every launcher records the code commit/status/patch, resolved Hydra configuration, dataset
selection, checkpoint hashes, environment lock hashes, installed packages, GPU details, and local
JSONL metrics needed to audit the result without W&B.

The completed-pilot auditor fails closed unless the pinned Dust2 selection and split counts,
match-disjointness, clean source provenance, exact timed stage sequence, identical loader settings,
equal 160-frame and ten-action-stream optimizer steps, final time-limit checkpoints, full 520-POV
test cardinality, fixed seed sets, checkpoint hashes, evaluator provenance, and all automatic
evaluation guards verify. The pilot run additionally records
five-second `nvidia-smi` telemetry for peak-memory
disclosure; because telemetry was enabled after the first single-arm checkpoint, single-arm peak
memory from that trace is explicitly labeled as partial, while the shared arm is fully covered.
