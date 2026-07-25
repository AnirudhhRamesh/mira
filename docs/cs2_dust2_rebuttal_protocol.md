# CounterStrike-1K Dust2 MIRA rebuttal protocol

The original pilot protocol was versioned before inspecting either pilot world-model result. This
document now also records a dated post-pilot amendment, frozen before corrective training or any
inspection of the new confirmatory test clips. It distinguishes exploratory/post-hoc evidence from
the untouched matched-information endpoint.

## Scope and data unit

- Map: Dust2 only.
- Source: CounterStrike-1K 360p WebDataset v12, materialized and verified by
  `scripts/prepare_counterstrike1k.py`.
- Release training support used by the pilot: 39 matches, 835 complete ten-POV rounds, 8,350 POV
  rows, 95.386 aligned POV-hours (the sum over the ten synchronized views inside their common
  temporal intersections).
- Release validation: 3 matches / 54 rounds. Release test consumed by the pilot: 3 disjoint
  matches / 52 rounds.
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

### Post-pilot confirmatory split amendment (frozen 2026-07-25 UTC)

The spatial action router described below was selected after looking at the pilot test result.
Therefore the release test is relabeled `pilot_test` and cannot be reused as untouched confirmatory
evidence. `scripts/prepare_cs2_confirmatory_split.py` deterministically hash-ranks only release-train
match identifiers; it never reads video, controls, events, duration, or a model metric. With salt
`cs1k-dust2-spatial-routing-confirmatory-v1`, the first three matches are:

- `0a3129ba726a`
- `798dd34447aa`
- `55065858bcce`

Those matches become the new `test`; the other 36 release-train matches remain `train`; release
validation stays `val`; and release test becomes `pilot_test`. The resulting contract is:

- train: 36 matches / 766 rounds / 7,660 POV rows / 87.090 aligned POV-hours;
- validation: 3 matches / 54 rounds / 540 POV rows / 4.959 aligned POV-hours;
- untouched confirmatory test: 3 matches / 69 rounds / 690 POV rows / 8.296 aligned POV-hours;
- quarantined pilot test: 3 matches / 52 rounds / 520 POV rows.

The semantic selection digest is
`056e60b7bb4435e212f43e6a2e4c2ea0f5c1265978180588ff59689a2fbabb0f`. The generated Parquet
manifest and provenance are verified independently on every training node before launch. Any
change to the salt, selected identifiers, split cardinalities, source-manifest digest, or semantic
digest defines a different experiment.

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

- the same ten-player wrapper with `action_routing=spatial`, token/action counts, global batch,
  codec, initialization seed, optimizer, four-node GH200 topology, and per-arm wall-clock budget;
- the same amended, frozen splits and complete-round eligibility rule;
- different training grouping only.

Both trained models are evaluated on synchronized test groups. Evaluating the shuffled-trained arm
on shuffled groups would change the estimand and is prohibited. During training,
`dataset.validation_group_mode=synchronized` also forces both periodic validation and rollout
metrics onto the same synchronized held-out task; only the training loader grouping differs. Run at
least three training seeds; counterbalance arm order across seeds.
`scripts/run_cs2_gh200_sync_control.sh` runs one seed and accepts an explicit arm order, while
`scripts/run_cs2_gh200_sync_control_eval.sh` forces synchronized test grouping. The held-out
launcher also runs paired true, cross-POV-shifted, time-shifted, and zero-action diffusion-loss
interventions for both arms on the same confirmatory windows and RNG seeds. Thus the primary
quality comparison is accompanied by a direct conditioning-use check rather than treating visual
metrics alone as evidence that either model uses player actions. Both training arms persist the
same deterministic validation rollout every 1,000 steps for private review, and every node records
five-second GPU utilization, memory, temperature, and power telemetry for the full timed run.

### Post-pilot action-routing amendment

The released MIRA multiplayer wrapper adds player identity, projects all ten player action streams,
and then averages the player axis into one global action vector. That vector is broadcast over the
entire ten-POV latent grid. A post-hoc cyclic cross-POV intervention on the completed pilot shared
checkpoint changed projected per-player conditioning by RMS `0.085145`, but the global router
passed only RMS `0.0001268` (mean attenuation `0.001494`; routed cosine `0.999995`). This explains
why the pilot shared model was effectively insensitive to player/action correspondence; it is not a
data-loader mismatch.

The corrective `spatial` mode keeps the parameterization unchanged but broadcasts player `p`'s
projected action only over player `p`'s latent height band before joint spatial attention. A
routing-only counterfactual on that same checkpoint preserved the intervention with mean attenuation
`1.000000`. Both confirmatory arms use this identical router, so the only arm-level treatment remains
synchronized versus shuffled training groups. The old global-router pilot remains reported and is
not retroactively reinterpreted as confirmatory evidence.

### Frozen G7e engineering gate

Before spending GH200 compute, one one-hour synchronized spatial-router model (training seed 28) is
trained on the amended training split using the frozen pilot codec. This is an operational
preflight, not a model-quality comparison. It evaluates only all 54 release-validation rounds with
diffusion seeds 37, 38, and 39. At each 1,000-step validation point, a fixed-seed, fixed-window
rollout MP4 and JSON sidecar are written locally and mirrored to private S3 for live review. The
preflight passes only if:

- spatial routing preserves at least 0.95 of the projected cross-POV action-delta RMS;
- cross-POV-shifted actions raise mean validation loss by at least 0.005 and at least 1% relative to
  true actions; and
- the paired degradation is positive for every one of the three fixed diffusion seeds.

The thresholds are encoded in `scripts/assess_cs2_spatial_routing_preflight.py`. They must not be
relaxed after seeing the run. Failure blocks the GH200 launch and is retained as a reported result.

## Held-out endpoints

The pilot suite evaluates every one of the 52 release-test rounds at its deterministic midpoint
(520 raw POV clips per model and seed). The confirmatory suite instead evaluates every one of the
69 newly frozen test rounds (690 raw POV clips per model and seed).

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

The metric backbone is public `dinov2_vitb14` for all arms. Test rollout seeds are fixed to 37, 38,
and 39 for the pilot and 37 through 41 for the GH200 control. The GH200 evaluator derives the exact
69-round count from the frozen split provenance and rejects a conflicting manual override.
Validation, rollout-metric, and speed phases reseed independently so enabling or skipping one phase
cannot change another. Sample counts must divide batch size exactly; silent truncation is an error.

## Action-conditioning diagnostic

For each final pilot checkpoint, diffusion loss is recomputed on all 520 release-test POV rows
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

For the final GH200 checkpoints, the midpoint true-versus-cross-POV-shifted diagnostic is repeated
on all 69 untouched confirmatory rounds / 690 POV rows with seeds 37 through 41. This is a
preregistered secondary endpoint. The validation-only G7e gate and post-hoc pilot routing diagnostic
must remain visibly separated from it.

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
- Untouched confirmatory split: `scripts/prepare_cs2_confirmatory_split.py`
- Isolated loader-benchmark staging: `scripts/stage_cs2_loader_benchmark_val.sh`
- Exact-contract data-loader benchmark: `scripts/bench_cs2_dataloader.py`
- G7e pilot: `scripts/run_cs2_rebuttal_pipeline.sh`
- Paired pilot evaluation: `scripts/run_cs2_rebuttal_eval.sh`
- Action loss diagnostic: `scripts/run_cs2_action_loss_ablation.sh`
- Spatial-router representation diagnostic: `scripts/diagnose_cs2_multi_action_routing.py`
- Validation-only spatial-router preflight: `scripts/run_cs2_spatial_routing_preflight.sh`
- Frozen preflight gate: `scripts/assess_cs2_spatial_routing_preflight.py`
- First-death-centered action diagnostic: `scripts/run_cs2_death_action_ablation.sh`
- Unattended paired-evaluation guard: `scripts/watch_cs2_rebuttal_eval.sh`
- Unattended midpoint-action guard: `scripts/watch_cs2_action_loss_ablation.sh`
- Unattended event-diagnostic guard: `scripts/watch_cs2_death_action_ablation.sh`
- Unattended final certification guard: `scripts/watch_cs2_rebuttal_audit.sh`
- GH200 matched control: `scripts/run_cs2_gh200_sync_control.sh`
- GH200 held-out evaluation: `scripts/run_cs2_gh200_sync_control_eval.sh`
- GH200 completed-run audit: `scripts/audit_cs2_gh200_sync_control.py`
- GH200 training-seed aggregate: `scripts/summarize_cs2_gh200_sweep.py`
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

The GH200 evaluator ends by running its separate fail-closed auditor. It verifies the exact
four-node/one-GPU topology, GH200 identity, common clean commit, repeated frozen loader selection,
per-node telemetry, 640 global model-facing frames per optimizer step, equal wall-clock limits,
spatial routing, fixed validation-rollout cadence, final checkpoint hashes, all 69 untouched test
rounds (690 POV rows), five evaluation seeds, and all four paired action interventions.
After at least three child audits pass, the sweep summarizer requires counterbalanced arm order and
common code/data provenance, then aggregates each nested evaluation-seed mean across training
seeds. The reported independent unit is therefore the training seed; diffusion evaluation seeds
are not incorrectly promoted to independent replications.
