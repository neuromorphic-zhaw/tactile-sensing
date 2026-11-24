# Deploy Checkpoints — Top 10 (updated 2026-10-19)

Best-val checkpoints ranked by **mean(val, test)**. All from the 10ms sweep
(`speck_compatible/devin_10ms/`) unless noted; new runs displace the bottom
entry when their mean beats it — see "Watch list" below.

## Shared training config (all 10)

- **Data**: `dataset_curv_padded_10ms` — 200 clips (100 flat / 100 curved),
  events binned at **10ms** windows → `T=1331`, tensor `(T, 2, 128, 128)`
- **Split**: 70/15/15 random, `seed=42` (except stratified runs)
- **Model**: MAX-taper 9-core DynapCNN SNN (~182K params), verified mapper
  `{1:0, 2:1, 3:3, 4:2, 5:8, 6:4, 7:6, 8:5, 9:7}`
- **Forward**: `x[:, ::4]` → **333 effective steps** (unless noted), mean
  spike-rate logits over steps → CE loss + label_smoothing 0.1
- **Optimizer**: AdamW + cosine schedule, 100 epochs (unless noted), grad-clip 1.0
- **Base aug** (present in ALL runs): 50% temporal roll ±12 bins
- **Load note**: checkpoint keys are `spiking_backbone.*`; filter `*.v_mem`
  buffers (IAF states). Never load with `strict=False` unverified — a prefix
  mismatch silently yields a random-weight model.
- **Deploy note**: chip is async — sim stride/binning does not carry to hardware.

## Leaderboard

| # | File | Val | Test | Mean | Recipe (Δ vs baseline) | Stride/steps | Batch | LR | WD | Best@ep | Job |
|---|------|-----|------|------|------------------------|--------------|-------|----|----|---------|-----|
| 1 | `wclamp_v93.33_t86.67.pt` | 93.33 | 86.67 | **90.0** | +\|W\|<1.0 clamp (deploy-safe) | ::4 / 333 | 10 | 2e-4 | 1e-4 | 47 | 1069317 |
| 2 | `stretch_aug_v93.33_t83.33.pt` | 93.33 | 83.33 | 88.3 | +time-stretch aug 0.8–1.2× | ::4 / 333 | 10 | 2e-4 | 1e-4 | 53 | 1069300 |
| 3 | `seed2_v83.33_t90.00.pt` | 83.33 | 90.00 | 86.7 | baseline, RNG seed2 | ::4 / 333 | 10 | 2e-4 | 1e-4 | 46 | 1069320 |
| 4 | `stride2_v93.33_t80.00.pt` | 93.33 | 80.00 | 86.7 | **::2 stride** (finer temporal) | ::2 / 666 | 10 | 2e-4 | 1e-4 | 47 | 1069302 |
| 5 | `wclamp_stretch_v90.00_t83.33.pt` ⭐NEW | 90.00 | 83.33 | 86.7 | **wclamp + stretch** — the two best deltas stacked | ::4 / 333 | 10 | 2e-4 | 1e-4 | 78 | 1069328 |
| 6 | `batch8_v87.50_t83.33.pt` | 87.50 | 83.33 | 85.4 | batch 8 | ::4 / 333 | 8 | 2e-4 | 1e-4 | 52 | 1069315 |
| 7 | `no_stride_v86.67_t83.33.pt` ⭐NEW | 86.67 | 83.33 | 85.0 | **::1 full stream**, 1331 steps/sample | ::1 / 1331 | 10 | 2e-4 | 1e-4 | 60 | 1069301 |
| 8 | `lr5e4_v93.33_t76.67.pt` | 93.33 | 76.67 | 85.0 | lr 5e-4 | ::4 / 333 | 10 | 5e-4 | 1e-4 | 46 | 1069313 |
| 9 | `lr1e4_v86.67_t80.00.pt` | 86.67 | 80.00 | 83.3 | lr 1e-4 | ::4 / 333 | 10 | 1e-4 | 1e-4 | 65 | 1069314 |
| 10 | `drop_aug_v86.67_t80.00.pt` | 86.67 | 80.00 | 83.3 | +10% frame-level event drop | ::4 / 333 | 10 | 2e-4 | 1e-4 | 36 | 1069299 |

**Why the new ones are better**: `wclamp_stretch` stacks the two best
individual deltas (weight-clamp + time-stretch) and matches the #3/#4 mean
with a balanced 15/15 test split; `no_stride` proves full-stream (::1, no
temporal downsampling) is viable — same test score as `batch8` at mean 85.
Displaced: `wd1e3` and `seed1` (m83.3 but t76.67, weakest test in the tie
band). `s4_b8` (5ms, m83.3/t75) and `aug_all_nostride` (m66.7) evaluated but
don't crack it. `aug_all` family is now 0-for-9 — all ≤70 test, dead end.

## Watch list (may displace entries when complete)

| Candidate | Dataset | Recipe | Status |
|---|---|---|---|
| `s2_warm` (5ms) | 5ms | warm-start from wclamp, ::2 | timed out ep32 @ **86.67v** — resubmitted 30h wall |
| `s2_b4_stream` (2ms) | 2ms | ::2 stream, b4 | running, 60h wall |
| `s1_b5_stream` (5ms) | 5ms | ::1 full stream, b5 | resubmitted earth-5, 72h wall |
| `s4_b4_stream` (1ms) | 1ms | ::4 stream | queued |
| `s8_*`/`s10_b10`/`s5_b5`/`s16_best` (1ms) | 1ms | stack variants | queued, mem bumped to 256G |
| `ev_*` finished | raw | best: ev_augs t83.33 | none crack top-10 (max mean 78.3) |

Deploy pick: **`wclamp`** — smallest val→test gap, all |W| ≤ 0.41 « θ=1.0.

## SNN+SSM hybrid checkpoints (2026-10-26)

Joint checkpoints containing `spiking_backbone.*` (deployable 9-core SNN) +
`ssm_head.*` (host-side S4D state-space readout, ~137K params, replaces the
mean-rate readout). The SNN part deploys to Speck exactly like the others;
the SSM head runs on the host over windowed readout spikes — same host-side
readout pattern as live_dvs.py, just smarter temporal integration.

| File | Backbone | Data family | Val | Test | Event-level (specksim) |
|------|----------|-------------|-----|------|------------------------|
| `ssm_t10ms_frz_v80.00_t86.67.pt` | frozen wclamp | time bins | 80.00 | **86.67** | pending (job 1074885) |
| `ssm_evT166_frz_v83.33_t73.33.pt` | frozen evT166_s2 | count bins | 83.33 | 73.33 | **63.33% val** |
| `ssm_t10ms_v83.33_t70.00.pt` | joint scratch | time bins | 83.33 | 70.00 | pending (job 1074884) |
| `ssm_evT166_v80.00_t56.67.pt` | joint scratch | count bins | 80.00 | 56.67 | pending (job 1074886) |

Eval: `demo/eval_specksim_ssm.py --model <ckpt> --win_mode {time|count}`
(time=40ms windows for t10ms, count=N/83 events for evT166).
Trainer: `speck_compatible/devin_ssm/<variant>/train.py`.
