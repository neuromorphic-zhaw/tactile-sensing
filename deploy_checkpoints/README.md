# Deploy Checkpoints — Top 10 (updated 2026-10-18)

Best-val checkpoints ranked by **mean(val, test)**. All from the 10ms sweep
(`speck_compatible/devin_10ms/`); new runs displace the bottom entry when their
mean beats it — see "Watch list" below.

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
- **Load note**: filter `*.v_mem` keys from the state dict
- **Deploy note**: chip is async — sim stride/binning does not carry to hardware.

## Leaderboard

| # | File | Val | Test | Mean | Recipe (Δ vs baseline) | Stride/steps | Batch | LR | WD | Best@ep | Job |
|---|------|-----|------|------|------------------------|--------------|-------|----|----|---------|-----|
| 1 | `wclamp_v93.33_t86.67.pt` | 93.33 | 86.67 | **90.0** | +\|W\|<1.0 clamp (deploy-safe) | ::4 / 333 | 10 | 2e-4 | 1e-4 | 47 | 1069317 |
| 2 | `stretch_aug_v93.33_t83.33.pt` | 93.33 | 83.33 | 88.3 | +time-stretch aug 0.8–1.2× | ::4 / 333 | 10 | 2e-4 | 1e-4 | 53 | 1069300 |
| 3 | `seed2_v83.33_t90.00.pt` | 83.33 | 90.00 | 86.7 | baseline, RNG seed2 | ::4 / 333 | 10 | 2e-4 | 1e-4 | 46 | 1069320 |
| 4 | `stride2_v93.33_t80.00.pt` | 93.33 | 80.00 | 86.7 | **::2 stride** (finer temporal) | ::2 / 666 | 10 | 2e-4 | 1e-4 | 47 | 1069302 |
| 5 | `batch8_v87.50_t83.33.pt` | 87.50 | 83.33 | 85.4 | batch 8 | ::4 / 333 | 8 | 2e-4 | 1e-4 | 52 | 1069315 |
| 6 | `lr5e4_v93.33_t76.67.pt` | 93.33 | 76.67 | 85.0 | lr 5e-4 | ::4 / 333 | 10 | 5e-4 | 1e-4 | 46 | 1069313 |
| 7 | `lr1e4_v86.67_t80.00.pt` | 86.67 | 80.00 | 83.3 | lr 1e-4 | ::4 / 333 | 10 | 1e-4 | 1e-4 | 65 | 1069314 |
| 8 | `drop_aug_v86.67_t80.00.pt` | 86.67 | 80.00 | 83.3 | +10% frame-level event drop | ::4 / 333 | 10 | 2e-4 | 1e-4 | 36 | 1069299 |
| 9 | `wd1e3_v90.00_t76.67.pt` | 90.00 | 76.67 | 83.3 | weight decay 1e-3 | ::4 / 333 | 10 | 2e-4 | 1e-3 | 41 | 1069318 |
| 10 | `seed1_v90.00_t76.67.pt` | 90.00 | 76.67 | 83.3 | baseline, RNG seed1 | ::4 / 333 | 10 | 2e-4 | 1e-4 | 33 | 1069319 |

Alternates at the 83.3 cutoff: `epochs200` v90/t76.67 (200 epochs, 1069303),
`baseline` v86.67/t73.33 (vanilla, 1069295). Both beat #10 on val but not mean.

## Watch list (may displace entries when complete)

| Candidate | Dataset | Recipe | Status |
|---|---|---|---|
| `wclamp_stretch` | 10ms | wclamp + stretch stack, ::4 | killed ep78 @ **90v** — eval job recovers test acc |
| `no_stride` | 10ms | ::1 full stream (1331 steps) | running, v86.67 @ ep60 |
| `ev_base` | raw→10ms | event-binning parity | killed ep28 @ 76.67v, rerunning |
| `ev_warp`/`ev_mixres` | raw | event-level timewarp / mixed bins | running |
| `s2_best`/`s2_strat` (5ms) | 5ms | stack, ::2 (1331 steps) | running |
| `s1_b5`/`s2_b4`/`s4_b4` streams | 5/2/1ms | stack, reduced-batch max-step | queued/running |

Deploy pick: **`wclamp`** — smallest val→test gap, all |W| ≤ 0.41 « θ=1.0.
