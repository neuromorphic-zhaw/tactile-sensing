# Deploy Checkpoints — Top 10 (10ms sweep, 2026-10-18)

Best-val checkpoints from the completed 10ms runs, ranked by mean(val,test).
All share the same Speck-fitting 9-core topology (MAX taper, verified mapper
{1:0, 2:1, 3:3, 4:2, 5:8, 6:4, 7:6, 8:5, 9:7}), trained on
`dataset_curv_padded_10ms`, input `(B,T,2,128,128)`, `x[:, ::4]` in forward.

| # | File | Val | Test | Recipe |
|---|------|-----|------|--------|
| 1 | `wclamp_v93.33_t86.67.pt` | 93.33 | 86.67 | baseline + \|W\|<1.0 clamp — **deploy pick** |
| 2 | `stretch_aug_v93.33_t83.33.pt` | 93.33 | 83.33 | +0.8–1.2× time-stretch aug |
| 3 | `seed2_v83.33_t90.00.pt` | 83.33 | 90.00 | baseline, seed2 split (lucky test) |
| 4 | `stride2_v93.33_t80.00.pt` | 93.33 | 80.00 | `::2` stride (666 steps) |
| 5 | `batch8_v87.50_t83.33.pt` | 87.50 | 83.33 | batch 8 |
| 6 | `lr5e4_v93.33_t76.67.pt` | 93.33 | 76.67 | lr 5e-4 |
| 7 | `lr1e4_v86.67_t80.00.pt` | 86.67 | 80.00 | lr 1e-4 |
| 8 | `drop_aug_v86.67_t80.00.pt` | 86.67 | 80.00 | +10% event drop |
| 9 | `wd1e3_v90.00_t76.67.pt` | 90.00 | 76.67 | weight decay 1e-3 |
| 10 | `seed1_v90.00_t76.67.pt` | 90.00 | 76.67 | baseline, seed1 |

Alternates (tied at ~83.3 mean): `epochs200` v90/t76.67, `baseline` v86.67/t73.33.

Notes:
- `stride2` ckpt was trained WITH `x[:, ::2]` — deploy as-is (chip is async;
  stride only matters in sim). All others are `::4` sim equivalents.
- State dicts include `*.v_mem` buffers from IAFSqueeze — filter
  `k.endswith(".v_mem")` when loading into a fresh model (see
  `speck_compatible/devin_10ms/*/eval_ckpt.py`).
- `wclamp` is the recommended deploy candidate: honest val→test gap, weights
  all \|W\| ≤ 0.41 << θ=1.0 (chip-safe by construction).
- Pending: `wclamp_stretch` (v90, killed at ep78, eval job running) and
  `no_stride` (v86.67, still training) may displace entries when done.
