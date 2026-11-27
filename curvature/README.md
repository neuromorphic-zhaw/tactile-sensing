# curvature — stage 1 of the thesis (flat vs curved)

Everything for the **completed** flat-vs-curved classifier on the
Speck2f + GelSight rig — the SynSense deliverables D1–D4.

| Dir | Contents |
|---|---|
| `speck_connect/` | `collect_gzip_curv.py` — the dataset collection GUI (D2) |
| `speck_compatible/` | **The model zoo** — every deployable-shape variant, grouped by input rasterization. See its README for the naming decoder |
| `demo/` | Deployment + the real evaluators: `live_dvs.py` (on-chip demo), `eval_specksim_events.py` / `eval_specksim_ssm.py` (event-level = the only metric that counts), `model.py`, quantized-eval scripts |
| `deploy_checkpoints/` | Git-tracked .pt files of every deployment-relevant model (leaderboard in its README) |
| `specksim_eval/` | Slurm jobs + logs for event-level evals |
| `quant_eval/` `specksim_sanity/` | Early quantized/binned eval + sim sanity checks |
| `non-speck-compatible/` | Legacy pre-Speck-constraint experiments (kept for reference only) |

Data lives centrally under `../data/` (`dataset_curv*` = this stage's).
Champion: `deploy_checkpoints/evT166_s2_ft_v83.33_t90.00.pt` + 200K-event
windowed vote = **76.67% val / 70.00% test** event-level.
