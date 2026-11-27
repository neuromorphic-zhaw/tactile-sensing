# specksim_eval — event-level evaluation jobs (THE deployment metric)

Slurm scripts + logs for `../demo/eval_specksim_*.py` — raw events
through a full chip-emulated DynapcnnNetwork on earth-3 CPU nodes (no
GPU needed). **This is the only result that counts for deployment**;
binned accuracy is a training diagnostic.

`job_*.slurm` are parameterized via `--export`: `CKPT`, `SPLIT`
(val|test), `WM` (window mode: count|time), `MAXEV` (event cap —
wclamp-backbone models bad_alloc at any cap). Logs: `*.out` per job
id, `Slurm-<jobid>/` holds per-job metadata.

Current reference: `evT166_s2_ft` + 200K windowed vote = 76.67 val /
70.00 test (`win_1079369.out`, `win_1315373.out`).
