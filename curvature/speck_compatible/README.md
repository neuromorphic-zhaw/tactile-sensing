# speck_compatible — the model zoo (stage 1: flat vs curved)

All models that fit the Speck2f 9-core DynapCNN constraint (verified
`model.py` topology, |W|≤1 weight-clamp → lossless int8 discretization).

## How the family dirs differ — the naming axis is INPUT RASTERIZATION

| Family | Input format | Why it exists |
|---|---|---|
| `devin_1ms/` `devin_2ms/` `devin_5ms/` `devin_10ms/` | **time-binned**: events rasterized into N-ms frames | The original sweep. `10ms` was the binned-era champion family (wclamp v93.33) — but binned ≠ deployable |
| `devin_evcount/` | **count-binned**: each bin holds ~equal event count | Deployment-natural ("decide every N events", no wall clock). **The champion family lives here** |
| `devin_rawevent/` | **raw event stream**, no rasterization | Parity anchors vs ANN literature. All dead ends — specksim deadlock/bad_alloc or low accuracy |
| `devin_ssm/` | backbone + **S4D state-space head** | Hybrid: 9-core SNN on-chip + host-side temporal readout. Failed event-level so far (46.67–63.33%) |

## Variant naming inside each family

- `evT166_s2` = **T**166 bins/clip, forward stride ::**2** (→83 steps)
- `evN2000_s8` = **N**2000 events-per-bin (bin count instead of bin width), stride ::8
- `wclamp*` = |W|<1 weight-clamped (deploy-safe); `wclamp_evT166ft` = wclamp backbone fine-tuned onto count-bin data
- `*_ft` = fine-tuned warm-start from a named ckpt; `*_frz` = frozen backbone; `*_l8` = last-conv (64-ch) feature mode; `*_l8ft` = l8 + joint fine-tune
- `v<val>_t<test>` in dir names = baked binned scores (diagnostic only!)

Each variant dir: `README.txt` (the recipe diff), `train.py`, `job.slurm`,
`run_<jobid>/` (ckpt + logs — gitignored). Deployment candidates are
promoted into `../deploy_checkpoints/` (git-tracked).

## Verdicts (event-level specksim — the only metric that counts)

- **Champion**: `devin_evcount/evT166_s2_ft` + 200K windowed vote = **76.67 val / 70.00 test**
- Runner-up: `devin_evcount/evT166_s2` + vote = 73.33/70.00
- Unrunnable in specksim: `devin_rawevent/*`, `evT332_s4` (deadlock), all wclamp-backbone SSMs (bad_alloc)
