# tactile-sensing — "Speck-touch"

An ultra-low-power, event-based tactile sensor: a **markerless GelSight**
elastomer watched by the integrated DVS of a **SynSense Speck2f** devkit.
A 9-core spiking CNN classifies contact-surface **curvature (flat vs
curved)** entirely on-chip — the host is USB transport + display only.

*ZHAW × SynSense — Neuromorphic Hand project. Deliverables D3 (software
& models) + D4 (live demo); contract scope = curvature detection.*

---

## ⛔ The one metric that matters

> **All model claims are made on the specksim event-level eval** —
> `phase1_curvature/demo/eval_specksim_*.py` runs raw events through a full chip-emulated
> `SpecksimNetwork` (discretized 9 cores, async spikes, C++ sim). Binned
> `.npz` val/test numbers are *training diagnostics only*; they do not
> transfer to the chip. A model is "better" only if it wins event-level.

## Current results (event-level, 30-clip split)

| Recipe | Event-level score |
|---|---|
| **evT166_s2 + windowed majority vote** | **73.33% val / 70.00% test — champion** |
| wclamp + readout rescale c0.5 | 73.33% val (flat-biased) |
| SNN+SSM hybrid (`ssm_evT166_frz`) | 63.33% val / 46.67% test |
| wclamp (deployed on devkit) | 46.67% val / 63.33% test |
| raw-event `ev_*` family | 36–47% val — all fail event-level |

Physical deployment status: `wclamp_v93.33_t86.67.pt` runs live on the
devkit via `phase1_curvature/demo/live_dvs.py` (real-time flat/curved GUI, ~0.6 s
decision latency, zero host inference). Closing the sim→chip gap is the
active workstream — see `PROJECT_STATUS.md`.

## Repository layout

The repo mirrors the three thesis stages:

| Stage | Project dir | Status |
|---|---|---|
| 1 — flat vs curved | `phase1_curvature/` | ✅ complete (deliverables D1–D4) |
| 2 — dot vs markerless gel A/B | `phase2_texture_dot_ab/` | 🟡 active — data collection next |
| 3 — force-sensor integration | `force/` (planned) | ⬜ upcoming |

`speck_toolkit/` holds the project-agnostic Speck2f building blocks
(start there for any new work). Each stage keeps its own project dir at
top level; all datasets live centrally under `data/`.

| Path | Contents |
|---|---|
| `speck_toolkit/` | Reusable Speck2f toolkit: devkit attach/view scripts, samnagui DVS+power GUI, verified 9-core `model.py`, `verify_convert.py`, `deploy_speck.py`, `replay_chip.py` — see its README |
| `phase1_curvature/speck_connect/` | `collect_gzip_curv.py` — the D2 dataset collection GUI (.h5+gzip) |
| `data/` | Strict phase separation: `phase1_curvature/` (dataset_curv 200 .h5, events, time/count bins — top-level names symlink here); `phase2_texture_dot_ab/` (`dataset_tex/{dot,nodot}/<class>/*.h5` + `labels.json`, `dataset_tex_evT166/` derived) |
| `phase1_curvature/speck_compatible/` | Training: `devin_10ms/` (champion sweep), `devin_evcount/` (count-bin), `devin_rawevent/`, `devin_1/2/5ms/`, `devin_ssm/` (SNN+SSM hybrids); `live_status.sh` |
| `phase1_curvature/demo/` | Deployment + eval: `live_dvs.py` (on-chip live demo), `live_ssm.py` (host-side SSM readout), `eval_specksim_events.py` / `eval_specksim_ssm.py` (**the deployable evaluator**), `eval_quantized*.py`, `model.py` (model defs + ckpt loader) |
| `phase1_curvature/deploy_checkpoints/` | Deployment-candidate `.pt` files, `v<val>_t<test>` naming; rescale variants `*_c*`; see its README |
| `phase1_curvature/specksim_eval/` | Slurm jobs + logs for event-level evals |
| `phase1_curvature/non-speck-compatible/` | Legacy pre-Speck-constraint experiments (kept for reference) |
| `phase1_curvature/eval_cm.py`, `quant_eval/`, `specksim_sanity/` | Early quantized/binned eval + sanity checks |
| `phase2_texture_dot_ab/` | Stage-2: multi-class texture classification (flat/curved/edge/soft/grid) as a controlled A/B — GelSight **with marker dots vs markerless**. Spacebar recorder (`record_touch.py`), batch collector (`collect_tex.py`), evT166 prep (`prep_count_bins.py`), N-class trainer (`train_tex.py` → `runs/`), paired eval (`eval_compare.py`). See `phase2_texture_dot_ab/README.md` |
| `PROJECT_STATUS.md` | Detailed progress report (architecture, quantization, deploy notes) |
| `LIVE_STATUS.md` | Auto-updating training dashboard snapshot |

## The model

182,290-param spiking CNN — 8 conv+IAF layers + Linear readout, MAX
taper (largest that fits the heterogeneous core memory table), pure
IAF neurons (θ=1.0, `min_v_mem=−1.0`, MultiSpike, MembraneSubtract). Occupies
all 9 DYNAP-CNN cores; mapper `{1:0,2:1,3:3,4:2,5:8,6:4,7:6,8:5,9:7}`.
Trained with BPTT + `PeriodicExponential` surrogate; `|W|<1` clamp
makes int8 discretization **lossless** (verified bit-identical).

Hybrid variants (`phase1_curvature/speck_compatible/devin_ssm/`): frozen backbone on-chip + small S4D
state-space readout head host-side — a learned replacement for the
windowed majority vote. `FEAT_MODE=l8` feeds the SSM 64-dim last-conv
feature spikes (SNN as pure feature extractor) instead of 2-dim
readout spikes.

## Quick start

```bash
# train a variant (HPC, GPU partition)
cd phase1_curvature/speck_compatible/devin_ssm/ssm_evT166_l8 && sbatch job.slurm

# event-level eval — THE deployment validation (HPC, CPU partition)
cd phase1_curvature/specksim_eval && sbatch --export=ALL,CKPT=<ckpt.pt>,SPLIT=val,WM=count,MAXEV=500000 job_ssm_l8.slurm

# volatile deploy to a connected devkit (WSL host, USB passthrough)
cd phase1_curvature/demo && python live_dvs.py --model ../deploy_checkpoints/<ckpt>.pt --window 0.3

# host-side SSM readout demo
cd phase1_curvature/demo && python live_ssm.py --model ../deploy_checkpoints/ssm_*_frz*.pt --win_ms 40
```

Deploy path: `ckpt → rebuild net → DynapcnnNetwork.to('speck2fdevkit:0')`
(auto-map + discretize). Config quirks are documented in
`PROJECT_STATUS.md` §6 (dvs_filter, raw_monitor, slow clock).

## Dataset

**Download (Deliverable D2): see `DATASET.md`** — SharePoint link to the
200-clip HDF5 archive.

200 clips (100 flat / 100 curved) recorded with `collect_gzip_curv.py`;
raw `.h5` events are the master (`data/dataset_curv_events/`). Derived
rasterizations: time-binned (1–10 ms) and **count-binned** (evT166 =
166 equal-count bins/clip — deployment-natural, "decide every N events",
and currently the strongest recipe).

## Working agreements

- One dataset family per training variant — no mixing time-bins with
  count-bins.
- Chip discipline: `bias=False`, no BatchNorm/Dropout/MaxPool, `|W|<1`,
  IAF only — anything else is a simulation crutch and must be folded
  before deploy.
- Checkpoint names carry scores: `name_v<bestval>_t<test>.pt`.
- Event-level eval is gated by `--max_events` (in-network spike
  amplification can blow the sim's inter-layer queues; ≥250K usually
  lands, some ckpts are unrunnable — see `model_registry.md` in the KB).

## For reviewers

Start with `PROJECT_STATUS.md` (full technical report), then
`phase1_curvature/deploy_checkpoints/README.md`. The deep implementation reference
(training→discretization→samna graph→readout internals).
