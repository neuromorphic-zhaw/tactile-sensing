# Tactile Sensing on Speck2f — Progress Report

## TL;DR
A trained 9-core spiking CNN is **deployed and classifying on the
physical Speck2f chip**: flat vs curved surface contact, ~0.6 s
decision latency, on-chip compute only. The
**chip-emulated event-level eval** — the honest deployment metric —
which revealed the frame-trained model loses ~20–40 pts of accuracy
under real event dynamics. Current best on-silicon recipe:
**count-bin model `evT166_s2` + windowed majority vote = 73.3% val /
70.0% test event-level.**

---

## ⛔ The metric that matters

Two different evaluations now exist — do not mix them:

| Eval | What it is | Verdict |
|---|---|---|
| Binned `.npz` val/test | frame-domain training diagnostic | does **not** transfer to chip |
| **specksim event-level** (`demo/eval_specksim_*.py`) | raw events → chip-emulated 9 cores → async spikes | **the deployment scoreboard** |

All "best model" claims below are stated **event-level**; binned
numbers are kept only as training diagnostics.

## 1. Hardware in use

| Item | Value |
|---|---|
| Chip | SynSense Speck2f devkit (USB `337d:5bca`) |
| Sensor | integrated 128×128 DVS event camera |
| Compute | 9 DYNAP-CNN spiking cores + readout unit |
| Weight format | 8-bit quantized, on-chip SRAM |
| Host role | USB transport + display only — **zero host inference** |

## 2. Model — exact numbers

**182,290 parameters**, IAF spiking neurons (`PeriodicExponential`,
θ=1.0, `min_v_mem=-1.0`).

| Layer | Conv | Kernel weights | Neuron mem | Core |
|---|---|---|---|---|
| 1 | 2→16, k4 s2 | 512 | 65,536 | 0 |
| 2 | 16→64, k4 s2 | 16,384 | 65,536 | 1 |
| 3 | 64→32, k4 s2 | 32,768 | 8,192 | 3 |
| 4 | 32→32, k3 s1 | 16,384 | 8,192 | 2 |
| 5 | 32→32, k4 s2 | 16,384 | 2,048 | 8 |
| 6 | 32→64, k4 s2 | 32,768 | 1,024 | 4 |
| 7 | 64→64, k3 | 65,536 | 1,024 | 6 |
| 8 | 64→64, k3 | 65,536 | 1,024 | 5 |
| 9 | AvgPool + Linear(→2) | 512 | 2 | 7 |

Mapper: `{1:0, 2:1, 3:3, 4:2, 5:8, 6:4, 7:6, 8:5, 9:7}` — all 9 cores
occupied; layers 7/8 sit at **100% of a core's 64K kernel memory**.
A greedy capacity sweep confirmed this is the largest taper that fits.

## 3. Training — exact numbers

| Item | Value |
|---|---|
| Dataset | 200 clips (100 flat / 100 curved), padded to uniform T |
| Input | `(1331, 2, 128, 128)` — events binned at 10 ms; **+ count-bin family** (`evT166` = 166 equal-event-count bins/clip) |
| Split | 70/15/15, seed 42 |
| Forward | `x[:, ::4]` → 333 effective steps, mean rate logits |
| Loss | CE + label smoothing 0.1; AdamW + cosine, 100 ep, clip 1.0 |
| Sweep size | ~30 runs (10 ms) + 24 variants (5/2/1 ms + raw-event) + 9 count-bin + 8 SNN+SSM hybrids |

**Two champions, two questions:**
- Binned-sim champion: `wclamp_v93.33_t86.67.pt` — mean 90.0 — but
  collapses event-level (§5a). Still the **physically deployed** ckpt.
- **Deployment champion: `evT166_s2`** (count-bin, ::2) — binned
  86.67v/83.33t, and the only model that survives event emulation.

## 4. Quantization — the concrete mechanism

1. Train in **float32** (no QAT needed)
2. At deploy: `DynapcnnNetwork(discretize=True)` → weights → **int8**,
   neurons → on-chip IAF dynamics
3. **The trap**: chip weights are scaled by spike threshold θ=1.0 —
   trained |W| > θ gets **clipped** at quantization → silent on-chip
   accuracy loss
4. **Why wclamp won (in frames)**: training clamps |W| < 1.0; measured
   max weight ≈ **0.41** → quantized accuracy ≈ float accuracy, zero
   clipping
5. **Verified (chip-sim, `eval_quantized.py`)**: the int8 model gives
   **bit-identical predictions** to float32 — 93.3% val / 86.7% test /
   90.5% all-200, same per-clip answers. Quantization loss: **zero**.
   (This proves int8 ≠ the accuracy problem — the gap is
   frame→event dynamics, not weight precision.)

## 5. How the SNN actually learns — BPTT + surrogate gradients

**The neuron model.** Every conv layer is followed by an
**Integrate-And-Fire** unit (sinabs `IAFSqueeze`): each neuron keeps a
membrane potential `v_mem`, integrates incoming weighted spikes, and
fires a spike when `v_mem ≥ θ = 1.0`, then subtracts θ (reset).
`min_v_mem = -1.0` bounds the floor. This is the exact circuit the
DYNAP-CNN cores implement in analog — simulation and hardware compute
the *same* neuron dynamics.

**The temporal dimension.** A clip is `(B, T=1331, 2, 128, 128)`.
The forward pass strides time `x[:, ::4]` → **T' = 333 timesteps**;
each timestep is pushed through the conv+IAF stack carrying membrane
state across steps (spikes depend on history — that's the "temporal" in
an SNN). Readout = mean output spike rate over the 333 steps → 2 logits
→ cross-entropy + label smoothing 0.1.

**BPTT — backprop through time.** The 333-step unrolled computation is
one differentiable graph: loss gradients flow backward through every
timestep and through the membrane dynamics. It's the same math as
training an RNN — each IAF state `v_mem[t]` feeds `v_mem[t+1]`, so the
optimizer sees how this timestep's spikes were produced by the
accumulated event history.

**The spike is non-differentiable — enter surrogate gradients.**
The firing function is a Heaviside step: gradient 0 almost everywhere,
undefined at θ. Standard autograd can't flow through it. Fix: keep the
hard step in the forward pass, but in the backward pass replace its
gradient with a smooth surrogate — sinabs **`PeriodicExponential`**
(a periodic bump around θ). The network behaves as a hard SNN forward
while gradients still propagate backward — the standard trick for
training spiking nets.

**Precision in practice**: AMP (`torch.cuda.amp` scaler) + grad-clip
1.0 + the wclamp recipe clamping |W| < θ so quantization is lossless
(section 4).

**Simulation vs hardware — why they're NOT equivalent.**
In simulation the network consumes discretized 10 ms bins — and every
bin is **binarized** (`clamp(0,1)`): a burst of 8 events looks
identical to 1. On the chip there is no clock and no clamp — 8 real
events = 8 synaptic kicks. Event-level emulation shows hidden layers
fire **2.3–4.5× more** under real event streams than under the binned
playback of the same clip; the asymmetric readout weights (w1 +0.96 vs
w0 −0.70) turn that global inflation into a **curved-class bias** —
the flat class collapses to ~21%. That is the sim→chip gap in one
sentence, and it is invisible to any binned metric.

## 5a. Event-level scoreboard — the real ranking

`demo/eval_specksim_events.py` streams raw `.h5` events through a full
`SpecksimNetwork` chip emulation (discretized layers, async queues,
C++ backend). 30-clip splits:

| Model / recipe | Event-level val | Event-level test | Note |
|---|---|---|---|
| **evT166_s2 + windowed vote** (200K-ev windows, fresh state) | **73.33%** | **70.00%** | 🏆 deployment champion |
| evT166_s2, plain clip argmax | 56.67% | 73.33% | same ckpt, weaker readout |
| wclamp + readout rescale c0.5 | 73.3% | — | SynSense rescale trick; flat-heavy, caps ~73% |
| wclamp + rescale c0.6 | 70.0% | — | most balanced rescale |
| ssm_evT166_frz (SNN+SSM head) | 63.33% | 46.67% | host-side S4D readout — degenerates all-flat on test |
| wclamp (deployed ckpt) | 46.67% | 63.33% | flat class → 21% |
| raw-event `ev_*` family | 36–47% | — | all fail; some unrunnable |

Unrunnable in emulation (intrinsic spike amplification → `bad_alloc`/hang):
`ev_base`, `evT332_s4`, `ev_bin5ms`, `ev_mixres`, `ssm_t10ms*`.
**In flight on HPC**: `l8` feature-extractor SSM variants (64-dim
last-conv spikes → SSM head, binned t76.67 both families) and
`wclamp_evT166ft` (v96.67 — wclamp transfers strongly to count-bin) —
these are the candidates to beat 73.33.

**Why the model is a sequence classifier**: class information lives in
contact *dynamics* (approach → wiggle/slip → liftoff), not in still
frames — a held touch emits ~no events. Per-window decisions are
therefore inherently unstable; the winning readout aggregates evidence
over windows (200K-event majority vote), and the SSM head tries to
learn that aggregation.

## 6. On-chip deployment path

```
checkpoint(.pt)
  → rebuild net (channels inferred from weight shapes)
  → DynapcnnNetwork.to('speck2fdevkit:0')   # auto-map + apply config
  → monitor taps: ["dvs", -1]               # raw sensor + readout spikes
  → live events over USB
```

**Config fixes that were required on real hardware:**

| Problem | Fix |
|---|---|
| DVS monitor silent under model config | `dvs_layer.raw_monitor_enable = True` |
| Noise "lines" + ~450 spikes/s idle | `dvs_filter.enable = True` (on-chip hot-pixel + background filter, `threshold=5`) |
| `make_config` API error | `slow_clk_frequency` belongs to `.to()` |

**To run the volatile deploy** (clears on unplug — rerun after replug):

```bash
cd demo && python live_dvs.py \
  --model ../deploy_checkpoints/wclamp_v93.33_t86.67.pt --window 0.3
```

**How the model runs — volatile deploy vs flash.** What the GUI uses
today is a **volatile deploy**: `hw.to()` pushes the quantized config to
the chip's registers over USB — the laptop writes it **once at startup**
and the chip then runs fully autonomously. The only difference vs
flash is persistence:

| | Volatile deploy (current) | Flashed (`--flash`) |
|---|---|---|
| Model lives in | chip registers/SRAM | chip flash |
| Survives power-off | no — cleared on unplug | yes — boots into it |
| Needs laptop | once, to write config | never |
| Where compute runs | **on-chip — identical in both** | on-chip |

So the system is already "flash-equivalent while powered" — the only
thing `--flash` adds is surviving a power cycle.

**Who computes what.** The chip's answer is a stream of **output
spikes**: the readout layer has 2 neurons (one per class); when a
neuron's membrane crosses θ it emits a `Spike` event tagged
`feature=0/1`, streamed over USB. Classification = counting which
feature line fires faster — there is no host-side forward pass.

| On the chip (all compute) | On the laptop (observation only) |
|---|---|
| sensor event encoding | spike counting + label display |
| all 9 conv+IAF layers | DVS/power rendering (samnagui) |
| output feature spikes (the inference) | baseline/debounce → FLAT/IDLE/CURVED |

Proof it's the chip computing: the spikes arrive as hardware `Spike`
events from the chip's source node; the on-chip power rails
(io/ram/logic/vddd/vdda) respond to activity; killing the GUI leaves
the chip classifying — unplugging USB gives the PC *nothing*.

## 7. Measured live performance

Numbers from actual on-chip runs (wclamp deployed):

| Metric | Idle | Touch |
|---|---|---|
| Output spikes (per feature) | **~0/s** (filter on) | **2,000–10,000/s** |
| DVS events in | ~500/s raw | up to ~17,000/s |
| Claim latency | — | **~0.6 s** |
| FLAT→IDLE→CURVED transition | — | **~1.2 s** |
| Power | on-chip rails (io/ram/logic/vddd/vdda), mW — *to re-measure with devkit attached* | |

**Live-behavior caveat (still open):** the deployed
wclamp output is **biphasic** — sustained holds fire ~1:1 on both
readout lines while onsets/transients produce strongly directional
runs. Consistent with the sequence-classifier finding (§5a): a still
hold is out-of-distribution for the model. The episode-level readout
below handles this in the demo, but real-world touch protocol
(speed/pressure/wiggle) differs from the dataset's and live verdicts
are less stable than the 73% eval suggests.

## 8. Readout decision logic (application layer)

Current `live_dvs.py` — leaky integrator on signed evidence,
calibrated on background only:

- **Signed log-ratio** `lr = log((cf+1)/(cc+1)) − idle_bias` per 300 ms
  window — ~1:1 ambiguous windows contribute ≈0, only directional
  bursts vote
- **Burst cap** — liftoff/approach mega-bursts (15–60k spikes/0.3 s)
  excluded from integration; score decays through them
- **Dead band** |score| < 2.5 → hold previous label (ties → default,
  same as SynSense's JitMajorityReadout)
- **IDLE calibration only** — activity floor + direction bias measured
  with contact removed; no per-class phases to mislabel
- At eval time the equivalent is the **windowed majority vote**
  (§5a): 200K-event windows, fresh neuron state, majority across
  windows — the recipe that posts 73.33/70.00.

## 9. Where we are vs the plan

| Milestone | Status |
|---|---|
| Speck-safe 9-core architecture | **done** — solver-verified |
| Trained model ≥90% (binned) | **done** — 90.0 mean (diagnostic only) |
| 8-bit quantized deploy | **done** — running on chip, bit-identical |
| Real-time GUI demo | **done** — samnagui DVS+power + prediction |
| Standalone flash binary | pipeline ready (`--flash`) |
| **Event-level (chip-emulated) validation** | **done — this is the comparative protocol**; revealed the sim→chip gap |
| Close the sim→chip gap ≥ demo-grade | **in progress** — evT166+win-vote 73.3/70.0; l8-SSM + count-bin fine-tunes in flight |
| Physical A/B: same clip → specksim vs silicon | **next gate** — proves specksim = truth machine |
| Force-sensor integration | late Oct–Nov |
| Thesis | Dec → deadline **8 Jan 2027** |

## 10. Repo state

- `deploy_checkpoints/` — deploy-candidate `.pt` files + leaderboard
  (binned ranking; event-level scoreboard lives in §5a + KB
  `model_registry.md`)
- `demo/live_dvs.py` — main live GUI (samnagui DVS+power + tk predict)
- `demo/live_ssm.py` — live demo with host-side SSM readout
- `demo/eval_specksim_{events,ssm}.py` — **the deployable evaluator**
- `demo/simple_gui.py`, `demo/calibrate.py` — fallback + noise meter
- `speck_connect/dvs_power_out.py` — original sensor viewer
- Sweeps: `devin_10ms/` (done), `devin_*/` multi-res + raw-event (done),
  `devin_evcount/` count-bin (done + fine-tunes running),
  `devin_ssm/` SNN+SSM hybrids (l8 variants in flight)
- `specksim_eval/`, `specksim_sanity/`, `quant_eval/` — slurm jobs + logs
- both branches (`ronald`, `main`) synced at HEAD
