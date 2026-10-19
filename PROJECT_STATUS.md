# Tactile Sensing on Speck2f — Progress Report
**Team meeting — 19 Oct 2026**

## TL;DR
A trained 9-core spiking CNN is **deployed and classifying on the
physical Speck2f chip right now**: flat vs curved surface contact,
~90% mean accuracy, ~0.6 s decision latency, on-chip compute only.

---

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
θ=1.0, `min_v_mem=-1.0` — same neuron model as the SynSense gesture demo).

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
| Input | `(1331, 2, 128, 128)` — events binned at 10 ms |
| Split | 70/15/15, seed 42 |
| Forward | `x[:, ::4]` → 333 effective steps, mean rate logits |
| Loss | CE + label smoothing 0.1; AdamW + cosine, 100 ep, clip 1.0 |
| Sweep size | ~30 runs (10 ms) + 24 variants (5/2/1 ms + raw-event) |

**Deployed checkpoint**: `wclamp_v93.33_t86.67.pt` — **val 93.3% /
test 86.7% (mean 90.0)**, best@epoch 47, rank #1 of the sweep.

## 4. Quantization — the concrete mechanism

1. Train in **float32** (no QAT needed)
2. At deploy: `DynapcnnNetwork(discretize=True)` → weights → **int8**,
   neurons → on-chip IAF dynamics
3. **The trap**: chip weights are scaled by spike threshold θ=1.0 —
   trained |W| > θ gets **clipped** at quantization → silent on-chip
   accuracy loss
4. **Why wclamp won**: training clamps |W| < 1.0; measured max weight
   ≈ **0.41** → quantized accuracy ≈ float accuracy, zero clipping

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

**Simulation vs hardware — why they're equivalent.** In simulation the
network consumes discretized 10 ms bins; on the chip there is *no*
clock stepping the net — events arrive asynchronously and neurons fire
when they cross θ. The sim bins are only a training-time convenience;
the deployed SNN runs the same spike-dynamics on the raw event stream.
That async operation is also the power argument: energy scales with
**SynOps** (actual spikes emitted), not MACs — silence costs nothing.

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

## 8. Readout decision logic (application layer)

- **Baseline subtraction** — 5 s idle calibration; classify on excess
  (model has ~450/s flat resting bias if unfiltered — a threshold
  alone can't separate touch from bias)
- **Debounce** — class must win 2 consecutive 300 ms windows → kills
  the liftoff event-burst that false-triggered CURVED
- **State machine** — FLAT→IDLE→CURVED only; direct flips rejected
  (physically can't touch both)
- Live-verified: flat press → FLAT, curved → CURVED, release → IDLE

## 9. Where we are vs the plan

| Milestone | Status |
|---|---|
| Speck-safe 9-core architecture | **done** — solver-verified |
| Trained model ≥90% | **done** — 90.0 mean |
| 8-bit quantized deploy | **done** — running on chip |
| Real-time GUI demo | **done** — samnagui DVS+power + prediction |
| Standalone flash binary | pipeline ready (`--flash`) |
| Comparative protocol (chip vs software) | next gate, mid–late Oct |
| Force-sensor integration | late Oct–Nov |
| Thesis | Dec → deadline **8 Jan 2027** |

## 10. Repo state

- `deploy_checkpoints/` — top-10 ranked .pt + leaderboard README
- `demo/live_dvs.py` — main live GUI (samnagui DVS+power + tk predict)
- `demo/simple_gui.py`, `demo/calibrate.py` — fallback + noise meter
- `speck_connect/dvs_power_out.py` — original sensor viewer
- 24-variant sweep dirs (`devin_5ms/2ms/1ms/rawevent`) + `gen_variants.py`
- both branches (`ronald`, `main`) synced at HEAD
