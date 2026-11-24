# speck_toolkit — reusable Speck2f building blocks

Everything in here is **project-agnostic**: the scripts and patterns verified
on the Speck2f devkit during the tactile-sensing project, kept separate from
`curvature/` (the finished D1–D4 project) so any future work can reuse them
directly.

| File | What it does |
|---|---|
| `speck_wsl_guide.txt` | WSL2 + usbipd setup steps for the devkit |
| `speck_dvs_output.py` | Minimal: open devkit → raw DVS event stream |
| `dvs_power_out.py` | Live samnagui GUI: 128×128 DVS activity + 5-rail power traces (the reliable pure-converter path) |
| `model.py` | The verified 9-core deployable architecture: conv+IAF backbone, `IAFSqueeze` neurons, checkpoint loader (`v_mem` stripping) |
| `verify_convert.py` | Full deployment proof **without hardware**: DynapcnnNetwork conversion → memory summary → real core mapping → samna config. Prints PASS/FAIL |
| `deploy_speck.py` | Checkpoint → chip deploy + Python windowed readout (self-contained model rebuild) |
| `replay_chip.py` | Replay a recorded `.h5`/`.npz` event clip through real silicon — the physical A/B validation tool |
| `live_visualizer.py` | samnagui live GUI helper (inference + DVS + power) |

## Devkit attach (WSL)

```powershell
usbipd list                                   # find Speck2fDevKit busid
usbipd attach --wsl --busid <BUSID>           # from elevated PowerShell
```

```bash
lsusb | grep 337d                             # verify device present
sudo chmod 666 /dev/bus/usb/*/*               # if samna can't open it
```

Python env: torch + sinabs + samna (see `../requirements.txt`). If the
device wedges ("Apply configuration timeout"), unplug/replug USB.

## The verified deployable architecture

`model.py` carries the reference: 8× `Conv2d(bias=False)` + `IAFSqueeze`
+ `AvgPool`/`Linear` readout → 9 DYNAP-CNN layers → 9 cores. The rules
that make a model survive `DynapcnnNetwork` conversion:

- `bias=False` everywhere; **no** BatchNorm, Dropout, MaxPool (fold or
  remove before conversion)
- `IAFSqueeze` (pure IAF): θ=1.0, `min_v_mem=−1.0`, no leak, MultiSpike +
  MembraneSubtract; surrogate `PeriodicExponential` or `SingleExponential`
- Per-core memory: neuron mem ≤ 16/32/64 Ki, kernel mem ≤ 16/32/64 Ki
  depending on the core — the MAX taper `(16,64,32,32,32,64,64,64)` is the
  largest fitting shape; auto-mapper finds it
- Checkpoints: strip `.v_mem` keys (keep `min_v_mem`), `strict=False`
- Whole-model `.pth` needs `weights_only=False`; state_dict `.pt` is fine

## Workflow for a new project

```bash
# 1. see the sensor
python speck_dvs_output.py            # raw events
python dvs_power_out.py               # DVS + power GUI

# 2. prove your checkpoint is deployable (no hardware needed)
python verify_convert.py --model <ckpt>.pt

# 3. run on chip
python deploy_speck.py --model <ckpt>.pt

# 4. validate against recorded data
python replay_chip.py --model <ckpt>.pt --clip <clip.h5>
```

## Hard-won quirks (don't rediscover these)

- **Filtered monitor emits ZERO events** — only
  `dvs_layer.raw_monitor_enable=True` gives the sensor stream.
- **samnagui must be imported after samna**, and the reliable viz path is
  pure converters (`Speck2fDvsToVizConverter` / `MeasurementToVizConverter`
  → `VizEventStreamer`). JIT/visualizer nodes can hang ~40 s compiling or
  die silently in WSL.
- **`dvs_filter` starves models trained on unfiltered events** — leave it
  off for inference, or retrain with it.
- External **slow clock (~1 kHz)** is required for flicker filter/readout
  features — pass `slow_clk_frequency` to `.to()`.
- `.to(device)` writes config immediately; killing mid-deploy can wedge
  the chip.
- Specksim event-level eval (see `curvature/demo/eval_specksim_*.py`) is
  the only trustworthy deployment metric — binned accuracy does NOT
  transfer to silicon.

For a guided data collector (phase-tagged `.h5` recording with live event
counter), see `../texture_dot_ab/record_touch.py` — it generalizes to any
`--classes` list.
