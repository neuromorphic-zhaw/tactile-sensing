"""Physical A/B: inject recorded dataset events into the real Speck2f.

Deploys the model with sensor input DISABLED (dvs_input=False) and
writes the clip's raw (x,y,t,p) event train through the model sink —
deterministic replay. Output spikes counted per event-chunk give a
per-phase class trajectory + a final verdict vs the clip label.

Since on-chip IAF is non-leaky, inter-event timing is irrelevant to
the computation — only the event sequence matters.

    python replay_chip.py --model ../deploy_checkpoints/wclamp_v93.33_t86.67.pt \
        --events /path/to/clip.npz
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import samna
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.io import (disable_timestamps,
                                      enable_timestamps, reset_timestamps)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SpeckDemoTrainer, load_checkpoint

CLASSES = {0: "FLAT", 1: "CURVED"}


def infer_channels(sd):
    return tuple(sd[f"spiking_backbone.{i * 2}.weight"].shape[0]
                 for i in range(8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--events", required=True,
                    help="npz with x,y,t,p arrays + label")
    ap.add_argument("--device", default="speck2fdevkit:0")
    ap.add_argument("--chunks", type=int, default=20,
                    help="split the event train into N chunks -> "
                         "per-chunk class trajectory")
    ap.add_argument("--max_events", type=int, default=1000000)
    args = ap.parse_args()

    # --- deploy: sensor OFF, events come from the host ---
    sd = torch.load(args.model, map_location="cpu")
    model = SpeckDemoTrainer(channels=infer_channels(sd), batch_size=1)
    with torch.no_grad():
        _ = model(torch.zeros((1, 40, 2, 128, 128)))
    load_checkpoint(model, args.model)

    hw = DynapcnnNetwork(snn=model.spiking_backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)

    def _no_sensor(cfg):
        # keep the DVSLayer event path (injected events reach cores)
        # but block the physical pixel array — deterministic replay
        cfg.dvs_layer.pass_sensor_events = False
        cfg.dvs_layer.monitor_enable = False
        cfg.dvs_layer.raw_monitor_enable = True   # count injected evs
        return cfg

    hw.to(device=args.device, chip_layers_ordering="auto",
          monitor_layers=["dvs", -1], config_modifier=_no_sensor)
    print(f"deployed {args.model} — map: {hw.chip_layers_ordering}",
          flush=True)
    # .to() already wired samna_input_buffer -> model sink (running
    # graph) and model source -> samna_output_buffer
    in_buf = hw.samna_input_buffer
    out_buf = hw.samna_output_buffer

    # --- load the clip's raw events ---
    with np.load(args.events) as z:
        x, y, t, p = z["x"], z["y"], z["t"], z["p"]
        label = int(z["label"]) if "label" in z else -1
    n = min(len(x), args.max_events) if args.max_events else len(x)
    x, y, t, p = x[:n], y[:n], t[:n], p[:n]
    t = t - t[0]
    print(f"{n} events ({(t[-1]-t[0])/1e6:.1f}s), label={label} "
          f"({CLASSES.get(label, '?')})", flush=True)

    events = []
    for i in range(n):
        e = samna.speck2f.event.DvsEvent()
        e.x = int(x[i]); e.y = int(y[i])
        e.p = bool(p[i]); e.timestamp = int(t[i])
        events.append(e)

    # use sinabs' own buffers — same graph .to() started
    sink = out_buf
    src = in_buf

    # the FPGA replays events honoring their µs timestamps ONLY in
    # timestamp mode (hw_forward does this) — without it injected
    # events are dropped
    _ = sink.get_events()
    reset_timestamps(hw.device)
    enable_timestamps(hw.device)

    # paced writes: small slices slept by their own timestamp span —
    # a single mega-write may overrun the FPGA-side replay queue
    _ = sink.get_events()
    print(f"replaying {n} events...", flush=True)
    run_f = run_c = run_d = 0
    SLICE = 500
    for s in range(0, n, SLICE):
        src.write(events[s:s + SLICE])
        span = (int(t[min(s + SLICE - 1, n - 1)]) - int(t[s])) * 1e-6
        time.sleep(max(0.005, span * 0.9))
        outs = sink.get_events()
        cf = sum(1 for e in outs if getattr(e, "feature", -1) == 0)
        cc = sum(1 for e in outs if getattr(e, "feature", -1) == 1)
        nd = sum(1 for e in outs
                 if hasattr(e, "x") and not hasattr(e, "feature"))
        run_f += cf; run_c += cc; run_d += nd
        if (cf or cc) and s % (SLICE * 20) == 0:
            print(f"  ev{s:6d}  +{cf:5d}f +{cc:5d}c   "
                  f"run {run_f} vs {run_c}", flush=True)
    time.sleep(3.0)                       # let the tail flush
    outs = sink.get_events()
    run_f += sum(1 for e in outs if getattr(e, "feature", -1) == 0)
    run_c += sum(1 for e in outs if getattr(e, "feature", -1) == 1)

    pred = 0 if run_f > run_c else 1
    print(f"\nVERDICT: {CLASSES[pred]} ({run_f} vs {run_c})  "
          f"label={CLASSES.get(label, '?')}  "
          f"{'OK' if pred == label else 'WRONG'}", flush=True)
    try:
        disable_timestamps(hw.device)
    except Exception:
        pass


if __name__ == "__main__":
    main()
