"""Event-level (chip-faithful) validation of a checkpoint.

Streams the RAW event files (x, y, t, p at microsecond resolution)
through DynapcnnNetwork(discretize=True) — every event is its own
timestep, exactly as the DVS input layer on the physical Speck would
receive it. DYNAP-CNN IAF layers have no leak, so processing only
non-empty steps is exact; coincident-timestamp events are grouped.

Runs on the VAL split only (same seed-42 membership convention as
eval_quantized.py). Requires a GPU — a clip is ~740K sequential steps.

    python eval_quantized_events.py --model ../deploy_checkpoints/wclamp_v93.33_t86.67.pt
"""

import argparse
import glob
import os
import time
import numpy as np
import torch
import torch.nn as nn
import sinabs
import sinabs.layers as sl
from sinabs.backend.dynapcnn import DynapcnnNetwork

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SpeckDemoTrainer, load_checkpoint

RAW_DATA = "/cfs/earth/scratch/matheron/tactile-sensing/data/dataset_curv_events"


def split_indices(n):
    tr = int(0.70 * n)
    va = int(0.15 * n)
    gen = torch.Generator().manual_seed(42)
    tr_i, va_i, te_i = torch.utils.data.random_split(
        range(n), [tr, va, n - tr - va], generator=gen)
    return set(tr_i.indices), set(va_i.indices), set(te_i.indices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=RAW_DATA)
    ap.add_argument("--chunk", type=int, default=2048,
                    help="events per forward call (memory/speed tradeoff)")
    ap.add_argument("--clip", default=None,
                    help="single clip basename (e.g. 2026-03-17_flat_01.npz)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_events", type=int, default=0,
                    help="truncate each clip at N events (smoke tests)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cpu" if args.cpu else "cuda")

    files = sorted(glob.glob(os.path.join(args.data, "**/*.npz"), recursive=True))
    print(f"{len(files)} raw clips")
    if args.clip:
        sel = [i for i, f in enumerate(files)
               if os.path.basename(f) == args.clip]
        assert sel, f"{args.clip} not found"
    else:
        _, va_i, _ = split_indices(len(files))
        sel = sorted(va_i)
    if args.limit:
        sel = sel[:args.limit]
    print(f"{len(sel)} clip(s) selected")

    # --- quantized chip-simulation ---
    sd = torch.load(args.model, map_location="cpu")
    channels = tuple(sd[f"spiking_backbone.{i * 2}.weight"].shape[0]
                     for i in range(8))
    model = SpeckDemoTrainer(channels=channels, batch_size=1)
    with torch.no_grad():
        _ = model(torch.zeros((1, 40, 2, 128, 128)))
    n_w = load_checkpoint(model, args.model)
    print(f"loaded {n_w} weights, channels={channels}")

    hw = DynapcnnNetwork(snn=model.spiking_backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)
    # workaround: .to() iterates self._merge_points which only exists when
    # the model has merge layers — stub it so device transfer works
    if not hasattr(hw, "_merge_points"):
        hw._merge_points = {}
    hw.to(dev)
    print(f"DynapcnnNetwork(discretize=True) on {dev} — event-stream mode")

    def _reset_quant():
        for layer in hw.dynapcnn_layers.values():
            for m in layer.modules():
                if isinstance(m, sl.IAFSqueeze):
                    m.v_mem.data = torch.zeros_like(m.v_mem)

    def predict_events(x, y, t, p):
        """Stream the event train through the chip sim; returns (pred, rates)."""
        _reset_quant()
        E = len(x)
        outs = []
        for s in range(0, E, args.chunk):
            n_ev = min(args.chunk, E - s)
            # (n_ev, 2, 128, 128): each timestep = the events at that instant
            inp = torch.zeros(n_ev, 2, 128, 128)
            ch = p[s:s + n_ev].astype(np.int64)
            inp[np.arange(n_ev), ch,
                y[s:s + n_ev].astype(np.int64),
                x[s:s + n_ev].astype(np.int64)] = 1.0
            with torch.no_grad():
                o = hw(inp.to(dev))          # (n_ev, 2) output spikes
            outs.append(o.float().cpu())
        rates = torch.cat(outs).mean(dim=0)
        return int(rates.argmax().item()), rates.tolist()

    n_ok, per_clip = 0, []
    t0 = time.time()
    for j, i in enumerate(sel):
        with np.load(files[i]) as z:
            x, y, t, p = z['x'], z['y'], z['t'], z['p']
            label = int(z['label'])
        if args.max_events:
            x, y, t, p = x[:args.max_events], y[:args.max_events], \
                         t[:args.max_events], p[:args.max_events]
        span_s = (t[-1] - t[0]) / 1e6
        pred, rates = predict_events(x, y, t, p)
        ok = pred == label
        n_ok += ok
        per_clip.append((os.path.basename(files[i]), label, pred, len(x), span_s))
        el = (time.time() - t0) / 60
        print(f"  [{j+1}/{len(sel)}] {per_clip[-1][0]} "
              f"E={len(x)} span={span_s:.1f}s "
              f"pred={pred} tgt={label} {'OK' if ok else 'XX'} "
              f"rates={[round(r,4) for r in rates]} ({el:.1f} min)",
              flush=True)

    n = len(sel)
    print(f"\n== EVENT-STREAM VAL ({n} clips) ==")
    print(f"  quant event-level : {n_ok}/{n} = {100*n_ok/n:.2f}%")
    print(f"  total events      : {sum(c[3] for c in per_clip):,}")
    print(f"  mean clip span    : {np.mean([c[4] for c in per_clip]):.1f}s")


if __name__ == "__main__":
    main()
