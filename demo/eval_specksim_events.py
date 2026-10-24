"""Full chip-emulation (samna.specksim) validation of a checkpoint.

Builds the network as a samna EventFilterGraph of actual specksim filter
nodes (convolution, IAF, sumpool — the same primitives SynSense uses to
emulate Speck silicon) and streams the RAW event train (x,y,t,p at us
resolution) through it — the exact interface the physical chip exposes.

Prediction = argmax over output-spike counts per class feature.

    python eval_specksim_events.py --model ../deploy_checkpoints/wclamp_v93.33_t86.67.pt
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
import samna
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.specksim import (
    SpecksimNetwork, convert_convolutional_layer, convert_iaf_layer,
    convert_pooling_layer)

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


def load_events(path, max_events=0):
    with np.load(path) as z:
        x, y, t, p = z['x'], z['y'], z['t'], z['p']
        label = int(z['label'])
    if max_events:
        x, y, t, p = x[:max_events], y[:max_events], t[:max_events], p[:max_events]
    t0 = t[0]
    ev = np.zeros(len(x), dtype=[('x', np.uint32), ('y', np.uint32),
                                 ('t', np.uint32), ('p', np.uint32)])
    ev['x'], ev['y'] = x.astype(np.uint32), y.astype(np.uint32)
    ev['t'] = (t - t0).astype(np.uint32)          # us, rebased to 0
    ev['p'] = p.astype(np.uint32)
    return ev, label


def load_binned_events(path, stride=4):
    """Synthesize an event stream from the padded frames tensor — the
    binned-equivalent playback used to validate the specksim graph against
    the quantized eval. Each occupied (t,c,y,x) emits `count` events at
    the bin's timestamp. IAF has no leak so only order/count matter."""
    with np.load(path) as z:
        frames, label = z['frames'][::stride], int(z['label'])
    ts, ps, ys, xs = np.nonzero(frames)
    counts = frames[ts, ps, ys, xs]
    idx = np.repeat(np.arange(len(ts)), counts)
    ev = np.zeros(len(idx), dtype=[('x', np.uint32), ('y', np.uint32),
                                   ('t', np.uint32), ('p', np.uint32)])
    ev['t'], ev['p'] = (ts[idx] * 10000).astype(np.uint32), ps[idx].astype(np.uint32)
    ev['x'], ev['y'] = xs[idx].astype(np.uint32), ys[idx].astype(np.uint32)
    return ev, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=RAW_DATA)
    ap.add_argument("--split", choices=["val", "test", "all"], default="val")
    ap.add_argument("--clip", default=None, help="single clip basename")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_events", type=int, default=1000000,
                    help="truncate clips beyond this many input events — "
                         "multi-M-event clips overrun in-network queues "
                         "(bad_alloc); 0 = no cap")
    ap.add_argument("--n_class", type=int, default=2)
    ap.add_argument("--invert_p", action="store_true",
                    help="flip polarity channel (p = 1 - p) — sensitivity check")
    ap.add_argument("--binned_data", default="",
                    help="padded-frames dataset dir; synthesize events from "
                         "binned tensors instead of raw streams")
    ap.add_argument("--binned_stride", type=int, default=4)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "**/*.npz"), recursive=True))
    print(f"{len(files)} raw clips")
    if args.clip:
        sel_map = {"CLIP": [i for i, f in enumerate(files)
                            if os.path.basename(f) == args.clip]}
        assert sel_map["CLIP"], f"{args.clip} not found"
    else:
        _, va_i, te_i = split_indices(len(files))
        sel_map = {"VAL": sorted(va_i), "TEST": sorted(te_i),
                   "ALL": list(range(len(files)))}
        if args.split == "val":
            sel_map = {"VAL": sel_map["VAL"]}
        elif args.split == "test":
            sel_map = {"TEST": sel_map["TEST"]}
        else:
            sel_map = {"VAL": sel_map["VAL"], "TEST": sel_map["TEST"]}

    # --- build the chip emulation ---
    sd = torch.load(args.model, map_location="cpu")
    channels = tuple(sd[f"spiking_backbone.{i * 2}.weight"].shape[0]
                     for i in range(8))
    model = SpeckDemoTrainer(channels=channels, batch_size=1)
    with torch.no_grad():
        _ = model(torch.zeros((1, 40, 2, 128, 128)))
    n_w = load_checkpoint(model, args.model)
    print(f"loaded {n_w} weights, channels={channels}")

    # Discretize to the chip grid first (int8 weights + per-layer scaled
    # thresholds) — feeding FLOAT weights to specksim truncates them to ~0.
    hw = DynapcnnNetwork(snn=model.spiking_backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)

    def build_specksim(hw, input_shape):
        """specksim graph from the DISCRETIZED dynapcnn layers."""
        graph = samna.graph.EventFilterGraph()
        filters = [samna.BasicSourceNode_specksim_events_spike()]
        cur = tuple(input_shape)
        wscale = 1.0
        n_layers = len(hw.dynapcnn_layers)
        for li, (idx, dl) in enumerate(sorted(hw.dynapcnn_layers.items())):
            conv = iaf = None
            for m in dl.children():
                if isinstance(m, nn.Conv2d):
                    conv = m
                elif isinstance(m, sl.IAFSqueeze):
                    iaf = m
            f, cur = convert_convolutional_layer(conv, cur, wscale)
            filters.append(f)
            wscale = 1.0
            f, cur = convert_iaf_layer(iaf, cur)
            filters.append(f)
            if li == n_layers - 2:
                # AvgPool(2,2) sits between layer 8 and the readout conv;
                # specksim only supports SumPool -> compensate next conv x1/4
                f, cur = convert_pooling_layer(nn.AvgPool2d(2, 2), cur)
                filters.append(f)
                wscale = float(2 * 2)
        filters.append(samna.BasicSinkNode_specksim_events_spike())
        # longer drains: output spikes land well after input write() returns
        return SpecksimNetwork(graph, graph.sequential(filters),
                               initial_sleep_duration=5.0,
                               subsequent_sleep_duration=1.0)

    sim = build_specksim(hw, (2, 128, 128))
    # the sink's async read can finish before the last layer fires — monitor
    # the final IAF instead (read_monitor drains after graph stop = reliable)
    n_iaf = len(hw.dynapcnn_layers)
    sim.add_monitor(n_iaf - 1)
    print("SpecksimNetwork built from discretized layers — full chip emulation")

    def sim_run(ev, chunk=20000, pause=0.05, seg=200000):
        """Chunked + segmented forward. bad_alloc comes from inter-layer
        queue backlog growing faster than the graph drains — pace writes
        in 20K slices and HARD-SETTLE (drain to quiet) every 200K events.
        Graph stays alive between segments so v_mem state persists."""
        spikes = sim.xytp_to_specksim_spikes(ev)
        sim.network.start()
        mons = list(sim.monitors.values())
        for m in mons:
            _ = m["sink"].get_events()
            m["graph"].start()
        counts = np.zeros(args.n_class, dtype=np.int64)
        n_out = [0]  # set after stop (events flush to sinks then)

        def drain():
            # frees + counts whatever has propagated to the queues;
            # anything still buffered flushes at stop -> read_monitor
            for m in mons:
                for e_ in m["sink"].get_events():
                    p_ = getattr(e_, "feature", getattr(e_, "p", -1))
                    if 0 <= p_ < args.n_class:
                        counts[p_] += 1
                    n_out[0] += 1
            while sim.members[-1].get_events():
                pass

        for s0 in range(0, len(spikes), seg):
            seg_end = min(s0 + seg, len(spikes))
            for s in range(s0, seg_end, chunk):
                sim.members[0].write(spikes[s:s + chunk])
                time.sleep(pause)
                drain()
            # hard settle: inter-layer queues empty before next segment
            time.sleep(5.0)
            for _ in range(10):
                drain()
                time.sleep(1.0)
        for m in mons:
            m["graph"].stop()
        sim.network.stop()

        # remainder flushed at stop
        mon = sim.read_monitor(n_iaf - 1)
        n_out[0] += len(mon)
        for p_ in mon["p"]:
            if 0 <= int(p_) < args.n_class:
                counts[int(p_)] += 1
        return counts, n_out[0]

    for name, idxs in sel_map.items():
        if args.limit:
            idxs = idxs[:args.limit]
        n_ok, per_clip = 0, []
        t0 = time.time()
        for j, i in enumerate(idxs):
            if args.binned_data:
                bp = os.path.join(args.binned_data,
                                  os.path.relpath(files[i], args.data))
                ev, label = load_binned_events(bp, args.binned_stride)
            else:
                ev, label = load_events(files[i], args.max_events)
            if args.invert_p:
                ev['p'] = 1 - ev['p']
            counts, n_out = sim_run(ev)         # drained-during-write
            pred = int(counts.argmax())
            ok = pred == label
            n_ok += ok
            per_clip.append(os.path.basename(files[i]))
            el = (time.time() - t0) / 60
            print(f"  [{name} {j+1}/{len(idxs)}] {per_clip[-1]} "
                  f"E={len(ev)} outspikes={n_out} counts={counts.tolist()}"
                  f" pred={pred} tgt={label} {'OK' if ok else 'XX'} "
                  f"({el:.1f} min)", flush=True)
        n = len(idxs)
        print(f"\n== SPECKSIM {name} ({n} clips) ==")
        print(f"  event-level accuracy : {n_ok}/{n} = {100*n_ok/n:.2f}%")


if __name__ == "__main__":
    main()
