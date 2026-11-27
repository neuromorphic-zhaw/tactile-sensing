"""N-class specksim (full chip-emulation) eval for phase-2 texture models.

Port of phase1_curvature/demo/eval_specksim_events.py. Differences:
  - 5-class readout (counts over 5 class features)
  - data = RAW .h5 clips (dataset_tex_raw), sliced to the exact touch
    window via phase_idx (matches training distribution)
  - clip selection = the run's OWN split.json test indices (stratified),
    mapped through the texture/evT166 npz ordering -> raw h5 paths
  - default readout = 200K-event windowed majority vote (deployment mode)

    python eval_specksim_tex.py --model runs/dot_evT166_s2/best.pt
"""

import argparse
import glob
import json
import os
import time
import h5py
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
from train_tex import SpeckFittingTrainer

VD = os.path.dirname(os.path.abspath(__file__))
TEX_NPZ = "/cfs/earth/scratch/matheron/tactile-sensing/data/phase2_texture_dot_ab/texture/evT166"
TEX_RAW = "/cfs/earth/scratch/matheron/tactile-sensing/data/phase2_texture_dot_ab/dataset_tex_raw"


def npz_file_list(cond):
    """Exact reproduction of TactileDataset ordering — split.json indices
    are positions in this list."""
    return sorted(glob.glob(os.path.join(TEX_NPZ, cond, "**/*.npz"), recursive=True))


def raw_h5_for(npz_path, cond):
    cls = os.path.basename(os.path.dirname(npz_path))
    base = os.path.basename(npz_path).replace(".npz", ".h5")
    return os.path.join(TEX_RAW, cond, cls, base)


def load_touch_events(h5_path, max_events=0):
    """Raw events of the exact touch window [touch, touch_10s), rebased to 0."""
    with h5py.File(h5_path) as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f["phase_name"][:]]
        idx = f["phase_idx"][:]
        i0, i1 = int(idx[names.index("touch")]), int(idx[names.index("touch_10s")])
        x = f["x"][i0:i1].astype(np.uint32)
        y = f["y"][i0:i1].astype(np.uint32)
        t = f["t"][i0:i1]
        p = f["p"][i0:i1].astype(np.uint32)
        raw_label = f.attrs.get("label", "")
        if isinstance(raw_label, (bytes, np.bytes_)):
            raw_label = raw_label.decode()
        label = {"flat": 0, "curved": 1, "edge": 2, "soft": 3, "grid": 4}[str(raw_label)]
    if max_events and len(x) > max_events:
        x, y, t, p = x[:max_events], y[:max_events], t[:max_events], p[:max_events]
    t0 = t[0]
    ev = np.zeros(len(x), dtype=[('x', np.uint32), ('y', np.uint32),
                                 ('t', np.uint32), ('p', np.uint32)])
    ev['x'], ev['y'] = x, y
    ev['t'] = (t - t0).astype(np.uint32)
    ev['p'] = p
    return ev, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="run-dir best.pt")
    ap.add_argument("--cond", choices=["dot", "nodot"], default=None,
                    help="default: infer from model path")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window_events", type=int, default=200000)
    ap.add_argument("--max_events", type=int, default=1000000)
    args = ap.parse_args()

    run_dir = os.path.dirname(os.path.abspath(args.model))
    cond = args.cond or os.path.basename(run_dir).split("_")[0]
    assert cond in ("dot", "nodot"), f"cannot infer condition from {run_dir}"

    # clip selection: the run's own stratified split, npz ordering -> raw h5
    sp = json.load(open(os.path.join(run_dir, "split.json")))
    files = npz_file_list(cond)
    key = "test_indices" if args.split == "test" else "val_indices"
    if key not in sp:
        raise KeyError(f"{key} not in {run_dir}/split.json — refusing to "
                       f"evaluate on the wrong split")
    sel = [(i, files[i]) for i in sp[key]]
    if args.limit:
        sel = sel[:args.limit]
    print(f"{len(sel)} {args.split} clips, cond={cond}, run={os.path.basename(run_dir)}")

    # --- build the chip emulation ---
    sd = torch.load(args.model, map_location="cpu")
    n_classes = int(sd["spiking_backbone.18.weight"].shape[0])
    n_bins = 166  # nt only shapes the torch-side dummy forward; conversion ignores it
    model = SpeckFittingTrainer(nt=n_bins // 2, n_classes=n_classes, time_stride=2)
    model.load_state_dict({k: v for k, v in sd.items() if not k.endswith(".v_mem")},
                          strict=False)
    with torch.no_grad():
        _ = model(torch.zeros((1, n_bins, 2, 128, 128)))
    print(f"loaded ckpt, n_classes={n_classes}")

    hw = DynapcnnNetwork(snn=model.spiking_backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)

    def build_specksim(hw, input_shape):
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
                f, cur = convert_pooling_layer(nn.AvgPool2d(2, 2), cur)
                filters.append(f)
                wscale = float(2 * 2)
        filters.append(samna.BasicSinkNode_specksim_events_spike())
        return SpecksimNetwork(graph, graph.sequential(filters),
                               initial_sleep_duration=5.0,
                               subsequent_sleep_duration=1.0)

    sim = build_specksim(hw, (2, 128, 128))
    n_iaf = len(hw.dynapcnn_layers)
    sim.add_monitor(n_iaf - 1)
    print(f"SpecksimNetwork built — {n_iaf} dynapcnn layers, full chip emulation")

    def sim_run(ev, chunk=20000, pause=0.05, seg=200000):
        spikes = sim.xytp_to_specksim_spikes(ev)
        sim.network.start()
        mons = list(sim.monitors.values())
        for m in mons:
            _ = m["sink"].get_events()
            m["graph"].start()
        counts = np.zeros(n_classes, dtype=np.int64)
        n_out = [0]

        def drain():
            for m in mons:
                for e_ in m["sink"].get_events():
                    p_ = getattr(e_, "feature", getattr(e_, "p", -1))
                    if 0 <= p_ < n_classes:
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
            time.sleep(5.0)
            for _ in range(10):
                drain()
                time.sleep(1.0)
        for m in mons:
            m["graph"].stop()
        sim.network.stop()

        mon = sim.read_monitor(n_iaf - 1)
        n_out[0] += len(mon)
        for p_ in mon["p"]:
            if 0 <= int(p_) < n_classes:
                counts[int(p_)] += 1
        return counts, n_out[0]

    n_ok, results = 0, []
    t0 = time.time()
    for j, (i, npz_path) in enumerate(sel):
        h5_path = raw_h5_for(npz_path, cond)
        ev, label = load_touch_events(h5_path, args.max_events)
        votes = np.zeros(n_classes, dtype=np.int64)
        n_out, wins = 0, []
        for w0 in range(0, len(ev), args.window_events):
            wc, wn = sim_run(ev[w0:w0 + args.window_events])
            n_out += wn
            wpred = int(wc.argmax())
            votes[wpred] += 1
            wins.append(wc.tolist())
        pred = int(votes.argmax())
        ok = bool(pred == label)
        n_ok += ok
        el = (time.time() - t0) / 60
        print(f"  [{j+1}/{len(sel)}] {os.path.basename(h5_path)} "
              f"E={len(ev)} windows={[w + ['->'] + [int(np.argmax(w))] for w in wins]} "
              f"votes={votes.tolist()} pred={pred} tgt={label} "
              f"{'OK' if ok else 'XX'} ({el:.1f} min)", flush=True)
        results.append({"clip": os.path.basename(h5_path), "label": label,
                        "pred": pred, "ok": ok, "votes": votes.tolist(),
                        "windows": wins, "n_events": int(len(ev))})
    n = len(sel)
    acc = 100.0 * n_ok / max(n, 1)
    print(f"\n== SPECKSIM {args.split.upper()} ({n} clips, {os.path.basename(run_dir)}) ==")
    print(f"  event-level accuracy : {n_ok}/{n} = {acc:.2f}%")
    out_json = os.path.join(run_dir, f"specksim_{args.split}.json")
    json.dump({"model": args.model, "cond": cond, "split": args.split,
               "window_events": args.window_events, "n_correct": n_ok, "n": n,
               "acc": acc, "clips": results}, open(out_json, "w"), indent=2)
    print(f"  written: {out_json}")


if __name__ == "__main__":
    main()
