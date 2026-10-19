"""Quantized (chip-equivalent) validation of a checkpoint.

Runs the SAME SNN through DynapcnnNetwork(discretize=True) on CPU —
sinabs simulates the DYNAP-CNN layers with the discretized weights that
would be written to the chip — and compares accuracy against the float32
model on the same splits.

    python eval_quantized.py --model ../deploy_checkpoints/wclamp_v93.33_t86.67.pt

NOTE on split fidelity: training used glob order on the HPC filesystem,
which is filesystem-dependent. Locally we use sorted filenames — the
val/test membership may differ slightly from the training run's, so we
also report the FULL 200-clip accuracy which is order-independent.
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

DATA_PATH = "/home/ronald/tactile-sensing/data/dataset_curv_padded_10ms"


def load_clip(path):
    with np.load(path) as z:
        d = torch.from_numpy(z['frames']).float()
        return torch.clamp(d, 0.0, 1.0), int(z['label'])


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
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="clips per split, 0=all")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_PATH, "**/*.npz"),
                            recursive=True))
    assert files, f"no npz under {DATA_PATH}"
    print(f"{len(files)} clips")

    tr_i, va_i, te_i = split_indices(len(files))
    print(f"split sizes: train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    # --- float model ---
    sd = torch.load(args.model, map_location="cpu")
    channels = tuple(sd[f"spiking_backbone.{i * 2}.weight"].shape[0]
                     for i in range(8))
    model = SpeckDemoTrainer(channels=channels, batch_size=1)
    with torch.no_grad():
        _ = model(torch.zeros((1, 40, 2, 128, 128)))
    n_w = load_checkpoint(model, args.model)
    model.eval()
    print(f"loaded {n_w} weights, channels={channels}")

    # --- quantized chip-simulation ---
    hw = DynapcnnNetwork(snn=model.spiking_backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)
    print("DynapcnnNetwork(discretize=True) built — simulating chip",
          flush=True)

    def _reset_quant():
        # DynapcnnNetwork.reset_states() is broken in this sinabs
        # version (view-detach bug) — zero the IAF v_mem buffers directly
        for layer in hw.dynapcnn_layers.values():
            for m in layer.modules():
                if isinstance(m, sl.IAFSqueeze):
                    m.v_mem.data = torch.zeros_like(m.v_mem)

    def predict_float(frames):
        # replicate SpeckFittingTrainer.forward exactly: stride ::4 then
        # reshape by ACTUAL strided length (T=1331 -> 333, not t//4)
        sinabs.utils.reset_states(model.spiking_backbone)
        x = frames[::args.stride]
        out = model.spiking_backbone(x)        # (T', 2)
        rates = out.mean(dim=0)
        return int(rates.argmax().item()), rates.tolist()

    def predict_quant(frames, stride):
        _reset_quant()
        x = frames[::stride]                  # stride=1 -> chip-faithful
        out = hw(x)                           # (T',2) output spikes
        rates = out.mean(dim=0)
        return int(rates.argmax().item()), rates.tolist()

    groups = [("VAL", va_i), ("TEST", te_i), ("ALL", set(range(len(files))))]
    for name, idxs in groups:
        sel = sorted(idxs)
        if args.limit:
            sel = sel[:args.limit]
        stats = {"f": [0, 0], "q": [0, 0], "qf": [0, 0]}
        per_clip = []
        t0 = time.time()
        for j, i in enumerate(sel):
            frames, label = load_clip(files[i])
            pf, lf = predict_float(frames)
            pq, lq = predict_quant(frames, args.stride)       # same stride
            pqf, _ = predict_quant(frames, 1)                 # full rate
            stats["f"][0] += pf == label
            stats["q"][0] += pq == label
            stats["qf"][0] += pqf == label
            per_clip.append((os.path.basename(files[i]), label, pf, pq, pqf))
            if (j + 1) % 10 == 0:
                print(f"  [{name}] {j+1}/{len(sel)} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
        n = len(sel)
        print(f"\n== {name} ({n} clips) ==")
        print(f"  float32 stride{args.stride} : {stats['f'][0]}/{n} = "
              f"{100*stats['f'][0]/n:.2f}%")
        print(f"  quant   stride{args.stride} : {stats['q'][0]}/{n} = "
              f"{100*stats['q'][0]/n:.2f}%")
        print(f"  quant   full-rate : {stats['qf'][0]}/{n} = "
              f"{100*stats['qf'][0]/n:.2f}%")
        mism = [c for c in per_clip if c[2] != c[3]]
        if mism and name != "ALL":
            print("  float!=quant on:", ", ".join(m[0] for m in mism))


if __name__ == "__main__":
    main()
