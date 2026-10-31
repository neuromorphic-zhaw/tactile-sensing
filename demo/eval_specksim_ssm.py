"""Event-level eval for SNN+SSM models — the honest deployment test.

Pipeline = what a real hybrid deployment looks like:
    raw events -> specksim graph (full 9-core chip emulation, discretized
    weights) -> per-window readout-spike counts -> trained SSM head -> class.

Two windowing modes mirror the two dataset families the SSM variants trained
on:
    --win_mode time   : fixed real-time windows of 40 ms (10ms bins x ::4
                        stride = the t10ms trainer's effective step)
    --win_mode count  : N input events per window (count-bin; evT166 used
                        T=166 bins x ::2 -> each window = clip_events/83)

Feature mode is auto-detected from the checkpoint (ssm_head.proj in_features):
    readout (T',2): monitor the final readout IAF — per-class spike counts.
    l8      (T',64): monitor the last-conv IAF — per-channel counts / 16
                     sites (the count-domain analog of the trainer's
                     mean-over-spatial). Binarized variant = channel presence.

Also reports the same head fed a binarized (0/1) sequence — event-mode firing
is hotter than the frame-trained regime, so both calibrations are worth seeing.

    python eval_specksim_ssm.py --model <best_ssm.pt> --win_mode count
"""

import argparse
import glob
import math
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

# --- SSM head (identical to devin_ssm/*/train.py — keep in sync) ---
class S4DLayer(nn.Module):
    def __init__(self, d_model, d_state=64, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.A_re = nn.Parameter(-0.5 * torch.ones(d_model, d_state))
        self.A_im = nn.Parameter(math.pi * torch.arange(d_state).float()
                                 .unsqueeze(0).repeat(d_model, 1))
        self.B = nn.Parameter(torch.randn(d_model, d_state, 2) / math.sqrt(2 * d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state, 2) / math.sqrt(d_state))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, u):
        dt = self.log_dt.exp()
        A = torch.complex(self.A_re, self.A_im)
        dA = A * dt[:, None]
        Abar = torch.exp(dA)
        Bbar = torch.view_as_complex(self.B.float()) * dt[:, None] * (Abar - 1.0) / dA
        C = torch.view_as_complex(self.C.float())
        h = u.new_zeros(u.shape[0], u.shape[2], A.shape[1], dtype=torch.cfloat)
        ys = []
        uh = u.transpose(0, 1)
        for t in range(u.shape[1]):
            h = Abar * h + uh[t][:, :, None] * Bbar
            ys.append(torch.einsum('bdn,dn->bd', h, C).real)
        return self.out(torch.stack(ys, dim=1))


class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.ssm = S4DLayer(d_model, d_state)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Linear(2 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.ssm(self.n1(x)))
        return x + self.drop(self.ff(self.n2(x)))


class SSMHead(nn.Module):
    def __init__(self, feat_dim, d_model, d_state, n_layers, n_classes, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.blocks = nn.ModuleList(SSMBlock(d_model, d_state, dropout)
                                    for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, feat):
        h = self.proj(feat)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h).mean(dim=1))


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
    ev['t'] = (t - t0).astype(np.uint32)
    ev['p'] = p.astype(np.uint32)
    return ev, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="best_ssm.pt")
    ap.add_argument("--data", default=RAW_DATA)
    ap.add_argument("--split", choices=["val", "test", "all"], default="val")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_events", type=int, default=500000,
                    help="truncate long clips (bad_alloc guard); 0 = no cap")
    ap.add_argument("--win_mode", choices=["time", "count"], default="count",
                    help="time: 40ms windows (t10ms SSM) | count: N/83 events "
                         "per window (evT166 SSM)")
    ap.add_argument("--win_us", type=int, default=40000,
                    help="time-mode window length in us")
    ap.add_argument("--win_events", type=int, default=0,
                    help="count-mode override: input events per window "
                         "(default n_clip/83)")
    ap.add_argument("--n_class", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--d_state", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=3)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "**/*.npz"), recursive=True))
    print(f"{len(files)} raw clips")
    _, va_i, te_i = split_indices(len(files))
    sel_map = {"VAL": sorted(va_i), "TEST": sorted(te_i),
               "ALL": list(range(len(files)))}
    if args.split == "val":
        sel_map = {"VAL": sel_map["VAL"]}
    elif args.split == "test":
        sel_map = {"TEST": sel_map["TEST"]}
    if args.clip:
        sel_map = {"CLIP": [i for i, f in enumerate(files)
                            if os.path.basename(f) == args.clip]}

    # --- rebuild model pieces from the joint checkpoint ---
    sd = torch.load(args.model, map_location="cpu")
    channels = tuple(sd[f"spiking_backbone.{i * 2}.weight"].shape[0]
                     for i in range(8))
    model = SpeckDemoTrainer(channels=channels, batch_size=1)
    with torch.no_grad():
        _ = model(torch.zeros((1, 40, 2, 128, 128)))
    n_w = load_checkpoint(model, args.model)          # spiking_backbone.* only
    print(f"backbone: {n_w} weights, channels={channels}")

    # feature mode auto-detected from the ckpt: proj in_features = 2 for the
    # readout-spike models, 64 for the l8 (last-conv feature) models
    feat_dim = int(sd["ssm_head.proj.weight"].shape[1])
    feat_l8 = feat_dim != args.n_class
    head = SSMHead(feat_dim=feat_dim, d_model=args.d_model, d_state=args.d_state,
                   n_layers=args.n_layers, n_classes=args.n_class)
    head_sd = {k[len("ssm_head."):]: v for k, v in sd.items()
               if k.startswith("ssm_head.")}
    head.load_state_dict(head_sd)
    head.eval()
    print(f"ssm_head: {len(head_sd)} tensors loaded "
          f"(d{args.d_model}/s{args.d_state}x{args.n_layers}) — "
          f"feat={'l8 (T,64) last-conv spikes' if feat_l8 else 'readout (T,2)'}")

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
    # readout models monitor the final (readout) IAF; l8 models monitor the
    # last CONV layer's IAF (64 ch, 4x4) — the SNN-as-feature-extractor tap
    mon_layer = n_iaf - 2 if feat_l8 else n_iaf - 1
    sim.add_monitor(mon_layer)
    print(f"SpecksimNetwork built — chip emulates 9 cores, SSM runs host-side "
          f"(monitoring layer {mon_layer})")

    def drain_counts(mons):
        """Read whatever the monitored IAF has emitted so far -> (feat_dim,) counts."""
        c = np.zeros(feat_dim, dtype=np.int64)
        for m in mons:
            for e_ in m["sink"].get_events():
                p_ = getattr(e_, "feature", getattr(e_, "p", -1))
                if 0 <= p_ < feat_dim:
                    c[p_] += 1
        while sim.members[-1].get_events():
            pass
        return c

    def clip_features(spikes, bounds):
        """Stream the clip; after each window boundary, settle briefly and
        count monitor spikes -> (n_windows, n_class) feature sequence."""
        sim.network.start()
        mons = list(sim.monitors.values())
        for m in mons:
            _ = m["sink"].get_events()
            m["graph"].start()
        feats = np.zeros((len(bounds) - 1, feat_dim), dtype=np.float32)
        chunk = 20000
        since_settle = 0
        for w in range(len(bounds) - 1):
            s0, s1 = bounds[w], bounds[w + 1]
            for s in range(s0, s1, chunk):
                sim.members[0].write(spikes[s:s + chunk])
                time.sleep(0.03)
                feats[w] += drain_counts(mons)
            time.sleep(0.4)                        # short settle per window
            for _ in range(3):
                feats[w] += drain_counts(mons)
                time.sleep(0.2)
            # hard settle every ~150K input events — matches the proven
            # drain pattern from eval_specksim_events.py (bad_alloc fix):
            # inter-layer queues must fully empty, not just per-window nap
            since_settle += s1 - s0
            if since_settle >= 150000:
                time.sleep(5.0)
                for _ in range(10):
                    feats[w] += drain_counts(mons)
                    time.sleep(1.0)
                since_settle = 0
        # final long settle, catch stragglers into the last window
        time.sleep(3.0)
        for _ in range(5):
            feats[-1] += drain_counts(mons)
            time.sleep(0.5)
        for m in mons:
            m["graph"].stop()
        sim.network.stop()
        mon = sim.read_monitor(mon_layer)          # remainder flushed at stop
        for p_ in mon["p"]:
            if 0 <= int(p_) < feat_dim:
                feats[-1][int(p_)] += 1
        if feat_l8:
            # training l8 = mean over the 4x4 spatial map -> count-domain
            # analog = spikes per channel / 16 sites (same [0,1]-ish scale)
            feats = feats / 16.0
        return feats

    for name, idxs in sel_map.items():
        if args.limit:
            idxs = idxs[:args.limit]
        n_ok = n_ok_bin = 0
        t0 = time.time()
        for j, i in enumerate(idxs):
            ev, label = load_events(files[i], args.max_events)
            spikes = sim.xytp_to_specksim_spikes(ev)
            if args.win_mode == "count":
                w = args.win_events or max(1, len(spikes) // 83)
                bounds = np.arange(0, len(spikes) + w, w)
            else:
                bounds = np.searchsorted(ev['t'], np.arange(1, 2000) * args.win_us)
                bounds = np.unique(np.concatenate([[0], bounds, [len(spikes)]]))
            feats = clip_features(spikes, bounds)
            x = torch.from_numpy(feats)[None].float()
            with torch.no_grad():
                pred = int(head(x).argmax(1))
                # l8: presence/absence per channel is the cleaner binary analog
                x_bin = (x > 0).float() if feat_l8 else x.clamp(0, 1)
                pred_bin = int(head(x_bin).argmax(1))
            n_ok += (pred == label)
            n_ok_bin += (pred_bin == label)
            el = (time.time() - t0) / 60
            print(f"  [{name} {j+1}/{len(idxs)}] {os.path.basename(files[i])} "
                  f"E={len(ev)} T'={len(feats)} outspk={int(feats.sum())} "
                  f"pred={pred}{'/b'+str(pred_bin) if pred_bin!=pred else ''} "
                  f"tgt={label} {'OK' if pred==label else 'XX'} ({el:.1f} min)",
                  flush=True)
        n = len(idxs)
        print(f"\n== SPECKSIM+SSM {name} ({n} clips, win={args.win_mode}) ==")
        print(f"  raw counts : {n_ok}/{n} = {100*n_ok/n:.2f}%")
        print(f"  binarized  : {n_ok_bin}/{n} = {100*n_ok_bin/n:.2f}%")


if __name__ == "__main__":
    main()
