"""Live real-time SNN+SSM demo on the Speck2f devkit.

Deploy = the model's 9-core spiking_backbone on chip (volatile), and the SSM
head runs host-side on windowed readout spikes — same host-readout pattern
as live_dvs.py, with the SSM doing temporal integration instead of the
leaky integrator.

    python live_ssm.py --model ../deploy_checkpoints/ssm_t10ms_frz_v80.00_t86.67.pt --win_ms 40
    python live_ssm.py --model ../deploy_checkpoints/ssm_evT166_frz_v83.33_t73.33.pt --win_events 2000

--win_ms:     fixed real-time windows (t10ms models; 40ms = trainer step)
--win_events: every N DVS input events (count-bin models; 'decide every N ev')
"""

import argparse
import glob
import math
import os
import sys
import threading
import time
import collections
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SpeckDemoTrainer, load_checkpoint

import samna
from sinabs.backend.dynapcnn import DynapcnnNetwork

LABELS = ["FLAT", "CURVED"]


class S4DLayer(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.log_dt = nn.Parameter(torch.zeros(d_model))
        self.A_re = nn.Parameter(-0.5 * torch.ones(d_model, d_state))
        self.A_im = nn.Parameter(torch.zeros(d_model, d_state))
        self.B = nn.Parameter(torch.zeros(d_model, d_state, 2))
        self.C = nn.Parameter(torch.zeros(d_model, d_state, 2))
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
    def __init__(self, d_model, d_state):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.ssm = S4DLayer(d_model, d_state)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Linear(2 * d_model, d_model))

    def forward(self, x):
        x = x + self.ssm(self.n1(x))
        return x + self.ff(self.n2(x))


class SSMHead(nn.Module):
    def __init__(self, feat_dim=2, d_model=64, d_state=64, n_layers=3, n_classes=2):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.blocks = nn.ModuleList(SSMBlock(d_model, d_state)
                                    for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, feat):
        h = self.proj(feat)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h).mean(dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="speck2fdevkit:0")
    ap.add_argument("--win_ms", type=float, default=0,
                    help="time-mode window (ms); t10ms models use 40")
    ap.add_argument("--win_events", type=int, default=0,
                    help="count-mode: DVS events per SSM step (evT166 models)")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="SSM sequence length kept (0 = auto: 333 time / 83 count)")
    ap.add_argument("--idle_floor", type=int, default=30,
                    help="output spikes/window below this -> IDLE")
    ap.add_argument("--slow_clk", type=int, default=1000)
    ap.add_argument("--binarize", action="store_true",
                    help="clamp per-window counts to 0/1 (matches eval check)")
    args = ap.parse_args()
    if not args.win_ms and not args.win_events:
        ap.error("need --win_ms or --win_events")
    mode = "time" if args.win_ms else "count"
    max_steps = args.max_steps or (333 if mode == "time" else 83)

    # --- model pieces ---
    sd = torch.load(args.model, map_location="cpu")
    channels = tuple(sd[f"spiking_backbone.{i * 2}.weight"].shape[0]
                     for i in range(8))
    model = SpeckDemoTrainer(channels=channels, batch_size=1)
    with torch.no_grad():
        _ = model(torch.zeros((1, 40, 2, 128, 128)))
    load_checkpoint(model, args.model)
    head = SSMHead()
    head.load_state_dict({k[len("ssm_head."):]: v for k, v in sd.items()
                          if k.startswith("ssm_head.")})
    head.eval()
    print(f"loaded {args.model} (channels={channels})")

    # --- deploy backbone to chip ---
    def cfg_mod(cfg):
        cfg.dvs_layer.raw_monitor_enable = True
        return cfg

    hw = DynapcnnNetwork(snn=model.spiking_backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)
    hw.to(device=args.device, chip_layers_ordering="auto",
          monitor_layers=["dvs", -1], slow_clk_frequency=args.slow_clk,
          config_modifier=cfg_mod)
    print("deployed to", args.device, "mapping:", hw.chip_layers_ordering)

    sink = samna.graph.sink_from(hw.samna_device.get_model().get_source_node())

    # --- windowed spike collector (background thread) ---
    seq = collections.deque(maxlen=max_steps)
    cur = np.zeros(2, dtype=np.int64)
    cur_in = [0]
    win_t0 = [time.time()]
    lock = threading.Lock()

    def collector():
        nonlocal cur
        while True:
            for e in sink.get_events():
                if getattr(e, "feature", -1) in (0, 1) and getattr(e, "layer", -1) != -1:
                    # Spike events from the readout core carry .feature
                    if hasattr(e, "feature"):
                        cur[e.feature] += 1
                elif type(e).__name__.endswith("DvsEvent"):
                    cur_in[0] += 1
            now = time.time()
            close = (now - win_t0[0] >= args.win_ms / 1000) if mode == "time" \
                else (cur_in[0] >= args.win_events)
            if close:
                with lock:
                    seq.append(cur.copy())
                    cur[:] = 0
                    cur_in[0] = 0
                    win_t0[0] = now
            else:
                time.sleep(0.005)

    threading.Thread(target=collector, daemon=True).start()

    # --- tkinter label ---
    import tkinter as tk
    root = tk.Tk()
    root.title("Speck SSM readout")
    lab = tk.Label(root, text="IDLE", font=("Helvetica", 64, "bold"), width=10)
    lab.pack(padx=40, pady=40)
    stat = tk.Label(root, text="", font=("Helvetica", 12))
    stat.pack()

    def tick():
        with lock:
            s = list(seq)
        if not s:
            label, detail = "IDLE", "warming up"
        else:
            feats = np.array(s, dtype=np.float32)
            last_tot = feats[-1].sum()
            if last_tot < args.idle_floor and len(s) >= 3:
                label, detail = "IDLE", f"win spikes {int(last_tot)}"
            else:
                x = torch.from_numpy(feats)[None]
                if args.binarize:
                    x = x.clamp(0, 1)
                with torch.no_grad():
                    logits = head(x)[0]
                pred = int(logits.argmax())
                conf = torch.softmax(logits, 0)[pred].item()
                label = LABELS[pred]
                detail = f"conf {conf:.2f} | steps {len(s)} | win {int(last_tot)}"
        lab.config(text=label,
                   fg={"FLAT": "#0066cc", "CURVED": "#cc3300", "IDLE": "#666666"}[label])
        stat.config(text=detail)
        root.after(100, tick)

    root.after(100, tick)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
