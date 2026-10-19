"""Simple native GUI — no samnagui, no JIT, no matplotlib event loop.

tkinter window rendered at ~30fps:
  left   : DVS event view 128x128 (zoomed x3) — red=ON, green=OFF,
           hard on/off per frame (no fade)
  right  : FLAT / CURVED / IDLE prediction (state machine: flips must
           pass through IDLE) + live spike rates + dvs rate

Background-noise handling: at startup (and on "Recalibrate") the app
measures ~5s of idle output rates and subtracts each feature's own
baseline — decisions run on the EXCESS above the noise floor, so the
model's resting bias (~500/s flat at idle) can't trigger a class.

    python simple_gui.py --model ../deploy_checkpoints/wclamp_v93.33_t86.67.pt
"""

import argparse
import threading
import time
import numpy as np
import torch
import samna
import tkinter as tk
from sinabs.backend.dynapcnn import DynapcnnNetwork
from model import SpeckDemoTrainer, load_checkpoint

CLASSES = {0: "FLAT", 1: "CURVED"}
ZOOM = 3
FPS_MS = 33                       # ~30 fps


def infer_channels(sd):
    return tuple(sd[f"spiking_backbone.{i * 2}.weight"].shape[0] for i in range(8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="speck2fdevkit:0")
    ap.add_argument("--threshold", type=float, default=500.0,
                    help="EXCESS spikes/s (above baseline) to declare a class")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--confirm", type=int, default=2,
                    help="consecutive confident windows needed to claim "
                         "a class — rejects liftoff bursts")
    ap.add_argument("--calib", type=float, default=5.0,
                    help="idle-baseline measurement time at start (s)")
    ap.add_argument("--slow_clk", type=int, default=1000)
    args = ap.parse_args()

    # --- deploy model ---
    sd = torch.load(args.model, map_location="cpu")
    model = SpeckDemoTrainer(channels=infer_channels(sd), batch_size=1)
    with torch.no_grad():
        _ = model(torch.zeros((1, 40, 2, 128, 128)))
    load_checkpoint(model, args.model)

    hw = DynapcnnNetwork(snn=model.spiking_backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)
    hw.to(device=args.device, chip_layers_ordering="auto",
          monitor_layers=["dvs", -1], slow_clk_frequency=args.slow_clk,
          config_modifier=lambda c: (
              setattr(c.dvs_layer, "monitor_enable", True),
              setattr(c.dvs_layer, "raw_monitor_enable", True), c)[2])
    print(f"deployed — map: {hw.chip_layers_ordering}", flush=True)
    sink = samna.graph.sink_from(
        hw.samna_device.get_model().get_source_node())

    # --- background drain thread: doing 15-50k python-level event objects
    # per second on the GUI thread is what froze the window ---
    shared = {"buf": np.zeros((128, 128, 3), dtype=np.uint8),
              "counts": [0, 0], "dvs": 0}
    drain_alive = [True]

    def drain():
        buf = shared["buf"]
        while drain_alive[0]:
            for e in sink.get_events():
                t = type(e).__name__
                if t == "Spike" and e.feature in (0, 1):
                    shared["counts"][e.feature] += 1
                elif t.endswith("DvsEvent"):
                    shared["dvs"] += 1
                    if 0 <= e.x < 128 and 0 <= e.y < 128:
                        if getattr(e, "p", 1):
                            buf[e.y, e.x] = (230, 40, 40)
                        else:
                            buf[e.y, e.x] = (40, 230, 40)
            time.sleep(0.005)

    threading.Thread(target=drain, daemon=True).start()

    # --- GUI ---
    root = tk.Tk()
    root.title("Speck2f — flat vs curved")
    alive = True

    def on_close():
        nonlocal alive
        alive = False
        drain_alive[0] = False
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)

    photo = tk.PhotoImage(width=128, height=128)
    zimg = [photo.zoom(ZOOM)]          # keep a ref or Tk GCs it -> blank
    img_lbl = tk.Label(root, image=zimg[0])
    img_lbl.grid(row=0, column=0, rowspan=5, padx=8, pady=8)

    pred = tk.Label(root, text="IDLE", font=("Arial", 42, "bold"),
                    fg="#777777", width=9)
    pred.grid(row=0, column=1, padx=12, pady=(8, 0))
    rates = tk.Label(root, text="", font=("Consolas", 12))
    rates.grid(row=1, column=1)
    dvslbl = tk.Label(root, text="", font=("Consolas", 12))
    dvslbl.grid(row=2, column=1)
    baselbl = tk.Label(root, text="", font=("Consolas", 9), fg="#888888")
    baselbl.grid(row=3, column=1)

    state = -1
    pend_win, pend_n = -1, 0        # candidate class + consecutive count
    last_win = time.time()

    # calibration state
    calibrating = True
    calib_f, calib_c = [], []
    calib_start = time.time()
    base = [0.0, 0.0]

    def start_calib():
        nonlocal calibrating, calib_start
        calibrating = True
        calib_f.clear()
        calib_c.clear()
        calib_start = time.time()
        pred.config(text="CAL…", fg="#777777")

    tk.Button(root, text="Recalibrate\n(lift off first)",
              command=start_calib).grid(row=4, column=1, pady=6)

    def tick():
        nonlocal state, last_win, calibrating, pend_win, pend_n
        # snapshot shared state — drain thread owns the buffer between frames
        f_cnt, c_cnt, dvs_cnt = shared["counts"][0], shared["counts"][1], \
            shared["dvs"]
        frame = shared["buf"].copy()

        now = time.time()
        if now - last_win >= args.window:
            dt = now - last_win
            f, c = f_cnt / dt, c_cnt / dt
            dvs_rate = dvs_cnt / dt
            shared["counts"][0] = shared["counts"][1] = 0
            shared["dvs"] = 0

            if calibrating:
                calib_f.append(f)
                calib_c.append(c)
                pred.config(text="CAL…")
                if now - calib_start >= args.calib:
                    base[0] = float(np.median(calib_f))
                    base[1] = float(np.median(calib_c))
                    calibrating = False
                    baselbl.config(
                        text=f"baseline: flat {base[0]:.0f}/s  "
                             f"curved {base[1]:.0f}/s")
                    print(f"baseline flat={base[0]:.0f}/s "
                          f"curved={base[1]:.0f}/s", flush=True)
            else:
                fe = max(0.0, f - base[0])     # excess above idle floor
                ce = max(0.0, c - base[1])
                top, runner = max(fe, ce), min(fe, ce)
                confident = top >= args.threshold and \
                    top >= max(runner * args.margin, 1e-9)
                winner = 1 if ce > fe else 0

                if confident:
                    if winner == pend_win:
                        pend_n += 1
                    else:
                        pend_win, pend_n = winner, 1
                    # claim only after `confirm` consecutive windows —
                    # a 1-window liftoff burst dies as IDLE
                    if state == winner:
                        pass                              # hold
                    elif state == -1 and pend_n >= args.confirm:
                        state = winner                    # claim from idle
                    elif state != winner:
                        state = -1                        # change -> IDLE
                else:
                    state = -1
                    pend_win, pend_n = -1, 0
                if state >= 0:
                    pred.config(text=CLASSES[state],
                                fg="#2e8b57" if state == 0 else "#d2691e")
                else:
                    pred.config(text="IDLE", fg="#777777")
                rates.config(text=f"flat {f:5.0f}/s (+{fe:4.0f})\n"
                                  f"curved {c:5.0f}/s (+{ce:4.0f})")
                dvslbl.config(text=f"dvs {dvs_rate:6.0f}/s")
            last_win = now

        photo.configure(data=b"P6 128 128 255\n" + frame.tobytes())
        zimg[0] = photo.zoom(ZOOM)
        img_lbl.config(image=zimg[0])
        shared["buf"][:] = 0          # hard reset — on/off per frame

        if alive:
            root.after(FPS_MS, tick)

    tick()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    print("done", flush=True)


if __name__ == "__main__":
    main()
