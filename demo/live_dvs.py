"""Deployed model + smooth native GUI — no JIT filters anywhere.

Two native windows:
  samnagui  : DVS ActivityPlot (top) + Power plot (bottom) — the exact
              event path as speck_connect/dvs_power_out.py, but with the
              model deployed (raw_monitor_enable needed for DVS events
              under a model config)
  tkinter   : FLAT / CURVED / IDLE prediction label + rates — text-only
              window, can't lag; state machine + baseline subtraction +
              2-window debounce (liftoff burst rejection)

    python live_dvs.py --model ../deploy_checkpoints/wclamp_v93.33_t86.67.pt
"""

import argparse
import subprocess
import threading
import time
import numpy as np
import torch
import samna
import tkinter as tk
from sinabs.backend.dynapcnn import DynapcnnNetwork
from model import SpeckDemoTrainer, load_checkpoint

CLASSES = {0: "FLAT", 1: "CURVED"}
ENDPOINT = "tcp://0.0.0.0:40001"
SAMNAGUI = "/home/ronald/speck/bin/samnagui"


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
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--calib", type=float, default=5.0)
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

    def _enable_monitors(cfg):
        cfg.dvs_layer.monitor_enable = True
        cfg.dvs_layer.raw_monitor_enable = True   # emits under model config
        # on-chip filter: kills hot pixels / background noise lines
        # (disabled by default -> raw stream is noisy)
        cfg.dvs_filter.enable = True
        cfg.dvs_filter.hot_pixel_filter_enable = True
        cfg.dvs_filter.threshold = 5
        return cfg

    hw.to(device=args.device, chip_layers_ordering="auto",
          monitor_layers=["dvs", -1], slow_clk_frequency=args.slow_clk,
          config_modifier=_enable_monitors)
    print(f"deployed {args.model} — map: {hw.chip_layers_ordering}",
          flush=True)
    dk = hw.samna_device

    # --- samnagui (pure converters — no JIT anywhere) ---
    gui = subprocess.Popen([SAMNAGUI, ENDPOINT, "-W", "0.85", "-H", "0.85"])
    graph = samna.graph.EventFilterGraph()
    _, _, streamer = graph.sequential(
        [dk.get_model_source_node(), "Speck2fDvsToVizConverter",
         "VizEventStreamer"])
    cfg_src, _ = graph.sequential(
        [samna.BasicSourceNode_ui_event(), streamer])
    streamer.set_streamer_endpoint(ENDPOINT)

    pm = dk.get_power_monitor()
    graph.sequential([pm.get_source_node(),
                      "MeasurementToVizConverter", streamer])

    for _ in range(300):
        if streamer.wait_for_receiver_count(100) > 0:
            break
        if gui.poll() is not None:
            raise RuntimeError("samnagui exited before connecting")
    else:
        raise RuntimeError("no visualizer receiver")
    graph.start()
    pm.start_auto_power_measurement(100.0)
    cfg_src.write([samna.ui.VisualizerConfiguration(plots=[
        samna.ui.ActivityPlotConfiguration(
            128, 128, "DVS Layer", layout=[0, 0, 1, 0.7]),
        samna.ui.PowerMeasurementPlotConfiguration(
            "Power Consumption", 5, ["io", "ram", "logic", "vddd", "vdda"],
            [0, 0.7, 1, 1], 10, 2),
    ])])
    print("samnagui live: DVS + power", flush=True)

    # --- readout sink -> background drain thread ---
    sink = samna.graph.sink_from(dk.get_model().get_source_node())
    shared = {"counts": [0, 0]}
    drain_alive = [True]

    def drain():
        while drain_alive[0]:
            for e in sink.get_events():
                if type(e).__name__ == "Spike" and e.feature in (0, 1):
                    shared["counts"][e.feature] += 1
            time.sleep(0.005)

    threading.Thread(target=drain, daemon=True).start()

    # --- prediction window (text-only tk) ---
    root = tk.Tk()
    root.title("Prediction")
    root.geometry("340x160")
    pred = tk.Label(root, text="IDLE", font=("Arial", 40, "bold"),
                    fg="#777777")
    pred.pack(pady=(10, 0))
    rates = tk.Label(root, text="", font=("Consolas", 11))
    rates.pack()
    baselbl = tk.Label(root, text="", font=("Consolas", 9), fg="#888888")
    baselbl.pack()

    state = -1
    pend_win, pend_n = -1, 0
    calibrating = True
    calib_f, calib_c = [], []
    calib_start = time.time()
    base = [0.0, 0.0]
    last_win = time.time()

    def start_calib():
        nonlocal calibrating, calib_start
        calibrating = True
        calib_f.clear(); calib_c.clear()
        calib_start = time.time()
        pred.config(text="CAL…", fg="#777777")

    tk.Button(root, text="Recalibrate\n(lift off first)",
              command=start_calib).pack(pady=4)

    def tick():
        nonlocal state, pend_win, pend_n, calibrating, last_win
        now = time.time()
        if now - last_win >= args.window:
            dt = now - last_win
            f, c = (shared["counts"][0] / dt, shared["counts"][1] / dt)
            shared["counts"][0] = shared["counts"][1] = 0
            if calibrating:
                calib_f.append(f); calib_c.append(c)
                pred.config(text="CAL…")
                if now - calib_start >= args.calib:
                    base[0] = float(np.median(calib_f))
                    base[1] = float(np.median(calib_c))
                    calibrating = False
                    baselbl.config(
                        text=f"baseline flat {base[0]:.0f}/s  "
                             f"curved {base[1]:.0f}/s")
                    print(f"baseline flat={base[0]:.0f} "
                          f"curved={base[1]:.0f}", flush=True)
            else:
                fe, ce = max(0.0, f - base[0]), max(0.0, c - base[1])
                top = max(fe, ce)
                confident = top >= args.threshold
                winner = 1 if ce > fe else 0
                if confident:
                    pend_n = pend_n + 1 if winner == pend_win else 1
                    pend_win = winner
                    if state == winner:
                        pass
                    elif state == -1 and pend_n >= args.confirm:
                        state = winner
                    elif state != winner:
                        state = -1
                else:
                    state, pend_win, pend_n = -1, -1, 0
                if state >= 0:
                    pred.config(text=CLASSES[state],
                                fg="#2e8b57" if state == 0 else "#d2691e")
                else:
                    pred.config(text="IDLE", fg="#777777")
                rates.config(text=f"flat {f:5.0f}/s (+{fe:4.0f})   "
                                  f"curved {c:5.0f}/s (+{ce:4.0f})")
                print(f"  flat {f:6.0f}/s  curved {c:6.0f}/s  -> "
                      f"{CLASSES[state] if state >= 0 else 'IDLE'}",
                      flush=True)
            last_win = now
        root.after(100, tick)

    def on_close():
        drain_alive[0] = False
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)

    print("LIVE — close either window or Ctrl-C to stop.", flush=True)
    tick()
    try:
        while True:
            root.update()
            if gui.poll() is not None:
                break
            time.sleep(0.05)
    except (KeyboardInterrupt, tk.TclError):
        pass
    finally:
        drain_alive[0] = False
        try:
            graph.stop()
            pm.stop_auto_power_measurement()
        except Exception:
            pass
        if gui.poll() is None:
            gui.terminate()
        print("done", flush=True)


if __name__ == "__main__":
    main()
