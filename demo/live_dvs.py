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
from collections import deque
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
    ap.add_argument("--window", type=float, default=0.3)
    ap.add_argument("--vote", type=int, default=4,
                    help="rate-weighted vote over last N windows")
    ap.add_argument("--idle_floor", type=float, default=100.0,
                    help="total spikes/s below this -> IDLE (near-silence)")
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
    # raw_monitor is the only DVS tap that emits under a model config,
    # but it's PRE-filter -> we denoise in python (spatiotemporal
    # correlation: keep only events with a recently-active 8-neighbour)
    # and forward the clean events to the visualizer
    gui = subprocess.Popen([SAMNAGUI, ENDPOINT, "-W", "0.85", "-H", "0.85"])
    graph = samna.graph.EventFilterGraph()
    clean_src = samna.BasicSourceNode_speck2f_event_output_event()
    _, _, streamer = graph.sequential(
        [clean_src, "Speck2fDvsToVizConverter", "VizEventStreamer"])
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
    last_ev = np.full((128, 128), -1.0)   # last event time per pixel
    pix_act = np.zeros((128, 128))        # leaky per-pixel activity
    last_decay = [time.time()]
    CORR_T = 0.02                          # 20 ms correlation window
    ACT_TAU = 0.25                         # activity decay (s)
    ACT_MAX = 20.0                         # ~sustained >80/s -> noise pixel

    def drain():
        while drain_alive[0]:
            keep = []
            now = time.time()
            pix_act *= np.exp(-(now - last_decay[0]) / ACT_TAU)
            last_decay[0] = now
            for e in sink.get_events():
                tn = type(e).__name__
                if tn == "Spike":
                    if e.feature in (0, 1):
                        shared["counts"][e.feature] += 1
                    continue
                if not hasattr(e, "x"):     # forward real DVS events only
                    continue
                x, y = e.x, e.y
                pix_act[y, x] += 1.0
                if pix_act[y, x] > ACT_MAX:
                    continue                # sustained-firing -> line/hot pixel
                x0, x1 = max(0, x - 1), min(128, x + 2)
                y0, y1 = max(0, y - 1), min(128, y + 2)
                nb = last_ev[y0:y1, x0:x1].copy()
                nb[y - y0, x - x0] = -1.0     # exclude self -> hot pixels die
                if (now - nb < CORR_T).any():
                    keep.append(e)
                last_ev[y, x] = now
            if keep:
                clean_src.write(keep)
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

    last_win = time.time()
    hist_f, hist_c = deque(maxlen=args.vote), deque(maxlen=args.vote)
    shown = None

    def tick():
        nonlocal last_win, shown
        now = time.time()
        if now - last_win >= args.window:
            dt = now - last_win
            f, c = (shared["counts"][0] / dt, shared["counts"][1] / dt)
            shared["counts"][0] = shared["counts"][1] = 0
            hist_f.append(f); hist_c.append(c)
            # rate-weighted: strong contact windows dominate weak
            # transient ones; hysteresis: challenger must beat the
            # incumbent's accumulated evidence to flip the label
            sf, sc = sum(hist_f), sum(hist_c)
            if sf + sc < args.idle_floor * len(hist_f):
                shown = None                      # near-silence -> IDLE
                pred.config(text="IDLE", fg="#777777")
            elif shown != 0 and sf >= sc:
                shown = 0
                pred.config(text="FLAT", fg="#2e8b57")
            elif shown != 1 and sc > sf:
                shown = 1
                pred.config(text="CURVED", fg="#d2691e")
            rates.config(text=f"flat {f:5.0f}/s   curved {c:5.0f}/s")
            print(f"  flat {f:6.0f}/s  curved {c:6.0f}/s  "
                  f"sum(f={sf:.0f},c={sc:.0f}) -> "
                  f"{CLASSES[shown] if shown is not None else 'IDLE'}",
                  flush=True)
            last_win = now
        root.after(50, tick)

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
