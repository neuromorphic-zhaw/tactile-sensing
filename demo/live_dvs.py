"""Deployed model + smooth native GUI — no JIT filters anywhere.

Two native windows:
  samnagui  : DVS ActivityPlot (top) + Power plot (bottom) — the exact
              event path as speck_connect/dvs_power_out.py, but with the
              model deployed (raw_monitor_enable needed for DVS events
              under a model config)
  tkinter   : FLAT / CURVED / IDLE prediction label + rates — text-only
              window, can't lag; Speck-intended stateless majority
              readout (SynSense gesture demo gates)

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
    ap.add_argument("--window", type=float, default=0.5,
                    help="readout window s (gesture demo: 0.5)")
    ap.add_argument("--thr_low", type=int, default=20,
                    help="min total output spikes per window else IDLE")
    ap.add_argument("--thr_high", type=int, default=15000,
                    help="max total spikes per window else IDLE (burst cap)")
    ap.add_argument("--det_thr", type=int, default=10,
                    help="min winner spike count per window else IDLE")
    ap.add_argument("--record", default="",
                    help="CSV path — logs per-window rates + state; also "
                         "dumps all output spikes to <path>.spikes.npz")
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
    shared = {"counts": [0, 0], "dvs": 0}
    drain_alive = [True]
    spike_log = []                # (t_mono, feature) every output spike
    dvs_log = []                  # (t_mono, chip_ts, x, y, p) raw sensor
    last_ev = np.full((128, 128), -1.0)   # last event time per pixel
    pix_act = np.zeros((128, 128))        # leaky per-pixel activity
    last_decay = [time.time()]
    CORR_T = 0.02                          # 20 ms correlation window
    ACT_TAU = 0.25                         # activity decay (s)
    ACT_MAX = 20.0                         # ~sustained >80/s -> noise pixel

    def drain():
        nonlocal pix_act
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
                        if args.record:
                            spike_log.append((now, e.feature))
                    continue
                if not hasattr(e, "x"):     # forward real DVS events only
                    continue
                shared["dvs"] += 1
                x, y = e.x, e.y
                if args.record:
                    dvs_log.append((now, getattr(e, "timestamp", 0),
                                    x, y, getattr(e, "p", 0)))
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
    root.geometry("360x160")
    pred = tk.Label(root, text="—", font=("Arial", 40, "bold"),
                    fg="#777777")
    pred.pack(pady=(10, 0))
    rates = tk.Label(root, text="", font=("Consolas", 11))
    rates.pack()
    conf = tk.Label(root, text="", font=("Consolas", 10), fg="#888888")
    conf.pack()

    # --- Speck-intended readout (SynSense gesture demo mechanism) ---
    # per window: count spikes per output feature; total must be within
    # [thr_low, thr_high], winner count > det_thr, unique max required —
    # otherwise the default (IDLE) is shown. Stateless: no memory,
    # no calibration.
    last_win = time.time()

    csv_f = None
    t0_rec = time.time()
    if args.record:
        csv_f = open(args.record, "w")
        csv_f.write("t,cf,cc,flat_rate,curved_rate,dvs,dvs_rate,state\n")

    def tick():
        nonlocal last_win
        now = time.time()
        if now - last_win >= args.window:
            dt = now - last_win
            cf, cc = shared["counts"][0], shared["counts"][1]
            nd = shared["dvs"]
            shared["counts"][0] = shared["counts"][1] = 0
            shared["dvs"] = 0
            f, c = cf / dt, cc / dt
            # raw output as-is: argmax, no gates, no memory
            if cf > cc:
                shown_state = "FLAT"
            elif cc > cf:
                shown_state = "CURVED"
            elif cf == 0:
                shown_state = "IDLE"
            else:
                shown_state = "TIE"
            col = {"IDLE": "#777777", "FLAT": "#2e8b57",
                   "CURVED": "#d2691e", "TIE": "#888888"}[shown_state]
            pred.config(text=shown_state, fg=col)
            conf.config(text=f"counts {cf} vs {cc}")
            print(f"  flat {f:6.0f}/s  curved {c:6.0f}/s  "
                  f"({cf:4d} vs {cc:4d}) -> {shown_state}", flush=True)
            rates.config(text=f"flat {f:5.0f}/s   curved {c:5.0f}/s"
                              f"   dvs {nd/dt:6.0f}/s")
            if csv_f:
                csv_f.write(f"{now-t0_rec:.3f},{cf},{cc},{f:.1f},"
                        f"{c:.1f},{nd},{nd/dt:.1f},{shown_state}\n")
                csv_f.flush()
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
        if csv_f:
            csv_f.close()
            np.savez(args.record + ".spikes.npz",
                     spikes=np.array(spike_log))
            np.savez(args.record + ".dvs.npz",
                     dvs=np.array(dvs_log))
            print(f"recorded -> {args.record} + .spikes.npz "
                  f"({len(spike_log)} spikes) + .dvs.npz "
                  f"({len(dvs_log)} events)", flush=True)
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
