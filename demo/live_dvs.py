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
    ap.add_argument("--window", type=float, default=0.3)
    ap.add_argument("--calib", type=float, default=3.0,
                    help="seconds per calibration phase (idle/flat/curved)")
    ap.add_argument("--pstay", type=float, default=0.9,
                    help="HMM self-transition prob (contact persistence)")
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
    root.geometry("360x200")
    pred = tk.Label(root, text="—", font=("Arial", 40, "bold"),
                    fg="#777777")
    pred.pack(pady=(10, 0))
    rates = tk.Label(root, text="", font=("Consolas", 11))
    rates.pack()
    phase = tk.Label(root, text="", font=("Consolas", 10), fg="#4466aa")
    phase.pack(pady=2)
    conf = tk.Label(root, text="", font=("Consolas", 10), fg="#888888")
    conf.pack()

    # --- directional-evidence readout ---
    # the chip's output is biphasic: sustained holds are ~1:1 mush while
    # contact/transient bursts fire one direction 10-45x. Only windows
    # with a strong direction carry information — integrate signed
    # log-ratio weighted by activity, threshold on accumulated score.
    collecting = [False]
    idle_recs = []                # (cf, cc) during idle calibration
    idle_floor = 30.0             # counts/window -> "touching"
    lr_bias = 0.0                 # idle log-ratio offset
    score = 0.0                   # leaky integrator of signed evidence
    shown_state = "IDLE"
    SCORE_T = 2.5                 # claim threshold
    DECAY = 0.90                  # per window (~2 s half-life @0.3 s)
    last_win = time.time()

    def _set_phase_text():
        if lam[0] is None:
            verb = "collecting..." if collecting[0] else \
                   "ready — press START"
            phase.config(text=f"CAL: remove all contact\n{verb}")
        else:
            phase.config(text="live — directional readout")

    lam = [None]

    def _finish_calib():
        n = len(idle_recs)
        if n >= 3:
            arr = np.array(idle_recs)
            tot = arr[:, 0] + arr[:, 1]
            # floor: idle total ~ mu + 4 sd, at least 30 counts
            global_idle = max(30.0, tot.mean() + 4 * tot.std())
            # bias: typical idle direction offset (e.g. flat bias)
            bias = float(np.log((arr[:, 0] + 1) /
                                (arr[:, 1] + 1)).mean())
            lam[0] = (global_idle, bias)
            print(f"idle: floor={global_idle:.0f}/win bias={bias:+.2f}",
                  flush=True)
        else:
            lam[0] = (30.0, 0.0)
            print("idle calib skipped (too few windows)", flush=True)

    def btn_press():
        nonlocal score, shown_state
        if lam[0] is not None:                # restart calibration
            lam[0] = None
            collecting[0] = False
            idle_recs.clear()
            score = 0.0
            btn.config(text="START")
            _set_phase_text()
            return
        if not collecting[0]:
            collecting[0] = True
            btn.config(text="STOP")
            _set_phase_text()
        else:
            collecting[0] = False
            _finish_calib()
            btn.config(text="RECALIBRATE")
            _set_phase_text()

    btn = tk.Button(root, text="START", command=btn_press,
                    font=("Arial", 11, "bold"))
    btn.pack(pady=4)
    root.bind("<Return>", lambda e: btn_press())
    _set_phase_text()

    csv_f = None
    t0_rec = time.time()
    if args.record:
        csv_f = open(args.record, "w")
        csv_f.write("t,cf,cc,flat_rate,curved_rate,dvs,dvs_rate,state,"
                    "score\n")

    def tick():
        nonlocal last_win, score, shown_state
        now = time.time()
        if now - last_win >= args.window:
            dt = now - last_win
            cf, cc = shared["counts"][0], shared["counts"][1]
            nd = shared["dvs"]
            shared["counts"][0] = shared["counts"][1] = 0
            shared["dvs"] = 0
            f, c = cf / dt, cc / dt
            if lam[0] is None:
                if collecting[0]:
                    idle_recs.append((cf, cc))
                pred.config(text="CAL", fg="#777777")
                shown_state = "CAL"
                csv_extra = ""
            else:
                floor, bias = lam[0]
                tot = cf + cc
                lr = np.log((cf + 1) / (cc + 1)) - bias
                score *= DECAY
                # gesture-demo-style gating: ignore windows below the
                # floor (no info) AND above the burst cap (liftoff/
                # approach transients saturate the readout — not real
                # contact evidence)
                burst = max(2000.0, 10 * floor)
                if floor < tot <= burst:
                    score += lr * min(1.0, tot / 500.0)
                if tot <= floor:
                    shown_state = "IDLE"
                elif score > SCORE_T:
                    shown_state = "FLAT"
                elif score < -SCORE_T:
                    shown_state = "CURVED"
                col = {"IDLE": "#777777", "FLAT": "#2e8b57",
                       "CURVED": "#d2691e"}[shown_state]
                pred.config(text=shown_state, fg=col)
                conf.config(text=f"score {score:+.2f}   "
                                 f"floor {floor:.0f}")
                csv_extra = f"{score:.3f}"
                print(f"  flat {f:6.0f}/s  curved {c:6.0f}/s  "
                      f"tot={tot:5d} score={score:+.2f} -> "
                      f"{shown_state}", flush=True)
            rates.config(text=f"flat {f:5.0f}/s   curved {c:5.0f}/s"
                              f"   dvs {nd/dt:6.0f}/s")
            if csv_f:
                csv_f.write(f"{now-t0_rec:.3f},{cf},{cc},{f:.1f},"
                            f"{c:.1f},{nd},{nd/dt:.1f},{shown_state},"
                            f"{csv_extra}\n")
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
            print(f"recorded -> {args.record} + .spikes.npz "
                  f"({len(spike_log)} spikes)", flush=True)
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
