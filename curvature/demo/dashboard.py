"""Live deployment dashboard — runs until you close the window.

Reads the REAL deployed model's events directly (the same proven path the
terminal readout uses) and shows four live panels:

  +----------------------+----------------------+
  | DVS sensor activity  |  BIG class decision  |
  | (128x128 heatmap)    |  FLAT/CURVED/IDLE    |
  +----------------------+----------------------+
  | output spike rates   |  chip power (mW)     |
  | flat vs curved       |  5 channels          |
  +----------------------+----------------------+

    python dashboard.py                          # mock_speck.pt
    python dashboard.py --model ../best_speck_fitting.pt
    python dashboard.py --model mock_speck.pt --threshold 30 --window 1.0
"""

import argparse
import collections
import time
import numpy as np
import torch
import samna
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sinabs.backend.dynapcnn import DynapcnnNetwork
from model import SpeckDemoTrainer, load_checkpoint

CLASSES = {0: ("FLAT", "#2e8b57"), 1: ("CURVED", "#d2691e")}


def infer_channels(sd):
    return tuple(sd[f"spiking_backbone.{i * 2}.weight"].shape[0] for i in range(8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock_speck.pt")
    ap.add_argument("--device", default="speck2fdevkit:0")
    ap.add_argument("--threshold", type=int, default=30,
                    help="spikes per second a feature needs to declare a class")
    ap.add_argument("--margin", type=float, default=1.5,
                    help="winner must exceed runner-up by this factor, else IDLE")
    ap.add_argument("--window", type=float, default=1.0, help="decision window, seconds")
    ap.add_argument("--slow_clk", type=int, default=1000)
    args = ap.parse_args()

    sd = torch.load(args.model, map_location="cpu")
    channels = infer_channels(sd)
    print(f"checkpoint {args.model}: channels={channels}")

    model = SpeckDemoTrainer(channels=channels, batch_size=1)
    with torch.no_grad():
        _ = model(torch.zeros((1, 40, 2, 128, 128)))
    load_checkpoint(model, args.model)

    hw = DynapcnnNetwork(snn=model.spiking_backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)

    def _enable_raw_dvs(cfg):
        cfg.dvs_layer.raw_monitor_enable = True   # monitor raw sensor events too
        return cfg

    hw.to(device=args.device, chip_layers_ordering="auto",
          monitor_layers=["dvs", -1], slow_clk_frequency=args.slow_clk,
          config_modifier=_enable_raw_dvs)
    print(f"deployed — layer->core map: {hw.chip_layers_ordering}")

    dk = hw.samna_device
    event_sink = samna.graph.sink_from(dk.get_model().get_source_node())
    power_sink = None
    try:
        dk.get_power_monitor().start_auto_power_measurement(100.0)
        power_sink = samna.graph.sink_from(dk.get_power_monitor().get_source_node())
    except Exception as e:
        print(f"power monitor unavailable ({e}) — power panel will stay empty")

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(11, 7))
    fig.suptitle("Speck2f live deployment — flat vs curved", fontsize=13)
    ax_dvs = fig.add_subplot(2, 2, 1)
    ax_cls = fig.add_subplot(2, 2, 2)
    ax_spk = fig.add_subplot(2, 2, 3)
    ax_pwr = fig.add_subplot(2, 2, 4)

    dvs_img = ax_dvs.imshow(np.zeros((128, 128, 3)), origin="lower")
    ax_dvs.set_title("DVS activity")
    ax_dvs.set_xticks([]); ax_dvs.set_yticks([])

    ax_cls.axis("off")
    cls_text = ax_cls.text(0.5, 0.55, "IDLE", ha="center", va="center",
                           fontsize=44, fontweight="bold", color="#777777")
    cls_sub = ax_cls.text(0.5, 0.3, "", ha="center", va="center", fontsize=11)

    hist = collections.deque(maxlen=60)          # (t, flat_rate, curved_rate)
    line_f, = ax_spk.plot([], [], color="#2e8b57", label="flat")
    line_c, = ax_spk.plot([], [], color="#d2691e", label="curved")
    ax_spk.set_title("output spikes / s")
    ax_spk.legend(loc="upper right", fontsize=8)
    ax_spk.set_xlabel("s"); ax_spk.grid(alpha=0.3)

    power_hist = collections.deque(maxlen=300)   # (t, [5 vals])
    pwr_names = ["io", "ram", "logic", "vddd", "vdda"]
    pwr_lines = [ax_pwr.plot([], [], label=n, lw=1)[0] for n in pwr_names]
    ax_pwr.set_title("power (mW)")
    ax_pwr.legend(loc="upper right", fontsize=7, ncol=3)
    ax_pwr.grid(alpha=0.3)

    fig.tight_layout()
    plt.show(block=False)
    print("DASHBOARD LIVE — close the window to stop.")

    dvs_buf = np.zeros((128, 128, 3))   # RGB: red=ON(p=1), green=OFF(p=0)
    seen_types = {}
    last_debug = 0.0
    win_counts = [0, 0]
    dvs_count = 0
    t0 = time.time()
    last_decision = t0
    last_spike_t = t0
    power_map = {n: 0.0 for n in pwr_names}
    state = -1                      # -1 = IDLE, 0 = FLAT, 1 = CURVED
    decision, sub = "IDLE", ""

    try:
        while plt.fignum_exists(fig.number):
            # --- drain events ---
            for e in event_sink.get_events():
                t = type(e).__name__
                seen_types[t] = seen_types.get(t, 0) + 1
                if t == "Spike" and e.feature in (0, 1):
                    win_counts[e.feature] += 1
                elif t.endswith("DvsEvent"):
                    dvs_count += 1
                    if 0 <= e.x < 128 and 0 <= e.y < 128:
                        if getattr(e, "p", 1):                       # ON -> red
                            dvs_buf[e.y, e.x] = (1.0, 0.15, 0.15)
                        else:                                        # OFF -> green
                            dvs_buf[e.y, e.x] = (0.15, 1.0, 0.15)
            if power_sink is not None:
                for e in power_sink.get_events():
                    try:
                        power_map[pwr_names[e.channel]] = e.power
                    except Exception:
                        try:
                            power_map[pwr_names[e.channel]] = e.value
                        except Exception:
                            pass

            now = time.time()
            if now - last_debug >= 5.0:
                print("event types seen:", seen_types, flush=True)
                seen_types = {}
                last_debug = now
            dt = now - last_spike_t
            if dt >= 0.1:
                hist.append((now - t0, win_counts[0] / dt, win_counts[1] / dt))
                win_counts = [0, 0]
                last_spike_t = now

            # --- decision per window — state machine: a class flip must
            # pass through IDLE (you can't touch flat and curved at once) ---
            if now - last_decision >= args.window:
                rates = [h for h in hist if h[0] > now - t0 - args.window]
                f = sum(r[1] for r in rates) / max(len(rates), 1)
                c = sum(r[2] for r in rates) / max(len(rates), 1)
                top, runner = max(f, c), min(f, c)
                confident = top >= args.threshold and top >= runner * args.margin
                winner = 1 if c > f else 0

                if confident:
                    if state == -1 or winner == state:
                        state = winner          # claim from idle, or hold
                    else:
                        state = -1              # different class won -> IDLE first
                else:
                    state = -1                  # no confident winner -> IDLE

                if state >= 0:
                    decision, color = CLASSES[state]
                else:
                    decision, color = "IDLE", "#777777"
                sub = f"flat {f:.0f}/s   curved {c:.0f}/s   dvs {dvs_count}/s"
                cls_text.set_text(decision); cls_text.set_color(color)
                cls_sub.set_text(sub)
                dvs_count = 0
                last_decision = now

            # --- redraw artists ---
            dvs_img.set_data(np.clip(dvs_buf, 0, 1))
            dvs_buf[:] = 0    # hard reset each frame — no fade, just on/off
            if hist:
                ts = [h[0] for h in hist]
                line_f.set_data(ts, [h[1] for h in hist])
                line_c.set_data(ts, [h[2] for h in hist])
                ax_spk.set_xlim(max(0, ts[-1] - 60), ts[-1] + 1)
                ax_spk.set_ylim(0, max(50, max(h[1] for h in hist),
                                     max(h[2] for h in hist)) * 1.15)
            if power_sink is not None:
                power_hist.append((now - t0, [power_map[n] for n in pwr_names]))
                ts = [h[0] for h in power_hist]
                for i, ln in enumerate(pwr_lines):
                    ln.set_data(ts, [h[1][i] for h in power_hist])
                if ts:
                    ax_pwr.set_xlim(max(0, ts[-1] - 30), ts[-1] + 1)
                    ax_pwr.relim(); ax_pwr.autoscale_view(scalex=False)

            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nclosing — stopping power monitor")
        try:
            dk.get_power_monitor().stop_auto_power_measurement()
        except Exception:
            pass


if __name__ == "__main__":
    main()
