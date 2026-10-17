"""Full deployment visualizer — runs until you close the GUI window.

Launches the samna GUI (same tool as speck_connect/dvs_power_out.py) with:
  - DVS activity plot   : live 128x128 event view of the sensor
  - Spike Count plot    : per-class output spikes over time (flat vs curved)
  - Readout plot        : big class icon — FLAT / CURVED / IDLE prediction
                          (samna's JitMajorityReadout = gesture-demo mechanism)
  - Power plot          : io / ram / logic / vdd / vda in mW

Terminal keeps printing the majority decision each second, mirroring the
GUI readout — the same "output" a real deployment produces.

    python live_visualizer.py                          # mock_speck.pt
    python live_visualizer.py --model mock_speck.pt
    python live_visualizer.py --model ../best_speck_fitting.pt --threshold 50
"""

import argparse
import multiprocessing
import os
import time
import torch
import samna
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer
from model import SpeckDemoTrainer, load_checkpoint

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
NAMES = {0: "FLAT", 1: "CURVED"}


def make_icons():
    """Tiny PNG class icons for the readout plot (generated once)."""
    os.makedirs(ASSET_DIR, exist_ok=True)
    specs = [(0, "flat", "#2e8b57"), (1, "curved", "#d2691e"), (2, "idle", "#777777")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for idx, name, color in specs:
        p = os.path.join(ASSET_DIR, f"{idx}_{name}.png")
        paths.append(p)
        if os.path.exists(p):
            continue
        fig = plt.figure(figsize=(2, 2))
        fig.text(0.5, 0.5, name.upper(), ha="center", va="center",
                 fontsize=34, fontweight="bold", color=color)
        fig.savefig(p, dpi=60, transparent=False)
        plt.close(fig)
    return paths


def infer_channels(state_dict):
    """Read conv weight shapes -> channel taper, so any trainer checkpoint
    (base or max) loads without mismatched args."""
    chans = []
    for i in range(8):
        w = state_dict[f"spiking_backbone.{i * 2}.weight"]
        chans.append(w.shape[0])
    return tuple(chans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock_speck.pt")
    ap.add_argument("--device", default="speck2fdevkit:0")
    ap.add_argument("--threshold", type=int, default=10,
                    help="spikes a feature needs per collection window to win")
    ap.add_argument("--interval", type=int, default=500,
                    help="spike-collection window in ms (GUI + readout)")
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
    hw.to(device=args.device, chip_layers_ordering="auto",
          monitor_layers=["dvs", -1], slow_clk_frequency=args.slow_clk)
    print(f"deployed — layer->core map: {hw.chip_layers_ordering}")

    icons = make_icons()
    viz = DynapcnnVisualizer(
        window_scale=(7, 12),
        dvs_shape=(128, 128),
        add_readout_plot=True,
        add_spike_count_plot=True,
        add_power_monitor_plot=True,
        spike_collection_interval=args.interval,
        readout_prediction_threshold=args.threshold,
        readout_default_return_value=2,          # index 2 -> idle icon
        power_monitor_number_of_items=5,
        feature_names=["flat", "curved"],
        feature_count=2,
        readout_images=icons,
    )
    viz.connect(hw, disjoint_process=False)      # GUI = child process

    # my own decision tap — same events, independent of the GUI branch
    sink = samna.graph.sink_from(hw.samna_device.get_model().get_source_node())
    print("\nVISUALIZER IS LIVE — close the window to stop.\n")

    try:
        while multiprocessing.active_children():
            time.sleep(1.0)
            counts = {0: 0, 1: 0}
            dvs_count = 0
            for e in sink.get_events():
                t = type(e).__name__
                if t == "Spike" and e.feature in counts:
                    counts[e.feature] += 1
                elif t == "DvsEvent":
                    dvs_count += 1
            top, val = max(counts.items(), key=lambda kv: kv[1])
            text = NAMES[top] if val >= args.threshold * (1000 / args.interval) \
                and val > min(counts.values()) else "idle"
            print(f"\r  out: {text:>6}  (flat={counts[0]:>5} curved={counts[1]:>5} "
                  f"| dvs={dvs_count})   ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nclosing — stopping graph + power monitor")
        try:
            hw.samna_device.get_power_monitor().stop_auto_power_measurement()
        except Exception:
            pass
        viz.stop()
        for p in multiprocessing.active_children():
            p.terminate()


if __name__ == "__main__":
    main()
