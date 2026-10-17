"""Deploy SpeckFittingTrainer checkpoint to Speck2f devkit + live 2-class readout.

Fixes vs the old local_flash_pipeline.py:
  - model class defined here (no import of the training script, which would
    execute the whole trainer and crash on the HPC DATA_PATH)
  - v_mem buffers stripped from the state_dict (they're transient sim state,
    saved with batch=10 shapes that can't strict-load into batch=1)
  - device passed as string "speck2fdevkit:0" (DeviceInfo objects are rejected)
  - slow clock enabled via .to(slow_clk_frequency=...) — required for any
    time-windowed readout
  - flash binary includes factory_config.io_sel = 24 (readout -> output pins)

Live readout = software majority-vote (SynSense gesture-demo pattern):
count last-core spikes per feature over a window; winner only if it beats
--threshold, otherwise "idle". feature 0 = flat, feature 1 = curved.

Usage:
    python deploy_speck.py --model best_speck_fitting.pt
    python deploy_speck.py --model best_speck_fitting.pt --flash
"""

import argparse
import time
import torch
import torch.nn as nn
import sinabs
import sinabs.layers as sl
import samna


def _iaf():
    return sl.IAFSqueeze(batch_size=1, spike_threshold=1.0, min_v_mem=-1.0,
                         surrogate_grad_fn=sinabs.activation.PeriodicExponential())


class SpeckFittingTrainer(nn.Module):
    """Must match speck_compatible/train_speck_fitting.py exactly."""
    def __init__(self):
        super().__init__()
        self.spiking_backbone = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=4, stride=2, padding=1, bias=False), _iaf(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), _iaf(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False), _iaf(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), _iaf(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False), _iaf(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), _iaf(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), _iaf(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), _iaf(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256, 2, bias=False), _iaf(),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x[:, ::4].reshape(b * (t // 4), c, h, w)
        return self.spiking_backbone(x).reshape(b, t // 4, -1).mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best_speck_fitting.pt")
    ap.add_argument("--device", default="speck2fdevkit:0")
    ap.add_argument("--threshold", type=int, default=30,
                    help="min winning-feature spikes per window to declare a class")
    ap.add_argument("--window", type=float, default=1.0,
                    help="readout window in seconds")
    ap.add_argument("--slow_clk", type=int, default=1000, help="external slow clock Hz")
    ap.add_argument("--flash", action="store_true", help="also write speck_standalone_tactile.bin")
    args = ap.parse_args()

    # --- 1. rebuild + warm-up + load (drop v_mem buffers) ---
    model = SpeckFittingTrainer()
    with torch.no_grad():
        _ = model(torch.zeros((1, 40, 2, 128, 128)))
    sd = torch.load(args.model, map_location="cpu")
    weights = {k: v for k, v in sd.items() if not k.endswith(".v_mem")}
    model.load_state_dict(weights, strict=False)  # v_mem dropped: warm-up already sized it
    print(f"loaded {args.model} ({len(weights)} tensors)")

    # --- 2. convert + deploy ---
    from sinabs.backend.dynapcnn import DynapcnnNetwork
    hw = DynapcnnNetwork(snn=model.spiking_backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)
    hw.to(device=args.device,
          chip_layers_ordering="auto",
          monitor_layers=["dvs", -1],
          slow_clk_frequency=args.slow_clk)
    print(f"deployed — layer->core map: {hw.chip_layers_ordering}")

    # --- 3. optional standalone flash binary ---
    if args.flash:
        cfg = hw.get_configuration()
        cfg.factory_config.io_sel = 24          # route readout pins (errata/standalone)
        binary = samna.speck2f.configuration_to_flash_binary(cfg)
        with open("speck_standalone_tactile.bin", "wb") as f:
            f.write(bytes(binary))
        print("flash binary -> speck_standalone_tactile.bin")

    # --- 4. live majority-vote readout (gesture-demo pattern, plain Python) ---
    names = {0: "flat", 1: "curved"}
    sink = samna.graph.sink_from(hw.samna_device.get_model().get_source_node())
    print(f"live readout: {args.window}s window, threshold {args.threshold} spikes — Ctrl-C to quit")
    try:
        while True:
            time.sleep(args.window)
            counts = {0: 0, 1: 0}
            dvs_count = 0
            for e in sink.get_events():
                t = type(e).__name__
                if t == "Spike" and e.feature in counts:
                    counts[e.feature] += 1
                elif t == "DvsEvent":
                    dvs_count += 1
            winner, top = max(counts.items(), key=lambda kv: kv[1])
            runner = min(counts.values())
            if top >= args.threshold and top > runner:
                print(f"{names[winner].upper():>6}  (f0={counts[0]:>5} f1={counts[1]:>5} | dvs={dvs_count})")
            else:
                print(f"  idle  (f0={counts[0]:>5} f1={counts[1]:>5} | dvs={dvs_count})")
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
