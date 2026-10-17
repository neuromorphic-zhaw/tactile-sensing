"""Deploy the mock checkpoint to the attached Speck2f devkit + live readout.

Identical deploy path to speck_compatible/deploy_speck.py, but uses the demo
model definition so the MAX_CHANNELS taper can be tested on real silicon.

    python run_live.py --model mock_speck.pt
    python run_live.py --model mock_speck.pt --flash --window 1.5 --threshold 50
"""

import argparse
import time
import torch
import samna
from sinabs.backend.dynapcnn import DynapcnnNetwork
from model import SpeckDemoTrainer, MAX_CHANNELS, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock_speck.pt")
    ap.add_argument("--channels", type=int, nargs=8, default=list(MAX_CHANNELS))
    ap.add_argument("--device", default="speck2fdevkit:0")
    ap.add_argument("--threshold", type=int, default=30)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--slow_clk", type=int, default=1000)
    ap.add_argument("--flash", action="store_true")
    args = ap.parse_args()

    model = SpeckDemoTrainer(channels=args.channels, batch_size=1)
    with torch.no_grad():
        _ = model(torch.zeros((1, 40, 2, 128, 128)))
    print(f"loaded {args.model} ({load_checkpoint(model, args.model)} tensors)")

    hw = DynapcnnNetwork(snn=model.spiking_backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)

    def _enable_raw_dvs(cfg):
        cfg.dvs_layer.raw_monitor_enable = True   # filtered dvs monitor emits nothing
        return cfg

    hw.to(device=args.device, chip_layers_ordering="auto",
          monitor_layers=["dvs", -1], slow_clk_frequency=args.slow_clk,
          config_modifier=_enable_raw_dvs)
    print(f"deployed — layer->core map: {hw.chip_layers_ordering}")

    if args.flash:
        cfg = hw.make_config(device=args.device, chip_layers_ordering="auto",
                             monitor_layers=["dvs", -1])
        cfg.factory_config.io_sel = 24
        with open("demo_speck_standalone.bin", "wb") as f:
            f.write(bytes(samna.speck2f.configuration_to_flash_binary(cfg)))
        print("flash binary -> demo_speck_standalone.bin")

    names = {0: "flat", 1: "curved"}
    sink = samna.graph.sink_from(hw.samna_device.get_model().get_source_node())
    print(f"live readout: {args.window}s window, threshold {args.threshold} — Ctrl-C quits")
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
            if top >= args.threshold and top > min(counts.values()):
                print(f"{names[winner].upper():>6}  (f0={counts[0]:>5} f1={counts[1]:>5} | dvs={dvs_count})")
            else:
                print(f"  idle  (f0={counts[0]:>5} f1={counts[1]:>5} | dvs={dvs_count})")
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
