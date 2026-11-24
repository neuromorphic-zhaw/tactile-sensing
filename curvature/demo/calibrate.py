"""Measure idle output spike rates so --threshold can be set above the
noise floor.  Deploys the model, runs N seconds, prints stats.

    python calibrate.py --model ../deploy_checkpoints/wclamp_v93.33_t86.67.pt
"""

import argparse
import time
import numpy as np
import torch
import samna
from sinabs.backend.dynapcnn import DynapcnnNetwork
from model import SpeckDemoTrainer, load_checkpoint


def infer_channels(sd):
    return tuple(sd[f"spiking_backbone.{i * 2}.weight"].shape[0] for i in range(8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="speck2fdevkit:0")
    ap.add_argument("--secs", type=float, default=15.0)
    ap.add_argument("--slow_clk", type=int, default=1000)
    args = ap.parse_args()

    sd = torch.load(args.model, map_location="cpu")
    model = SpeckDemoTrainer(channels=infer_channels(sd), batch_size=1)
    with torch.no_grad():
        _ = model(torch.zeros((1, 40, 2, 128, 128)))
    load_checkpoint(model, args.model)

    hw = DynapcnnNetwork(snn=model.spiking_backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)
    hw.to(device=args.device, chip_layers_ordering="auto",
          monitor_layers=["dvs", -1], slow_clk_frequency=args.slow_clk,
          config_modifier=lambda cfg: (
              setattr(cfg.dvs_layer, "raw_monitor_enable", True), cfg)[1])
    print("deployed — DON'T touch the sensor; measuring idle rates",
          flush=True)

    sink = samna.graph.sink_from(hw.samna_device.get_model().get_source_node())
    t0 = time.time()
    last = t0
    counts = [0, 0]
    dvs = 0
    rates_f, rates_c, rates_d = [], [], []
    try:
        while time.time() - t0 < args.secs:
            for e in sink.get_events():
                t = type(e).__name__
                if t == "Spike" and e.feature in (0, 1):
                    counts[e.feature] += 1
                elif t.endswith("DvsEvent"):
                    dvs += 1
            now = time.time()
            if now - last >= 0.5:
                rates_f.append(counts[0] / (now - last))
                rates_c.append(counts[1] / (now - last))
                rates_d.append(dvs / (now - last))
                counts = [0, 0]
                dvs = 0
                last = now
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

    def stats(v):
        v = np.asarray(v)
        return (f"med {np.median(v):.0f}  p95 {np.percentile(v, 95):.0f}  "
                f"max {v.max():.0f}")

    print(f"\n--- idle output rates over {len(rates_f)} half-second windows ---")
    print(f"flat   spikes/s : {stats(rates_f)}")
    print(f"curved spikes/s : {stats(rates_c)}")
    print(f"dvs    events/s : {stats(rates_d)}")
    floor = max(np.percentile(rates_f, 95), np.percentile(rates_c, 95))
    print(f"\nsuggested --threshold {int(floor * 1.5) + 10}"
          f"  --margin 1.5   (p95 noise ~{floor:.0f}/s)")


if __name__ == "__main__":
    main()
