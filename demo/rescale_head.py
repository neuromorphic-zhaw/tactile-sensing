"""Post-training per-class readout weight rescale (SynSense gesture demo).

The gesture demo multiplies the readout-layer weights per output class
*before* discretization (`rescale_output_weights = {cls: gain}`) to
de-bias classes that over-fire — the class's IAF membrane then charges
slower -> fewer output spikes. Cheap calibration: no retraining, one
float multiply, quantized deployment still lossless (weights stay well
under theta).

Usage:
    python rescale_head.py --in ../deploy_checkpoints/wclamp_v93.33_t86.67.pt \
        --flat 1.0 --curved 0.7 \
        --out ../deploy_checkpoints/wclamp_rescale_c0.7.pt
"""

import argparse
import torch

READOUT_KEY = "spiking_backbone.18.weight"  # Linear(256 -> n_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flat", type=float, default=1.0,
                    help="gain on feature-0 (FLAT) readout weights")
    ap.add_argument("--curved", type=float, default=1.0,
                    help="gain on feature-1 (CURVED) readout weights")
    args = ap.parse_args()

    sd = torch.load(args.src, map_location="cpu")
    w = sd[READOUT_KEY]
    print(f"readout {tuple(w.shape)}  |w|max "
          f"{w.abs().max():.4f}  norms flat {w[0].norm():.2f} "
          f"curved {w[1].norm():.2f}")
    w[0] *= args.flat
    w[1] *= args.curved
    print(f"rescaled x{args.flat}/{args.curved} -> |w|max "
          f"{w.abs().max():.4f}")
    torch.save(sd, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
