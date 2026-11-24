"""Verify a checkpoint converts and maps — full proof WITHOUT touching hardware.

Runs every step of deployment except the actual .to(device) write:
  1. rebuild model + load checkpoint (v_mem stripped, like deploy_speck.py)
  2. DynapcnnNetwork(discretize=True)        -> int8/int16 quantization
  3. memory_summary() vs per-core limits     -> which core must host each layer
  4. ChipFactory.map_layers_to_cores         -> real solver, real assignment
  5. make_config(device=...)                 -> full Samna config generation

If this prints PASS, the .to() call in run_live.py will succeed.

    python verify_convert.py                      # uses MAX_CHANNELS, random ckpt ok
    python verify_convert.py --model mock_speck.pt --channels 16 32 32 32 32 64 64 64
"""

import argparse
import torch
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from model import SpeckDemoTrainer, MAX_CHANNELS, load_checkpoint

CORE_LIMITS = {i: (16, 64) for i in range(3)}
CORE_LIMITS.update({i: (32, 32) for i in (3, 4)})
CORE_LIMITS.update({i: (64, 16) for i in (5, 6)})
CORE_LIMITS.update({i: (16, 16) for i in (7, 8)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="checkpoint to load (optional)")
    ap.add_argument("--channels", type=int, nargs=8, default=list(MAX_CHANNELS))
    ap.add_argument("--device", default="speck2fdevkit:0")
    args = ap.parse_args()

    print(f"channels={tuple(args.channels)}")
    model = SpeckDemoTrainer(channels=args.channels, batch_size=1)
    with torch.no_grad():
        _ = model(torch.zeros((1, 40, 2, 128, 128)))
    if args.model:
        print(f"loaded {args.model} ({load_checkpoint(model, args.model)} tensors)")
    else:
        print("no checkpoint — using random init (mapping test is weight-agnostic)")

    print("\n[1] DynapcnnNetwork + discretization ...")
    hw = DynapcnnNetwork(snn=model.spiking_backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)
    print("    OK — all layers are Speck-legal types, weights quantized")

    print("\n[2] memory_summary vs per-core limits ...")
    ms = hw.memory_summary()
    feasible = True
    for i in sorted(ms['kernel']):
        k, n = ms['kernel'][i] / 1024, ms['neuron'][i] / 1024
        ok = any(k <= kl and n <= nl for kl, nl in CORE_LIMITS.values())
        feasible &= ok
        print(f"    layer {i}: kernel={k:6.1f}Ki neuron={n:6.1f}Ki  "
              f"{'fits some core' if ok else '!!! NO CORE FITS THIS !!!'}")

    print("\n[3] real mapping solver (map_layers_to_cores) ...")
    builder = ChipFactory(args.device).get_config_builder()
    try:
        mapping = builder.map_layers_to_cores(hw.dynapcnn_layers)
        print(f"    OK — layer->core: {mapping}")
    except Exception as e:
        print(f"    FAIL — {e}")
        feasible = False

    print("\n[4] make_config — full chip configuration ...")
    try:
        cfg = hw.make_config(device=args.device, chip_layers_ordering="auto",
                             monitor_layers=["dvs", -1])
        print(f"    OK — Samna config generated ({len(str(cfg))//1024} KB object)")
    except Exception as e:
        print(f"    FAIL — {e}")
        feasible = False

    print(f"\n{'PASS — this architecture+checkpoint will deploy to Speck2f' if feasible else 'FAIL — shrink channels (see layer table above)'}")
    return 0 if feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
