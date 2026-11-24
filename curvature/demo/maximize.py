"""Find the LARGEST channel taper that still maps onto Speck2f's 9 cores.

Strategy: greedy growth from the proven BASE_CHANNELS. For each conv layer,
try progressively larger output-channel counts; a candidate is accepted only
if the REAL mapping solver (ChipFactory.map_layers_to_cores — the same code
that runs inside DynapcnnNetwork.to()) finds a valid layer->core assignment.
Iterate to a fixpoint, then print the max taper + per-core utilization.

This answers: "how big can the real training architecture get before it
breaks the chip?" Train at or below the printed MAX_CHANNELS.

    python maximize.py            # run the sweep
    python maximize.py --quick    # fewer candidates, faster
"""

import argparse
import torch
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from model import BASE_CHANNELS, build_backbone

# per-core memory ceilings from sinabs/backend/dynapcnn/chips/dynapcnn.py
CORE_LIMITS = {i: (16, 64) for i in range(3)}     # kernel KiB, neuron KiB
CORE_LIMITS.update({i: (32, 32) for i in (3, 4)})
CORE_LIMITS.update({i: (64, 16) for i in (5, 6)})
CORE_LIMITS.update({i: (16, 16) for i in (7, 8)})

_builder = ChipFactory("speck2fdevkit:0").get_config_builder()


def try_mapping(channels):
    """Returns (mapping dict, memory_summary) or (None, memory_summary)."""
    backbone = build_backbone(channels, batch_size=1)
    hw = DynapcnnNetwork(snn=backbone, discretize=True,
                         input_shape=(2, 128, 128), dvs_input=True)
    ms = hw.memory_summary()
    try:
        mapping = _builder.map_layers_to_cores(hw.dynapcnn_layers)
    except Exception:
        mapping = None
    del hw
    return mapping, ms


def param_count(channels):
    return sum(p.numel() for p in build_backbone(channels).parameters())


def report(channels, mapping, ms, tag=""):
    print(f"\n=== {tag} channels={tuple(channels)} params={param_count(channels):,} ===")
    usage = {c: [0, 0] for c in CORE_LIMITS}
    for i in sorted(ms['kernel']):
        core = mapping.get(i, -1) if mapping else -1
        k, n = ms['kernel'][i] / 1024, ms['neuron'][i] / 1024
        if core >= 0:
            usage[core][0] += k
            usage[core][1] += n
        print(f"  layer {i}: kernel={k:6.1f}Ki neuron={n:6.1f}Ki -> core {core}")
    if mapping:
        print("  core usage:")
        for c, (kl, nl) in sorted(CORE_LIMITS.items()):
            ku, nu = usage[c]
            print(f"    core {c}: kernel {ku:5.1f}/{kl}Ki ({100*ku/kl:4.0f}%)  "
                  f"neuron {nu:5.1f}/{nl}Ki ({100*nu/nl:4.0f}%)")
    else:
        print("  NO VALID MAPPING")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    candidates = [48, 64, 96, 128] if args.quick else [40, 48, 64, 80, 96, 128, 160]

    channels = list(BASE_CHANNELS)
    mapping, ms = try_mapping(channels)
    report(channels, mapping, ms, "BASELINE")

    # greedy fixpoint: keep raising whichever layer can grow while still mapping
    improved = True
    rounds = 0
    while improved and rounds < 6:
        improved = False
        rounds += 1
        for li in range(8):
            for cand in candidates:
                if cand <= channels[li]:
                    continue
                trial = list(channels)
                trial[li] = cand
                m, _ms = try_mapping(trial)
                if m is not None:
                    channels = trial
                    mapping, ms = m, _ms
                    improved = True
                    print(f"  + layer {li+1}: -> {cand}ch  (params {param_count(channels):,})")
                    break  # take first growth, move to next layer
    report(channels, mapping, ms, "MAX FITTING")

    print(f"\nMAX_CHANNELS = {tuple(channels)}")
    print("-> paste this into demo/model.py and mirror it in the real trainer.")
    print("-> rule for real training: never exceed these per-layer channel counts.")


if __name__ == "__main__":
    main()
