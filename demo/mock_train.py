"""Mock training: synthetic event frames -> real .pt checkpoint.

Generates a tiny dataset in-memory with the same tensor contract as the real
pipeline ((T, 2, 128, 128) float frames, clamped to [0,1], labels {0,1}),
then runs a few epochs of the real forward/optimizer path. Point is NOT
accuracy — it's producing a genuine checkpoint that exercises the exact
deploy path (v_mem buffers included, like the real trainer writes).

Synthetic signal: class 0 = events clustered top half, class 1 = bottom half
(+ noise). The net can actually learn this, so val accuracy is meaningful
enough to sanity-check the training loop.

    python mock_train.py                 # default: MAX_CHANNELS (the big one)
    python mock_train.py --base          # proven taper instead
    python mock_train.py --epochs 5 --samples 64
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import sinabs
from model import SpeckDemoTrainer, BASE_CHANNELS, MAX_CHANNELS

T = 40          # mock clip length in 10ms bins -> ::4 -> 10 sim steps
BATCH = 4


def make_sample(label, rng):
    frames = np.zeros((T, 2, 128, 128), dtype=np.float32)
    rows = (slice(16, 56), slice(72, 112))[label]
    for t in range(T):
        n = rng.integers(200, 600)                      # variable activity like real clips
        ys = rng.integers(rows.start, rows.stop, n)
        xs = rng.integers(8, 120, n)
        ps = rng.integers(0, 2, n)
        frames[t, ps, ys, xs] = 1.0
        noise = rng.integers(0, 80)                     # background activity everywhere
        frames[t, rng.integers(0, 2, noise),
               rng.integers(0, 128, noise), rng.integers(0, 128, noise)] = 1.0
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="use BASE_CHANNELS not MAX")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--out", default="mock_speck.pt")
    args = ap.parse_args()

    channels = BASE_CHANNELS if args.base else MAX_CHANNELS
    print(f"channels={channels}  samples={args.samples}  epochs={args.epochs}")

    rng = np.random.default_rng(0)
    X = torch.stack([torch.from_numpy(make_sample(i % 2, rng))
                     for i in range(args.samples)])
    y = torch.tensor([i % 2 for i in range(args.samples)], dtype=torch.long)
    n_train = int(0.8 * args.samples)
    ds = torch.utils.data.TensorDataset(X[:n_train], y[:n_train])
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True,
                                         drop_last=True)
    Xv, yv = X[n_train:], y[n_train:]

    model = SpeckDemoTrainer(channels=channels, batch_size=BATCH)
    with torch.no_grad():                       # allocate v_mem before training
        _ = model(torch.zeros((BATCH, T, 2, 128, 128)))

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for data, target in loader:
            opt.zero_grad(set_to_none=True)
            sinabs.utils.reset_states(model.spiking_backbone)
            loss = crit(model(data), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()

        model.eval()
        with torch.no_grad():
            sinabs.utils.reset_states(model.spiking_backbone)
            acc = (model(Xv).argmax(1) == yv).float().mean().item() * 100
        print(f"epoch {ep+1}/{args.epochs}  loss {tot/len(loader):.4f}  val {acc:.0f}%")

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}  ({sum(p.numel() for p in model.parameters()):,} params)")


if __name__ == "__main__":
    main()
