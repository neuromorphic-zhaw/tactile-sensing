"""N-class Speck-fitting trainer for texture_dot_ab — faithful port of
curvature/speck_compatible/devin_evcount/evT166_s2/train.py (the deployment-proven
recipe) parameterized for an arbitrary class list.

Same backbone (verified 9-core topology, MAX taper), same splits (70/15/15,
seed 42), same augmentations (time-roll ±12 always; drop 10%; stretch
0.8–1.2×), same readout (mean spike-rate logits -> CE), same weight clamp
|W|<=1 for chip compatibility. Only the readout width changes to N classes.

Run once per condition; identical seed/hyperparams keep the A/B controlled.
Saves best.pt, metrics.json, test_predictions.npz (for eval_compare.py),
confusion.png into --out.
"""

import argparse
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sinabs
import sinabs.layers as sl
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="evT npz dir for ONE condition")
    ap.add_argument("--out", required=True, help="run dir for checkpoints/metrics")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", default="")
    ap.add_argument("--time_stride", type=int, default=2, help="s2 recipe = 2")
    return ap.parse_args()


class TactileDataset(Dataset):
    def __init__(self, data_dir):
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "**/*.npz"),
                                          recursive=True))
        if not self.file_list:
            raise FileNotFoundError(f"No .npz files in {data_dir}")
        print(f"Dataset: {len(self.file_list)} samples")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        with np.load(self.file_list[index]) as d:
            data = torch.from_numpy(d["frames"]).float()
            label = torch.tensor(int(d["label"]), dtype=torch.long)
        if torch.is_grad_enabled():
            if np.random.rand() > 0.5:
                data = torch.roll(data, shifts=int(np.random.randint(-12, 12)), dims=0)
            drop = (torch.rand_like(data) < 0.10).float()
            data = data * (1.0 - drop)
            s = float(np.random.uniform(0.8, 1.2))
            t = data.shape[0]
            new_t = max(2, int(round(t / s)))
            idx = torch.linspace(0, t - 1, new_t).round().long().clamp(0, t - 1)
            d2 = data[idx]
            if new_t < t:
                data = torch.cat([d2, torch.zeros((t - new_t,) + d2.shape[1:],
                                                  dtype=d2.dtype)], dim=0)
            else:
                off = int(np.random.randint(0, new_t - t + 1))
                data = d2[off:off + t]
        return torch.clamp(data, 0.0, 1.0), label


def _iaf(bs):
    return sl.IAFSqueeze(batch_size=bs, spike_threshold=1.0, min_v_mem=-1.0,
                         surrogate_grad_fn=sinabs.activation.PeriodicExponential())


class SpeckFittingTrainer(nn.Module):
    """Verified 9-core topology; readout widened to n_classes."""

    def __init__(self, bs, n_classes, time_stride):
        super().__init__()
        self.time_stride = time_stride
        self.spiking_backbone = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=4, stride=2, padding=1, bias=False),
            _iaf(bs),
            nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1, bias=False),
            _iaf(bs),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            _iaf(bs),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            _iaf(bs),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            _iaf(bs),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            _iaf(bs),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            _iaf(bs),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            _iaf(bs),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256, n_classes, bias=False),
            _iaf(bs),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x[:, ::self.time_stride]
        nt = x.shape[1]
        out = self.spiking_backbone(x.reshape(b * nt, c, h, w))
        return out.reshape(b, nt, -1).mean(dim=1)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_json = os.path.join(args.data, "labels.json")
    if not os.path.exists(labels_json):
        raise FileNotFoundError(f"{labels_json} missing — run prep_count_bins.py first")
    classes = json.load(open(labels_json))["classes"]
    n_classes = len(classes)

    full = TactileDataset(args.data)
    n = len(full)
    train_sz, val_sz = int(0.70 * n), int(0.15 * n)
    test_sz = n - train_sz - val_sz
    tr, va, te = torch.utils.data.random_split(
        full, [train_sz, val_sz, test_sz],
        generator=torch.Generator().manual_seed(args.seed))
    # persist test indices so the A/B uses identical held-out logic
    json.dump({"test_indices": te.indices, "classes": classes,
               "seed": args.seed},
              open(os.path.join(args.out, "split.json"), "w"))

    loader_args = dict(batch_size=args.batch, num_workers=4, pin_memory=True,
                       persistent_workers=True, drop_last=True)
    train_loader = DataLoader(tr, shuffle=True, **loader_args)
    val_loader = DataLoader(va, shuffle=False, **loader_args)
    test_loader = DataLoader(te, shuffle=False, **loader_args)
    print(f"Classes {classes} | split {len(tr)}/{len(va)}/{len(te)}")

    model = SpeckFittingTrainer(args.batch, n_classes, args.time_stride).to(device)
    if args.pretrained:
        sd = torch.load(args.pretrained, map_location=device)
        model.load_state_dict({k: v for k, v in sd.items() if not k.endswith(".v_mem")},
                              strict=False)
        print(f"Warm-started from {args.pretrained}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val, best_path = 0.0, os.path.join(args.out, "best.pt")
    for epoch in range(args.epochs):
        model.train()
        t0, tot_loss, nb = time.time(), 0.0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            sinabs.utils.reset_states(model.spiking_backbone)
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        m.weight.clamp_(-1.0, 1.0)
            if not np.isfinite(loss.item()):
                print("NaN/inf loss — aborting run"); return
            tot_loss += loss.item(); nb += 1

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                sinabs.utils.reset_states(model.spiking_backbone)
                vp.extend(model(data).argmax(1).cpu().numpy())
                vt.extend(target.cpu().numpy())
        vacc = 100.0 * np.mean(np.array(vp) == np.array(vt))
        marker = ""
        if vacc > best_val:
            best_val = vacc
            torch.save(model.state_dict(), best_path)
            marker = "  [BEST]"
        print(f"ep {epoch+1:03d}/{args.epochs} | loss {tot_loss/max(nb,1):.4f} | "
              f"val {vacc:.2f}% | best {best_val:.2f}% | "
              f"{time.time()-t0:.0f}s{marker}", flush=True)
        scheduler.step()

    sd = torch.load(best_path, map_location=device)
    model.load_state_dict({k: v for k, v in sd.items() if not k.endswith(".v_mem")},
                          strict=False)
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            sinabs.utils.reset_states(model.spiking_backbone)
            tp.extend(model(data).argmax(1).cpu().numpy())
            tt.extend(target.cpu().numpy())
    tp, tt = np.array(tp), np.array(tt)
    test_acc = 100.0 * np.mean(tp == tt)
    print(f"\nFINAL TEST ACCURACY: {test_acc:.2f}%")
    print(classification_report(tt, tp, target_names=classes, zero_division=0))

    np.savez(os.path.join(args.out, "test_predictions.npz"),
             y_true=tt, y_pred=tp, test_indices=np.array(te.indices))
    json.dump({"classes": classes, "best_val_acc": best_val,
               "test_acc": test_acc, "data": args.data, "seed": args.seed},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)

    cm = confusion_matrix(tt, tp, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"test {test_acc:.1f}%")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "confusion.png"))


if __name__ == "__main__":
    main()
