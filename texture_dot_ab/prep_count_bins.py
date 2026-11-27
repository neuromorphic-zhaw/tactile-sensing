"""h5 -> evT count-binned npz for texture_dot_ab.

Replicates the evT166 recipe used for dataset_curv (the deployment-proven
rasterization): each clip's events are split into N equal-count bins and each
bin is rasterized to a binary (2,128,128) frame — channel 0 = OFF events,
channel 1 = ON events, occupancy-clamped. Output contract matches the
speck-compatible trainers: 'frames' (T,2,128,128) + 'label'.

Uniformity note: count-binning makes tensor SHAPE uniform automatically —
raw clips do NOT need matching durations. What slicing controls is content:
which physical phase the bins span. Use the spacebar phase marks
(phase_idx) to cut a controlled window before binning:

  --slice touch    touch -> touch_10s   (exactly 10.00 s of contact — the
                   controlled comparison window; default)
  --slice contact  approach -> touch_10s (adds the approach transient)
  --slice all      everything (raw recorder output, includes uncontrolled
                   post-retreat dead time — diagnostic only)

Clips missing the needed phase marks are skipped (run qa_dataset.py first).

Usage (per condition):
    python prep_count_bins.py --src ../data/phase2_texture_dot_ab/dataset_tex/dot \
        --dst ../data/phase2_texture_dot_ab/dataset_tex_evT166/dot --bins 166 --slice touch
"""

import argparse
import glob
import json
import os

import h5py
import numpy as np


def clip_to_frames(x, y, p, n_bins, sensor=128):
    n = len(x)
    if n == 0:
        return np.zeros((n_bins, 2, sensor, sensor), dtype=np.uint8)
    order = np.argsort(range(n))  # events already recorded in time order
    x, y, p = np.asarray(x), np.asarray(y), np.asarray(p)
    # normalize polarity to {0,1}: map min->0 (OFF), max->1 (ON)
    uniq = np.unique(p)
    p_bin = (p == uniq.max()).astype(np.uint8)

    idx = np.array_split(np.arange(n), n_bins)
    frames = np.zeros((n_bins, 2, sensor, sensor), dtype=np.uint8)
    for b, sel in enumerate(idx):
        if len(sel) == 0:
            continue
        xb, yb, pb = x[sel], y[sel], p_bin[sel]
        np.add.at(frames[b], (pb, yb, xb), 1)
    frames[frames > 1] = 1  # occupancy clamp -> binary frames
    return frames


def slice_span(f, slc):
    """(i0, i1) event-index window for slicing before binning.
    touch = touch->touch_10s, contact = approach->touch_10s, all = full stream.
    Returns None if the clip lacks the required phase marks."""
    n = int(f.attrs.get("event_count", len(f["x"])))
    if slc == "all":
        return (0, n)
    if "phase_idx" not in f or "phase_name" not in f:
        return None
    names = [x.decode() for x in f["phase_name"][:]]
    idx = f["phase_idx"][:].tolist()
    if "touch_10s" not in names:
        return None
    i1 = idx[names.index("touch_10s")]
    start = "touch" if slc == "touch" else "approach"
    if start not in names:
        return None
    i0 = idx[names.index(start)]
    return (i0, i1) if 0 <= i0 < i1 <= n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset_tex/<condition> dir")
    ap.add_argument("--dst", required=True, help="output npz dir (per condition)")
    ap.add_argument("--bins", type=int, default=166)
    ap.add_argument("--slice", dest="slc", default="touch",
                    choices=["touch", "contact", "all"],
                    help="phase window to bin (default: touch)")
    ap.add_argument("--classes", default="flat,curved,edge,soft,grid",
                    help="comma list; order defines label ids. Ignored if labels.json exists in src.")
    args = ap.parse_args()

    # label map: reuse src/labels.json if present so both conditions agree
    labels_json = os.path.join(args.src, "labels.json")
    if os.path.exists(labels_json):
        classes = json.load(open(labels_json))["classes"]
    else:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    label_of = {c: i for i, c in enumerate(classes)}

    os.makedirs(args.dst, exist_ok=True)
    json.dump({"classes": classes, "slice": args.slc, "bins": args.bins},
              open(os.path.join(args.dst, "labels.json"), "w"))

    h5s = sorted(glob.glob(os.path.join(args.src, "**", "*.h5"), recursive=True))
    if not h5s:
        raise FileNotFoundError(f"No .h5 under {args.src}")

    n_done = n_skip = 0
    for path in h5s:
        cls = os.path.basename(os.path.dirname(path))
        if cls not in label_of:
            print(f"  skip (class '{cls}' not in label map): {path}")
            n_skip += 1
            continue
        with h5py.File(path) as f:
            span = slice_span(f, args.slc)
            if span is None:
                print(f"  skip (missing phase marks for --slice {args.slc}): {path}")
                n_skip += 1
                continue
            i0, i1 = span
            x, y, p = f["x"][i0:i1], f["y"][i0:i1], f["p"][i0:i1]
        frames = clip_to_frames(x, y, p, args.bins)
        out_dir = os.path.join(args.dst, cls)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, os.path.basename(path).replace(".h5", ".npz"))
        np.savez_compressed(out, frames=frames, label=np.int64(label_of[cls]))
        n_done += 1
        print(f"  {os.path.basename(path)} -> {out}  ({len(x):,} events sliced, label={label_of[cls]})")

    print(f"\nDone: {n_done} clips rasterized ({args.bins} bins, slice={args.slc}), {n_skip} skipped.")
    print(f"Label map: {label_of}")


if __name__ == "__main__":
    main()
