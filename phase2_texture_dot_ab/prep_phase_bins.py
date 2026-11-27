"""prep_phase_bins.py — build the PHASE-RECOGNITION dataset (track 2).

Slices each clip down to its useful segments and count-bins ONLY those;
every bin inherits a PHASE label from the clip's phase_idx marks:

  0 = approach   [approach press,  touch press)    — descent transient
  1 = touch      [touch press,     touch + CAP)    — contact, capped
  2 = retreat    [retreat press,   clip end)       — withdrawal transient
 -1 = excluded   only bins straddling a segment join (<=2 per clip)

The dead zones never enter the binning at all: pre-approach margin,
touch tail beyond the cap, and the cap..retreat-press lag are dropped
BEFORE binning — so bins are never wasted on content training discards.

Why the touch cap: stationary contact is homogeneous — the full 10 s
teaches nothing beyond the first few seconds, but would dominate the
bin allocation (count-binning allocates bins ∝ events) and starve the
transient phases. Default cap 3 s (--touch_cap_s).

Output npz per clip: frames (T,2,128,128) uint8, phase (T,) int8,
label (surface class) int64. Trainer/eval masks phase == -1.

Usage (per condition):
    python prep_phase_bins.py --src ../data/phase2_texture_dot_ab/dataset_tex_raw/dot \
        --dst ../data/phase2_texture_dot_ab/phase/evT166/dot --bins 166
"""

import argparse
import glob
import json
import os

import h5py
import numpy as np

PHASES = ["approach", "touch", "retreat"]


def clip_to_frames(x, y, p, n_bins, sensor=128):
    n = len(x)
    frames = np.zeros((n_bins, 2, sensor, sensor), dtype=np.uint8)
    if n == 0:
        return frames
    x, y, p = np.asarray(x), np.asarray(y), np.asarray(p)
    uniq = np.unique(p)
    p_bin = (p == uniq.max()).astype(np.uint8)
    idx = np.array_split(np.arange(n), n_bins)
    for b, sel in enumerate(idx):
        if len(sel) == 0:
            continue
        np.add.at(frames[b], (p_bin[sel], y[sel], x[sel]), 1)
    frames[frames > 1] = 1
    return frames


def phase_marks(f):
    """(approach_i, touch_i, retreat_i, touch_cap_i) event indices, or None."""
    if "phase_idx" not in f or "phase_name" not in f:
        return None
    names = [x.decode() for x in f["phase_name"][:]]
    idx = f["phase_idx"][:].tolist()
    if not all(k in names for k in ("approach", "touch", "retreat")):
        return None
    return (idx[names.index("approach")],
            idx[names.index("touch")],
            idx[names.index("retreat")])


def extract_segments(x, y, p, t, marks, touch_cap_s):
    """Slice the stream down to the useful parts and concatenate:
    [approach, touch_press) + [touch_press, touch+cap) + [retreat_press, end).
    Returns concatenated (x,y,p) + seg_edges — (start,end) index pairs of
    each phase within the concatenated array. The dead zones (pre-approach
    margin, touch tail beyond cap, cap..retreat lag) never enter the
    binning — no wasted bins."""
    ai, ti, ri = marks
    n = len(x)
    cap_i = min(int(np.searchsorted(t, t[min(ti, n - 1)] + touch_cap_s * 1e6)), n)
    segs = [(ai, ti), (ti, cap_i), (ri, n)]  # approach, touch(cap), retreat
    xs, ys, ps, edges, pos = [], [], [], [], 0
    for a, z in segs:
        xs.append(x[a:z]); ys.append(y[a:z]); ps.append(p[a:z])
        edges.append((pos, pos + (z - a)))
        pos += z - a
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ps), edges


def bin_phases(n_events, n_bins, seg_edges):
    """phase (n_bins,) int8; -1 only for bins straddling a segment join
    (at most 2 per clip). Everything else is fully inside one phase."""
    phase = np.full(n_bins, -1, dtype=np.int8)
    edges = np.array_split(np.arange(n_events), n_bins)
    for b, sel in enumerate(edges):
        if len(sel) == 0:
            continue
        lo, hi = sel[0], sel[-1] + 1
        for ph, (a, z) in enumerate(seg_edges):
            if lo >= a and hi <= z:
                phase[b] = ph
                break
    return phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset_tex_raw/<condition> dir")
    ap.add_argument("--dst", required=True, help="output npz dir (per condition)")
    ap.add_argument("--bins", type=int, default=166)
    ap.add_argument("--max_per_class", type=int, default=20,
                    help="cap clips per class (first N chronologically; filenames are date-prefixed so sorted == chronological)")
    ap.add_argument("--touch_cap_s", type=float, default=3.0)
    args = ap.parse_args()

    labels_json = os.path.join(args.src, "labels.json")
    classes = json.load(open(labels_json))["classes"]
    label_of = {c: i for i, c in enumerate(classes)}

    os.makedirs(args.dst, exist_ok=True)
    json.dump({"classes": classes, "phases": PHASES, "bins": args.bins,
               "touch_cap_s": args.touch_cap_s, "excluded": -1},
              open(os.path.join(args.dst, "labels.json"), "w"))

    h5s = sorted(glob.glob(os.path.join(args.src, "**", "*.h5"), recursive=True))
    if not h5s:
        raise FileNotFoundError(f"No .h5 under {args.src}")

    n_done = n_skip = 0
    hist = np.zeros(4, dtype=np.int64)  # per-phase bin counts + excluded
    per_class_count = {}
    for path in h5s:
        cls_name = os.path.basename(os.path.dirname(path))
        if args.max_per_class and per_class_count.get(cls_name, 0) >= args.max_per_class:
            continue
        cls = os.path.basename(os.path.dirname(path))
        if cls not in label_of:
            print(f"  skip (class '{cls}' not in label map): {path}")
            n_skip += 1
            continue
        with h5py.File(path) as f:
            marks = phase_marks(f)
            if marks is None:
                print(f"  skip (missing phase marks): {path}")
                n_skip += 1
                continue
            x, y, p, t = f["x"][:], f["y"][:], f["p"][:], f["t"][:]
        x, y, p, seg_edges = extract_segments(x, y, p, t, marks, args.touch_cap_s)
        n = len(x)
        frames = clip_to_frames(x, y, p, args.bins)
        phase = bin_phases(n, args.bins, seg_edges)
        for v, c in zip((0, 1, 2, -1), range(4)):
            hist[c] += int((phase == v).sum())
        out_dir = os.path.join(args.dst, cls)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, os.path.basename(path).replace(".h5", ".npz"))
        np.savez_compressed(out, frames=frames, phase=phase,
                            label=np.int64(label_of[cls]))
        per_class_count[cls] = per_class_count.get(cls, 0) + 1
        n_done += 1

    print(f"\nDone: {n_done} clips rasterized ({args.bins} bins, "
          f"touch_cap={args.touch_cap_s}s), {n_skip} skipped.")
    tot = hist.sum()
    print(f"bin allocation: approach {hist[0]/tot:.1%} | touch {hist[1]/tot:.1%} | "
          f"retreat {hist[2]/tot:.1%} | excluded {hist[3]/tot:.1%}")


if __name__ == "__main__":
    main()
