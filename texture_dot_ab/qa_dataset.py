"""qa_dataset.py — data-cleaning inspection for dataset_tex clips.

Walks every .h5 under a dataset dir and validates collection quality:
  - all 4 phase marks present (approach/touch/touch_10s/retreat)
  - touch duration >= 10 s (held past the beep)
  - minimum event count / no empty clips
  - event timestamps monotonic
  - phase_idx consistency (indices inside the stream, ordered)

Prints a per-clip table + summary. Exit code 1 if any clip FAILs —
wire this into the workflow before prep/training.

    python qa_dataset.py --root ../data/dataset_tex/dot
    python qa_dataset.py --root ../data/dataset_tex        # both conditions
"""

import argparse
import glob
import os

import h5py
import numpy as np

REQUIRED_PHASES = {"approach", "touch", "touch_10s", "retreat"}
MIN_EVENTS = 50_000       # aborted clips sit far below this
MIN_TOUCH_S = 9.9         # allow 0.1 s slack on the 10 s timer


def check_clip(path, min_events=MIN_EVENTS):
    issues = []
    with h5py.File(path) as f:
        n = f.attrs.get("event_count", len(f["x"]))
        t = f["t"][:]
        dur = float(t[-1] - t[0]) / 1e6 if len(t) > 1 else 0.0
        names = [x.decode() for x in f["phase_name"][:]] if "phase_name" in f else []
        idx = f["phase_idx"][:].tolist() if "phase_idx" in f else []
        td = f.attrs.get("touch_duration_s")

        if n < min_events:
            issues.append(f"low events ({n:,})")
        if len(t) > 1 and np.any(np.diff(t.astype(np.int64)) < 0):
            issues.append("timestamps non-monotonic")
        missing = REQUIRED_PHASES - set(names)
        if missing:
            issues.append(f"missing phases: {sorted(missing)}")
        if td is not None and td < MIN_TOUCH_S:
            issues.append(f"short touch ({td:.2f}s)")
        if "touch" in names and "touch_10s" in names and idx:
            i0, i1 = idx[names.index("touch")], idx[names.index("touch_10s")]
            if not (0 <= i0 < i1 <= n):
                issues.append(f"bad phase_idx span {i0}->{i1} of {n:,}")
        touch_ev = 0
        if "touch" in names and "touch_10s" in names and idx:
            touch_ev = idx[names.index("touch_10s")] - idx[names.index("touch")]
    return dict(path=path, events=n, dur=dur, phases=len(names),
                touch_s=td, touch_ev=touch_ev, issues=issues)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--min-events", type=int, default=MIN_EVENTS)
    args = ap.parse_args()

    h5s = sorted(glob.glob(os.path.join(args.root, "**", "*.h5"), recursive=True))
    if not h5s:
        raise SystemExit(f"no .h5 under {args.root}")

    n_fail = 0
    print(f"{'clip':52s} {'events':>10s} {'dur':>6s} {'touch':>6s} "
          f"{'touch_ev':>9s}  status")
    for p in h5s:
        r = check_clip(p, args.min_events)
        rel = p.split(args.root.rstrip("/") + "/")[-1] if args.root in p \
            else os.path.relpath(p, args.root)
        ok = not r["issues"]
        n_fail += 0 if ok else 1
        status = "ok" if ok else "FAIL: " + "; ".join(r["issues"])
        td = f"{r['touch_s']:.2f}" if isinstance(r["touch_s"], float) else "  -"
        print(f"{rel:52s} {r['events']:>10,} {r['dur']:>5.1f}s {td:>6s} "
              f"{r['touch_ev']:>9,}  {status}")

    n_ok = len(h5s) - n_fail
    print(f"\n{n_ok}/{len(h5s)} clips pass" +
          (f"  — {n_fail} need attention (exclude or re-record)" if n_fail else ""))
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
