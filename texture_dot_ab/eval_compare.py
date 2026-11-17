"""Paired A/B comparison for texture_dot_ab: dot vs nodot.

Reads the two trainer run dirs (test_predictions.npz + split.json +
metrics.json) and reports the controlled comparison:

  - overall test accuracy per condition and delta (dot - nodot)
  - per-class accuracy table
  - both confusion matrices
  - paired McNemar test when clip order was matched across gels
    (--paired: aligns predictions by sorted test-file order; requires the
    same number of test clips per class in both conditions)

    python eval_compare.py --dot_run runs/dot_evT166 \
        --nodot_run runs/nodot_evT166 [--paired]
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix


def load_run(run_dir):
    preds = np.load(os.path.join(run_dir, "test_predictions.npz"))
    split = json.load(open(os.path.join(run_dir, "split.json")))
    metrics = json.load(open(os.path.join(run_dir, "metrics.json")))
    return preds["y_true"], preds["y_pred"], split["classes"], metrics


def per_class_acc(y_true, y_pred, n_classes):
    out = []
    for c in range(n_classes):
        m = y_true == c
        out.append(100.0 * np.mean(y_pred[m] == c) if m.any() else float("nan"))
    return out


def mcnemar(correct_a, correct_b):
    """McNemar's test on paired per-clip correctness. Returns (b, c, stat).
    b = A right/B wrong, c = A wrong/B right. chi2 with continuity fix."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    return b, c, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dot_run", required=True)
    ap.add_argument("--nodot_run", required=True)
    ap.add_argument("--paired", action="store_true",
                    help="align clips by test order; needs matched protocol")
    args = ap.parse_args()

    yt_d, yp_d, cls_d, m_d = load_run(args.dot_run)
    yt_n, yp_n, cls_n, m_n = load_run(args.nodot_run)
    if cls_d != cls_n:
        raise SystemExit(f"class mismatch: {cls_d} vs {cls_n}")
    classes, nc = cls_d, len(cls_d)

    acc_d = 100.0 * np.mean(yp_d == yt_d)
    acc_n = 100.0 * np.mean(yp_n == yt_n)

    print("=" * 56)
    print(f"  DOT    test acc: {acc_d:5.2f}%   (n={len(yt_d)})")
    print(f"  NODOT  test acc: {acc_n:5.2f}%   (n={len(yt_n)})")
    print(f"  DELTA (dot - nodot): {acc_d - acc_n:+5.2f} pts")
    print("=" * 56)

    print(f"\n{'class':<12}{'dot %':>8}{'nodot %':>9}{'delta':>8}")
    pd_, pn_ = per_class_acc(yt_d, yp_d, nc), per_class_acc(yt_n, yp_n, nc)
    for i, cname in enumerate(classes):
        print(f"{cname:<12}{pd_[i]:>8.2f}{pn_[i]:>9.2f}{pd_[i]-pn_[i]:>+8.2f}")

    for name, yt, yp in [("DOT", yt_d, yp_d), ("NODOT", yt_n, yp_n)]:
        print(f"\n{name} confusion (rows=true):")
        cm = confusion_matrix(yt, yp, labels=list(range(nc)))
        print("        " + " ".join(f"{c:>7}" for c in classes))
        for i, row in enumerate(cm):
            print(f"{classes[i]:<7} " + " ".join(f"{v:>7}" for v in row))

    if args.paired:
        if len(yt_d) != len(yt_n) or not np.array_equal(
                np.sort(yt_d), np.sort(yt_n)):
            print("\n[paired] test sets not aligned (different n or class "
                  "mix) — skipping McNemar")
        else:
            cd, cn = (yp_d == yt_d), (yp_n == yt_n)
            b, c, stat = mcnemar(cd, cn)
            print(f"\n[paired] McNemar: dot-only-correct={b}, "
                  f"nodot-only-correct={c}, chi2={stat:.3f}"
                  f"  (>3.84 ~ p<0.05)")

    print("\nNote: binned accuracy is a diagnostic. Confirm any claimed "
          "difference at event level (specksim) / chip replay before "
          "reporting it.")


if __name__ == "__main__":
    main()
