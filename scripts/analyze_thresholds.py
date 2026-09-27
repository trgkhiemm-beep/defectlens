"""Operating-point analysis from saved image scores (CPU, seconds, no re-training).

  python scripts/analyze_thresholds.py --artifacts artifacts/patchcore

For each category: every threshold policy chosen on VAL, evaluated on TEST, next to
the test oracle; the val/test normal-score shift; and recall per defect type at the
primary operating point (which real defects the line would let through).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.metrics import best_f1_threshold, binary_stats, score_shift, select_threshold  # noqa: E402


def load(path: Path) -> tuple[dict, dict]:
    d = json.loads(path.read_text())
    conv = lambda s: {**s, "scores": np.array(s["scores"]), "labels": np.array(s["labels"])}  # noqa: E731
    return conv(d["val"]), conv(d["test"])


def analyze(cat: str, val: dict, test: dict, fprs: list[float], primary_fpr: float) -> str:
    oracle = best_f1_threshold(test["scores"], test["labels"])
    lines = [f"## {cat}", "", "| Policy (chosen on val) | Threshold | Test F1 | Recall | FPR | Gap to oracle |",
             "|---|---|---|---|---|---|"]
    policies = [("val_f1", "val_f1", None)] + [(f"normal_quantile @ FPR {f:.0%}", "normal_quantile", f) for f in fprs]
    for name, policy, f in policies:
        thr = select_threshold(val, policy, f or 0.05)
        s = binary_stats(test["scores"], test["labels"], thr)
        lines.append(f"| {name} | {thr:.3f} | {s['f1']:.3f} | {s['recall']:.3f} | {s['fpr']:.3f} | "
                     f"{oracle['f1'] - s['f1']:+.3f} |")
    lines.append(f"| *oracle (peeks at test)* | {oracle['threshold']:.3f} | {oracle['f1']:.3f} | "
                 f"{oracle['recall']:.3f} | {oracle['fpr']:.3f} | 0 |")

    sh = score_shift(val, test)
    lines += ["", f"Normal-score shift: val median {sh['val_normal_median']:.3f} -> test median "
              f"{sh['test_normal_median']:.3f} (x{sh['median_ratio_test_over_val']:.2f}); "
              f"p95 {sh['val_normal_p95']:.3f} -> {sh['test_normal_p95']:.3f}"]

    thr = select_threshold(val, "normal_quantile", primary_fpr)
    types = np.array(test["defect_types"])
    lines += ["", f"Recall per defect type @ normal_quantile FPR {primary_fpr:.0%}:", ""]
    for t in sorted(set(types[test["labels"] == 1])):
        sel = types == t
        lines.append(f"- `{t}`: {(test['scores'][sel] >= thr).mean():.0%} of {sel.sum()}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=Path("artifacts/patchcore"), type=Path)
    ap.add_argument("--fprs", nargs="*", type=float, default=[0.01, 0.02, 0.05, 0.10])
    ap.add_argument("--primary-fpr", type=float, default=0.05)
    args = ap.parse_args()

    files = sorted(args.artifacts.glob("*/scores.json"))
    if not files:
        sys.exit(f"no */scores.json under {args.artifacts} (train with the current train_patchcore.py)")
    report = "\n".join(analyze(f.parent.name, *load(f), args.fprs, args.primary_fpr) for f in files)
    (args.artifacts / "thresholds.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
