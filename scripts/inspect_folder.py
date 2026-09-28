"""Run the deployed model on every image in a folder and, when folder names carry labels,
score the results. Works on a test pack (scripts/make_test_pack.py) or on your own photos.

  python scripts/inspect_folder.py --images test_pack/transistor --category transistor
  python scripts/inspect_folder.py --images my_photos --category capsule --overlays

Labels from folder names: an image inside a folder named `good` is expected OK; inside any
other sub-folder (e.g. `scratch`, `bent_lead`) it is expected DEFECT, with that folder as the
defect type; images directly in --images, or under `ground_truth`, are skipped / unlabelled.
Outputs <out>/results.csv, <out>/summary.json and, with --overlays, heatmap images.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.service import render_overlay  # noqa: E402
from src.edge.runtime import EdgeInspector  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def label_of(path: Path, root: Path) -> tuple[str | None, str]:
    parts = path.relative_to(root).parts
    if "ground_truth" in parts:
        return None, "mask"
    if len(parts) < 2:
        return None, "unlabelled"
    folder = parts[-2]
    return ("OK", "good") if folder == "good" else ("DEFECT", folder)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--category", required=True)
    ap.add_argument("--model", default=ROOT / "deploy/model", type=Path)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--out", type=Path, help="default: reports/inspect/<category>")
    ap.add_argument("--overlays", action="store_true", help="save heatmap overlays")
    args = ap.parse_args()
    out = args.out or ROOT / "reports" / "inspect" / args.category
    out.mkdir(parents=True, exist_ok=True)

    insp = EdgeInspector.from_bundle(args.model, args.device, categories=[args.category])
    files = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    rows = []
    for p in files:
        expected, dtype = label_of(p, args.images)
        if dtype == "mask":
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip (unreadable): {p}")
            continue
        r = insp.predict(img, args.category)
        verdict = "DEFECT" if r["is_defect"] else "OK"
        rows.append({"file": p.relative_to(args.images).as_posix(), "defect_type": dtype, "expected": expected or "",
                     "verdict": verdict, "score": round(r["score"], 4), "threshold": round(r["threshold"], 4),
                     "ratio": round(r["score"] / r["threshold"], 3),
                     "correct": "" if expected is None else int(expected == verdict),
                     "latency_ms": round(r["timing_ms"]["total"], 1)})
        if args.overlays:
            dst = out / "overlays" / p.relative_to(args.images).with_suffix(".png")
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), np.hstack([cv2.resize(img, r["anomaly_map"].shape[::-1]),
                                             render_overlay(img, r["anomaly_map"], r["threshold"])]))
    if not rows:
        raise SystemExit(f"no images found under {args.images}")

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    labelled = [r for r in rows if r["expected"]]
    summary = {"category": args.category, "images": len(rows), "labelled": len(labelled),
               "latency_p50_ms": float(np.median([r["latency_ms"] for r in rows]))}
    if labelled:
        good = [r for r in labelled if r["expected"] == "OK"]
        bad = [r for r in labelled if r["expected"] == "DEFECT"]
        summary.update(
            accuracy=float(np.mean([r["correct"] for r in labelled])),
            false_reject_rate=float(np.mean([r["verdict"] == "DEFECT" for r in good])) if good else None,
            recall=float(np.mean([r["verdict"] == "DEFECT" for r in bad])) if bad else None,
            recall_per_defect_type={t: float(np.mean([r["verdict"] == "DEFECT" for r in bad if r["defect_type"] == t]))
                                    for t in sorted({r["defect_type"] for r in bad})},
            mistakes=[f"{r['file']} expected {r['expected']} got {r['verdict']} ({r['ratio']}x thr)"
                      for r in labelled if not r["correct"]])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    for r in rows:
        mark = "" if r["correct"] == "" else ("  ok" if r["correct"] else "  <-- WRONG")
        print(f"{r['file']:45s} {r['verdict']:6s} {r['ratio']:5.2f}x thr{mark}")
    print(json.dumps({k: v for k, v in summary.items() if k != "mistakes"}, indent=2))
    print(f"-> {out / 'results.csv'}")


if __name__ == "__main__":
    main()
