"""Visual QA of synthetic defects: image | image + mask contour + ROI, per category.
Every grid is stamped with the dataset's run id / commit / version, so a
screenshot always tells which build it came from.

  python scripts/preview_synthetic.py --data data/processed --out reports/synthetic_preview
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path("data/processed"), type=Path)
    ap.add_argument("--out", default=Path("reports/synthetic_preview"), type=Path)
    ap.add_argument("--n", default=8, type=int, help="samples per category (even number)")
    args = ap.parse_args()

    meta = json.loads((args.data / "dataset_meta.json").read_text())
    status = "PASSED" if meta.get("validation", {}).get("passed") else "NOT VALIDATED"
    stamp = f"run={meta['run_id']} code={meta['git_commit']} synthetic_v{meta['synthetic_version']} {status}"
    with open(args.data / "manifest.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["defect_type"].startswith("synthetic_")]
    args.out.mkdir(parents=True, exist_ok=True)

    for cat in sorted({r["category"] for r in rows}):
        roi = meta["config"]["categories"][cat].get("roi")
        tiles = []
        for r in [r for r in rows if r["category"] == cat][: args.n]:
            img = cv2.imread(str(args.data / r["image_path"]))
            mask = cv2.imread(str(args.data / r["mask_path"]), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            overlay = img.copy()
            if roi:
                cv2.rectangle(overlay, (int(roi[0] * w), int(roi[1] * h)), (int(roi[2] * w) - 1, int(roi[3] * h) - 1),
                              (255, 200, 0), 1)
            contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
            cv2.putText(overlay, Path(r["image_path"]).stem, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(np.hstack([img, overlay]))
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)])
        banner = np.full((28, grid.shape[1], 3), 32, np.uint8)
        cv2.putText(banner, f"{cat} | {stamp}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(str(args.out / f"{cat}.png"), np.vstack([banner, grid]))
        print(f"[{cat}] -> {args.out / f'{cat}.png'}")


if __name__ == "__main__":
    main()
