from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path("data/processed"), type=Path)
    ap.add_argument("--out", default=Path("reports/synthetic_preview"), type=Path)
    ap.add_argument("--n", default=8, type=int, help="samples per category (even number)")
    args = ap.parse_args()

    with open(args.data / "manifest.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["defect_type"].startswith("synthetic_")]
    args.out.mkdir(parents=True, exist_ok=True)

    for cat in sorted({r["category"] for r in rows}):
        tiles = []
        for r in [r for r in rows if r["category"] == cat][: args.n]:
            img = cv2.imread(str(args.data / r["image_path"]))
            mask = cv2.imread(str(args.data / r["mask_path"]), cv2.IMREAD_GRAYSCALE)
            overlay = img.copy()
            contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
            cv2.putText(overlay, r["defect_type"][10:], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(np.hstack([img, overlay]))
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)])
        cv2.imwrite(str(args.out / f"{cat}.png"), grid)
        print(f"[{cat}] -> {args.out / f'{cat}.png'}")


if __name__ == "__main__":
    main()
