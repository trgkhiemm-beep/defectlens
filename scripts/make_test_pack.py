"""Pack a small, labelled set of REAL test images from a processed dataset (run on Kaggle),
to download and test the model locally, through the API or on the web demo.

  python scripts/make_test_pack.py --data /kaggle/working/data/processed --out /kaggle/working/test_pack --per-type 5

Layout (folder names are the labels, which scripts/inspect_folder.py understands):
  <out>/<category>/good/*.png
  <out>/<category>/<defect_type>/*.png
  <out>/<category>/ground_truth/<defect_type>/*.png     (defect masks, --with-masks)
  <out>/manifest.csv                                   (category, label, defect_type, file)
Only the official TEST split is used (never train/val), and images are 256x256 as the model sees them.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path("data/processed"), type=Path)
    ap.add_argument("--out", default=Path("test_pack"), type=Path)
    ap.add_argument("--per-type", type=int, default=5, help="images per (category, defect type); 0 = all")
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--with-masks", action="store_true")
    args = ap.parse_args()

    with open(args.data / "manifest.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"
                and (not args.categories or r["category"] in args.categories)]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["image_path"]):
        groups[(r["category"], r["defect_type"])].append(r)

    if args.out.exists():
        shutil.rmtree(args.out)
    out_rows = []
    for (cat, dtype), items in sorted(groups.items()):
        for r in items[: args.per_type or None]:
            dst = args.out / cat / dtype / Path(r["image_path"]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(args.data / r["image_path"], dst)
            if args.with_masks and r["mask_path"]:
                m = args.out / cat / "ground_truth" / dtype / Path(r["mask_path"]).name
                m.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(args.data / r["mask_path"], m)
            out_rows.append({"category": cat, "label": r["label"], "defect_type": dtype,
                             "file": dst.relative_to(args.out).as_posix()})

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "label", "defect_type", "file"])
        w.writeheader()
        w.writerows(out_rows)
    counts = defaultdict(int)
    for r in out_rows:
        counts[(r["category"], r["defect_type"])] += 1
    for (cat, dtype), n in sorted(counts.items()):
        print(f"{cat:11s} {dtype:22s} {n}")
    print(f"{len(out_rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
