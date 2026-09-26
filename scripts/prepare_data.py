from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

# Thêm thư mục gốc dự án vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.synthetic import SYNTHETIC_VERSION, inspection_region, synthesize_defect  # noqa: E402

FIELDS = ["split", "category", "image_path", "mask_path", "label", "defect_type"]


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def resize_write(src: Path, dst: Path, size: int, is_mask: bool = False) -> np.ndarray:
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Unreadable file: {src}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)
    return img


def find_category_root(src: Path, cat: str) -> Path:
    for p in [src / cat, *src.glob(f"*/{cat}"), *src.glob(f"{cat}/{cat}")]:
        if (p / "train" / "good").is_dir():
            return p
    raise FileNotFoundError(f"Category {cat!r} not found under {src}")


def process_category(src: Path, dst: Path, cat: str, cfg: dict, rng: np.random.Generator) -> list[dict]:
    size, rows = cfg["image_size"], []
    root = find_category_root(src, cat)
    rel = lambda p: p.relative_to(dst).as_posix()  # noqa: E731

    # train / val (normal only)
    goods = sorted((root / "train" / "good").glob("*.png"))
    idx = rng.permutation(len(goods))
    n_val = max(1, int(len(goods) * cfg["val_ratio"]))
    val_goods = []
    for rank, i in enumerate(idx):
        split = "val" if rank < n_val else "train"
        out = dst / cat / split / "good" / goods[i].name
        img = resize_write(goods[i], out, size)
        rows.append(dict(split=split, category=cat, image_path=rel(out), mask_path="", label=0, defect_type="good"))
        if split == "val":
            val_goods.append(img)

    # Foreground & ROI config
    cat_cfg = cfg.get("categories", {}).get(cat, {})
    fg_cfg = cat_cfg.get("foreground", False)
    fg_mode = "color" if (isinstance(fg_cfg, bool) and fg_cfg) else (fg_cfg if isinstance(fg_cfg, str) else "none")
    roi_cfg = cat_cfg.get("roi", None)

    # val synthetic defects
    for k in range(cfg["synthetic_per_category"]):
        base = val_goods[k % len(val_goods)]
        region = inspection_region(base, foreground=fg_mode, roi=roi_cfg)
        bad, mask, kind = synthesize_defect(base, rng, region=region)
        
        img_out = dst / cat / "val" / "synthetic" / f"{k:03d}_{kind}.png"
        mask_out = dst / cat / "val" / "synthetic_mask" / f"{k:03d}_{kind}.png"
        img_out.parent.mkdir(parents=True, exist_ok=True)
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_out), bad)
        cv2.imwrite(str(mask_out), mask * 255)
        rows.append(dict(split="val", category=cat, image_path=rel(img_out), mask_path=rel(mask_out),
                         label=1, defect_type=f"synthetic_{kind}"))

    # official test set
    for dtype_dir in sorted(p for p in (root / "test").iterdir() if p.is_dir()):
        dtype = dtype_dir.name
        for f in sorted(dtype_dir.glob("*.png")):
            out = dst / cat / "test" / dtype / f.name
            resize_write(f, out, size)
            mask_rel = ""
            if dtype != "good":
                m_src = root / "ground_truth" / dtype / f"{f.stem}_mask.png"
                m_out = dst / cat / "test" / f"{dtype}_mask" / f.name
                resize_write(m_src, m_out, size, is_mask=True)
                mask_rel = rel(m_out)
            rows.append(dict(split="test", category=cat, image_path=rel(out), mask_path=mask_rel,
                             label=int(dtype != "good"), defect_type=dtype))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", default=Path("data/processed"), type=Path)
    ap.add_argument("--config", default=Path("configs/data.yaml"), type=Path)
    ap.add_argument("--categories", nargs="*", help="subset; default = all in config")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    args = ap.parse_args()

    if args.overwrite and args.dst.exists():
        shutil.rmtree(args.dst)
        print(f"Cleared existing output directory: {args.dst}")

    cfg = yaml.safe_load(args.config.read_text())
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for cat in args.categories or list(cfg["categories"]):
        cat_rows = process_category(args.src, args.dst, cat, cfg, rng)
        rows += cat_rows
        stats = Counter((r["split"], r["label"]) for r in cat_rows)
        print(f"[{cat}] " + "  ".join(f"{s}/{'bad' if l else 'good'}={n}" for (s, l), n in sorted(stats.items())))

    args.dst.mkdir(parents=True, exist_ok=True)
    
    # Ghi manifest.csv
    with open(args.dst / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.dst / 'manifest.csv'}")

    # Ghi dataset_meta.json
    counts = Counter(f"{r['category']}/{r['split']}/{'bad' if r['label'] else 'good'}" for r in rows)
    meta = {
        "git_commit": get_git_commit(),
        "synthetic_version": SYNTHETIC_VERSION,
        "config": cfg,
        "counts": dict(counts),
    }
    with open(args.dst / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata -> {args.dst / 'dataset_meta.json'}")


if __name__ == "__main__":
    main()