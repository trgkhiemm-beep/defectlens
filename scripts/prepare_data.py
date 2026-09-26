from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

# Thêm thư mục gốc dự án vào sys.path trước khi import từ src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.synthetic import inspection_region, synthesize_defect  # noqa: E402

FIELDS = ["split", "category", "image_path", "mask_path", "label", "defect_type"]


def resize_write(src: Path, dst: Path, size: int, is_mask: bool = False) -> np.ndarray:
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Unreadable file: {src}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)
    return img


def find_category_root(src: Path, cat: str) -> Path:
    # Kaggle mirrors sometimes nest folders (e.g. ipythonx/metal_nut/metal_nut/...).
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

    # Đọc cấu hình foreground & roi của category
    cat_cfg = cfg.get("categories", {}).get(cat, {})
    fg_cfg = cat_cfg.get("foreground", False)
    fg_mode = "color" if (isinstance(fg_cfg, bool) and fg_cfg) else (fg_cfg if isinstance(fg_cfg, str) else "none")
    roi_cfg = cat_cfg.get("roi", None)

    # val synthetic defects, generated from held-out normals (never from train split)
    for k in range(cfg["synthetic_per_category"]):
        base = val_goods[k % len(val_goods)]
        
        # 1. Tính vùng kiểm tra (region) tương thích synthetic_v2
        region = inspection_region(base, foreground=fg_mode, roi=roi_cfg)
        
        # 2. Sinh vết lỗi dựa trên region
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
    with open(args.dst / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.dst / 'manifest.csv'}")


if __name__ == "__main__":
    main()