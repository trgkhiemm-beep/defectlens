from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.data.synthetic import SYNTHETIC_VERSION, inspection_region, mask_contrast

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["split", "category", "image_path", "mask_path", "label", "defect_type", "source_path"]
# Files whose content defines the dataset; their hash goes into dataset_meta.json.
FINGERPRINT_FILES = ["configs/data.yaml", "scripts/prepare_data.py", "src/data/synthetic.py"]


def code_fingerprint() -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:12] for f in FINGERPRINT_FILES}


def _read(path: Path, flags: int) -> np.ndarray:
    img = cv2.imread(str(path), flags)
    if img is None:
        raise IOError(path)
    return img


def validate_dataset(data: Path, cfg: dict, check_meta: bool = True) -> tuple[list[str], dict]:
    """Return (errors, per-category stats). Empty errors == dataset is valid."""
    errors: list[str] = []
    stats: dict = {}

    if check_meta:
        meta_path = data / "dataset_meta.json"
        if not meta_path.exists():
            return [f"missing {meta_path} (dataset not built by prepare_data.py?)"], stats
        meta = json.loads(meta_path.read_text())
        if meta.get("synthetic_version") != SYNTHETIC_VERSION:
            errors.append(f"dataset synthetic_v{meta.get('synthetic_version')} != code v{SYNTHETIC_VERSION}: rebuild")
        if meta.get("config") != cfg:
            errors.append("dataset was built with a different configs/data.yaml: rebuild")
        if meta.get("code_fingerprint") != code_fingerprint():
            errors.append("prepare_data.py / synthetic.py / data.yaml changed since build: rebuild")

    with open(data / "manifest.csv", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            return errors + [f"manifest columns {reader.fieldnames} != {FIELDS}"], stats
        rows = list(reader)

    for r in rows:
        for key in ("image_path", "mask_path", "source_path"):
            if r[key] and not (data / r[key]).exists():
                errors.append(f"missing file {r[key]}")
    if errors:
        return errors, stats

    min_val_good = cfg.get("min_val_good", 30)
    min_contrast = cfg.get("min_contrast", 20.0)
    for cat, ccfg in cfg["categories"].items():
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            errors.append(f"[{cat}] no rows")
            continue
        split = lambda s, lbl=None: [r for r in cat_rows if r["split"] == s and (lbl is None or r["label"] == lbl)]  # noqa: E731
        name = lambda r: Path(r["image_path"]).name  # noqa: E731

        if any(r["label"] != "0" for r in split("train")):
            errors.append(f"[{cat}] anomalous image in train split")
        if {name(r) for r in split("train")} & {name(r) for r in split("val", "0")}:
            errors.append(f"[{cat}] train/val normal images overlap (leakage)")
        if len(split("val", "0")) < min_val_good:
            errors.append(f"[{cat}] only {len(split('val', '0'))} val normals (< {min_val_good})")

        synth = [r for r in split("val", "1") if r["defect_type"].startswith("synthetic_")]
        if len(synth) != cfg["synthetic_per_category"]:
            errors.append(f"[{cat}] {len(synth)} synthetic defects != {cfg['synthetic_per_category']}")

        contrasts, kinds = [], Counter()
        for r in synth:
            img = _read(data / r["image_path"], cv2.IMREAD_COLOR)
            src = _read(data / r["source_path"], cv2.IMREAD_COLOR)
            mask = (_read(data / r["mask_path"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            kind = r["defect_type"].removeprefix("synthetic_")
            kinds[kind] += 1
            tag = f"[{cat}] {name(r)}"
            if mask.sum() == 0:
                errors.append(f"{tag}: empty mask")
                continue
            region = inspection_region(src, ccfg.get("foreground", "none"), ccfg.get("roi"))
            outside = int((mask & (1 - region)).sum())
            if outside:
                errors.append(f"{tag}: {outside} mask pixels outside the inspection region")
            if np.abs(img.astype(np.int16) - src.astype(np.int16))[mask == 0].max() > 0:
                errors.append(f"{tag}: image changed outside the mask (mask is not ground truth)")
            if kind != "scratch" and cv2.connectedComponents(mask)[0] != 2:
                errors.append(f"{tag}: blob defect has more than one region")
            c = mask_contrast(src, img, mask)
            contrasts.append(c)
            if c < min_contrast:
                errors.append(f"{tag}: defect nearly invisible (contrast {c:.1f} < {min_contrast})")

        stats[cat] = {
            "train_good": len(split("train")), "val_good": len(split("val", "0")),
            "val_synthetic": len(synth), "test": len(split("test")),
            "kinds": dict(kinds),
            "contrast_min": round(min(contrasts), 1) if contrasts else None,
            "contrast_median": round(float(np.median(contrasts)), 1) if contrasts else None,
        }
    return errors, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path("data/processed"), type=Path)
    ap.add_argument("--config", default=ROOT / "configs/data.yaml", type=Path)
    args = ap.parse_args()
    errors, stats = validate_dataset(args.data, yaml.safe_load(args.config.read_text()))
    for cat, s in stats.items():
        print(f"[{cat}] {s}")
    for e in errors:
        print(f"ERROR {e}")
    print("VALIDATION PASSED" if not errors else f"VALIDATION FAILED ({len(errors)} errors)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
