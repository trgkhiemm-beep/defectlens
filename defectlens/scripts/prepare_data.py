"""Pre-process MVTec AD into a compact, reproducible, VALIDATED dataset.

  python scripts/prepare_data.py --src /kaggle/input/mvtec-ad --dst data/processed --overwrite

Output:
  <dst>/<category>/{train,val,test}/...png
  <dst>/manifest.csv        one row per image (source_path = clean image a synthetic defect was made from)
  <dst>/dataset_meta.json   git commit, code fingerprint, config, counts, validation result
  <dst>/prepare_data.log    full log of THIS run (never paste logs from memory: read this file)
Splits:
  train : (1 - val_ratio) of train/good          (fit the model)
  val   : val_ratio of train/good + synthetic     (pick threshold)
  test  : official MVTec test set                 (report ONLY, never tuned on)
Exit code 1 if validation fails -> a notebook / CI step stops right here.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.synthetic import SYNTHETIC_VERSION, inspection_region, synthesize_defect  # noqa: E402
from src.data.validate import FIELDS, code_fingerprint, preflight, validate_dataset  # noqa: E402

log = logging.getLogger("prepare_data")


def setup_logging(log_file: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def git_commit() -> str:
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
        return run("rev-parse", "--short", "HEAD") + ("-dirty" if run("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "no-git"


def resize_write(src: Path, dst: Path, size: int, is_mask: bool = False) -> np.ndarray:
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Unreadable file: {src}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)
    return img


def find_category_root(src: Path, cat: str) -> Path:
    # Kaggle mirrors sometimes nest folders (e.g. mvtec-ad/metal_nut/metal_nut/...).
    for p in [src / cat, *src.glob(f"*/{cat}"), *src.glob(f"{cat}/{cat}")]:
        if (p / "train" / "good").is_dir():
            return p
    raise FileNotFoundError(f"Category {cat!r} not found under {src}")


def process_category(src: Path, dst: Path, cat: str, cfg: dict, rng: np.random.Generator) -> list[dict]:
    size, ccfg, rows = cfg["image_size"], cfg["categories"][cat], []
    root = find_category_root(src, cat)
    rel = lambda p: p.relative_to(dst).as_posix()  # noqa: E731
    row = lambda split, img, mask="", label=0, dtype="good", source="": dict(  # noqa: E731
        split=split, category=cat, image_path=rel(img), mask_path=mask, label=label,
        defect_type=dtype, source_path=source)

    # train / val (normal only)
    goods = sorted((root / "train" / "good").glob("*.png"))
    n_val = max(1, int(len(goods) * cfg["val_ratio"]))
    val_goods: list[tuple[Path, np.ndarray]] = []
    for rank, i in enumerate(rng.permutation(len(goods))):
        split = "val" if rank < n_val else "train"
        out = dst / cat / split / "good" / goods[i].name
        img = resize_write(goods[i], out, size)
        rows.append(row(split, out))
        if split == "val":
            val_goods.append((out, img))

    # val synthetic defects, generated from held-out normals only
    for k in range(cfg["synthetic_per_category"]):
        base_path, base = val_goods[k % len(val_goods)]
        region = inspection_region(base, ccfg.get("foreground", "none"), ccfg.get("roi"))
        bad, mask, kind = synthesize_defect(base, rng, region=region,
                                            min_contrast=cfg.get("min_contrast", 20.0))
        img_out = dst / cat / "val" / "synthetic" / f"{k:03d}_{kind}.png"
        mask_out = dst / cat / "val" / "synthetic_mask" / f"{k:03d}_{kind}.png"
        for p, a in ((img_out, bad), (mask_out, mask * 255)):
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), a)
        rows.append(row("val", img_out, rel(mask_out), 1, f"synthetic_{kind}", rel(base_path)))

    # official test set
    for dtype_dir in sorted(p for p in (root / "test").iterdir() if p.is_dir()):
        dtype = dtype_dir.name
        for f in sorted(dtype_dir.glob("*.png")):
            out = dst / cat / "test" / dtype / f.name
            resize_write(f, out, size)
            mask_rel = ""
            if dtype != "good":
                m_out = dst / cat / "test" / f"{dtype}_mask" / f.name
                resize_write(root / "ground_truth" / dtype / f"{f.stem}_mask.png", m_out, size, is_mask=True)
                mask_rel = rel(m_out)
            rows.append(row("test", out, mask_rel, int(dtype != "good"), dtype))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", default=Path("data/processed"), type=Path)
    ap.add_argument("--config", default=ROOT / "configs/data.yaml", type=Path)
    ap.add_argument("--overwrite", action="store_true", help="delete an existing --dst first")
    args = ap.parse_args()

    problems = preflight()
    if problems:
        sys.exit("PREFLIGHT FAILED (repo has mixed old/new files; replace the whole repo):\n  "
                 + "\n  ".join(problems))

    # Stale files from an older run silently mixing into a new dataset is a classic bug.
    if args.dst.exists() and any(args.dst.iterdir()):
        if not args.overwrite:
            sys.exit(f"{args.dst} is not empty. Re-run with --overwrite to rebuild it.")
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True, exist_ok=True)
    setup_logging(args.dst / "prepare_data.log")

    cfg = yaml.safe_load(args.config.read_text())
    run_id, commit = uuid.uuid4().hex[:8], git_commit()
    log.info("run=%s code=%s synthetic_v%d val_ratio=%s seed=%s", run_id, commit, SYNTHETIC_VERSION,
             cfg["val_ratio"], cfg["seed"])
    if commit.endswith("-dirty") or commit == "no-git":
        log.warning("code is not a clean git commit -> results are harder to reproduce")

    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for cat in cfg["categories"]:
        cat_rows = process_category(args.src, args.dst, cat, cfg, rng)
        rows += cat_rows
        c = Counter((r["split"], r["label"]) for r in cat_rows)
        log.info("[%s] %s", cat, "  ".join(f"{s}/{'bad' if l else 'good'}={n}" for (s, l), n in sorted(c.items())))

    with open(args.dst / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "code_fingerprint": code_fingerprint(),
        "synthetic_version": SYNTHETIC_VERSION,
        "source": str(args.src),
        "config": cfg,
        "n_rows": len(rows),
    }
    meta_path = args.dst / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    errors, stats = validate_dataset(args.dst, cfg)
    meta["stats"] = stats
    meta["validation"] = {"passed": not errors, "errors": errors}
    meta_path.write_text(json.dumps(meta, indent=2))
    for cat, s in stats.items():
        log.info("[%s] contrast min/median=%s/%s kinds=%s", cat, s["contrast_min"], s["contrast_median"], s["kinds"])
    for e in errors:
        log.error(e)
    if errors:
        log.error("VALIDATION FAILED (%d errors) run=%s -> do NOT train on this dataset", len(errors), run_id)
        sys.exit(1)
    log.info("VALIDATION PASSED run=%s rows=%d -> %s", run_id, len(rows), args.dst)


if __name__ == "__main__":
    main()
