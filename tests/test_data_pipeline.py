"""Data pipeline tests on a fake MVTec-shaped folder: python -m pytest tests -q"""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import AnomalyDataset, make_loader  # noqa: E402
from src.data.synthetic import inspection_region, synthesize_defect  # noqa: E402
from src.data.transforms import CORRUPTIONS, build_corruption_transform, build_eval_transform, build_train_transform  # noqa: E402
from src.data.validate import preflight, validate_dataset  # noqa: E402

from conftest import TRANSISTOR_ROI, _fake_nut, _fake_transistor  # noqa: E402


def _tampered(built, tmp_path):
    dst, cfg, *_ = built
    copy = tmp_path / "copy"
    shutil.copytree(dst, copy)
    with open(copy / "manifest.csv", newline="") as f:
        synth = [r for r in csv.DictReader(f)
                 if r["category"] == "transistor" and r["defect_type"].startswith("synthetic_")]
    return copy, cfg, synth[0]


# ---------- generator ----------
def test_synthetic_defect_is_inside_object():
    rng = np.random.default_rng(1)
    img = _fake_nut(rng)
    region = inspection_region(img, foreground="color")
    for _ in range(30):
        bad, mask, kind = synthesize_defect(img, rng, region=region)
        ys, xs = np.nonzero(mask)
        assert len(ys) and ((ys - 200) ** 2 + (xs - 200) ** 2 <= 120 ** 2).all(), kind
        assert (bad[mask == 0] == img[mask == 0]).all(), "pixels changed outside the mask"


def test_roi_defect_is_single_visible_region():
    rng = np.random.default_rng(2)
    img = _fake_transistor(rng)
    region = inspection_region(img, roi=TRANSISTOR_ROI)
    for _ in range(30):
        bad, mask, kind = synthesize_defect(img, rng, region=region)
        assert not (mask & (1 - region)).any(), kind
        if kind != "scratch":
            assert cv2.connectedComponents(mask)[0] == 2, kind
        diff = np.abs(bad.astype(int) - img.astype(int))[mask.astype(bool)].mean()
        assert diff >= 20, f"{kind} too faint: {diff:.1f}"


# ---------- build + validation gate ----------
def test_preflight_passes_on_consistent_repo():
    assert preflight() == []


def test_preflight_catches_old_synthetic_module(monkeypatch):
    from src.data import synthetic, validate
    monkeypatch.setattr(validate, "SYNTHETIC_VERSION", 2)
    monkeypatch.setattr(synthetic, "mask_contrast", lambda a, b: 0.0)  # old 2-arg API
    errors = preflight()
    assert any("v2" in e for e in errors) and any("API mismatch" in e for e in errors)


def test_build_passes_validation(built):
    dst, cfg, *_ = built
    errors, stats = validate_dataset(dst, cfg)
    assert errors == []
    assert stats["transistor"]["val_synthetic"] == 12
    assert (dst / "prepare_data.log").read_text().count("VALIDATION PASSED") == 1


def test_refuses_to_reuse_non_empty_dst(built):
    dst, _, src, cfg_path = built
    r = subprocess.run([sys.executable, "scripts/prepare_data.py", "--src", str(src), "--dst", str(dst),
                        "--config", str(cfg_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "--overwrite" in r.stderr


def test_gate_catches_defect_outside_roi(built, tmp_path):
    data, cfg, r = _tampered(built, tmp_path)
    mask = cv2.imread(str(data / r["mask_path"]), cv2.IMREAD_GRAYSCALE)
    mask[:10, :10] = 255  # top-left corner = PCB, outside the ROI
    cv2.imwrite(str(data / r["mask_path"]), mask)
    errors, _ = validate_dataset(data, cfg)
    assert any("outside the inspection region" in e for e in errors)


def test_gate_catches_invisible_defect(built, tmp_path):
    data, cfg, r = _tampered(built, tmp_path)
    shutil.copy(data / r["source_path"], data / r["image_path"])  # "defect" == clean image
    errors, _ = validate_dataset(data, cfg)
    assert any("nearly invisible" in e for e in errors)


def test_gate_catches_stale_config(built):
    dst, cfg, *_ = built
    errors, _ = validate_dataset(dst, {**cfg, "val_ratio": 0.5})
    assert any("different configs/data.yaml" in e for e in errors)


# ---------- loading ----------
@pytest.mark.parametrize("split,expected_labels", [("train", {0}), ("val", {0, 1}), ("test", {0, 1})])
def test_splits(built, split, expected_labels):
    ds = AnomalyDataset(built[0], "metal_nut", split, build_eval_transform(256))
    assert {int(ds[i]["label"]) for i in range(len(ds))} == expected_labels
    s = ds[len(ds) - 1]
    assert s["image"].shape == (3, 256, 256) and s["mask"].shape == (1, 256, 256)
    assert s["mask"].max() <= 1


def test_loader_and_transforms(built):
    aug = {"hflip": True, "vflip": True, "rot90": True, "max_rotate": 3}
    ds = AnomalyDataset(built[0], "transistor", "train", build_train_transform(256, aug))
    batch = next(iter(make_loader(ds, batch_size=4, num_workers=0)))
    assert batch["image"].shape == (4, 3, 256, 256)
    for name in CORRUPTIONS:
        ds_c = AnomalyDataset(built[0], "transistor", "test", build_corruption_transform(name, 256))
        assert ds_c[0]["image"].shape == (3, 256, 256), name


def test_make_test_pack_uses_only_test_split(built, tmp_path):
    out = tmp_path / "pack"
    subprocess.run([sys.executable, "scripts/make_test_pack.py", "--data", str(built[0]), "--out", str(out),
                    "--per-type", "2", "--with-masks"], cwd=ROOT, check=True)
    with open(out / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {(r["category"], r["defect_type"]) for r in rows} == {
        (c, t) for c in ("metal_nut", "transistor") for t in ("good", "scratch")}
    assert all((out / r["file"]).exists() and "/test/" not in r["file"] for r in rows)
    assert len(rows) == 8 and len(list(out.rglob("ground_truth/*/*.png"))) == 4
