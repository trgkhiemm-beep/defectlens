"""Smoke test on a fake MVTec-shaped folder: python -m pytest tests -q"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import AnomalyDataset, make_loader  # noqa: E402
from src.data.synthetic import synthesize_defect  # noqa: E402
from src.data.transforms import CORRUPTIONS, build_corruption_transform, build_eval_transform, build_train_transform  # noqa: E402


def _fake_object(rng):
    img = np.full((400, 400, 3), 30, np.uint8)  # dark background, bright object
    cv2.circle(img, (200, 200), 120, tuple(int(c) for c in rng.integers(150, 220, 3)), -1)
    return img


@pytest.fixture(scope="module")
def fake_mvtec(tmp_path_factory):
    src = tmp_path_factory.mktemp("mvtec")
    rng = np.random.default_rng(0)
    for cat in ["metal_nut", "carpet"]:
        root = src / cat
        for i in range(20):
            p = root / "train" / "good" / f"{i:03d}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), _fake_object(rng))
        for dtype in ["good", "scratch"]:
            for i in range(4):
                p = root / "test" / dtype / f"{i:03d}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), _fake_object(rng))
                if dtype != "good":
                    m = np.zeros((400, 400), np.uint8)
                    m[180:220, 180:220] = 255
                    mp = root / "ground_truth" / dtype / f"{i:03d}_mask.png"
                    mp.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(mp), m)
    dst = src.parent / "processed"
    subprocess.run([sys.executable, "scripts/prepare_data.py", "--src", str(src), "--dst", str(dst),
                    "--categories", "metal_nut", "carpet"], cwd=ROOT, check=True)
    return dst


def test_synthetic_defect_is_inside_object():
    rng = np.random.default_rng(1)
    img = _fake_object(rng)
    for _ in range(20):
        bad, mask, kind = synthesize_defect(img, rng, use_foreground=True)
        assert bad.shape == img.shape and mask.sum() > 0
        ys, xs = np.nonzero(mask)
        assert ((ys - 200) ** 2 + (xs - 200) ** 2 <= 120 ** 2).mean() > 0.95, kind


@pytest.mark.parametrize("split,expected_labels", [("train", {0}), ("val", {0, 1}), ("test", {0, 1})])
def test_splits(fake_mvtec, split, expected_labels):
    ds = AnomalyDataset(fake_mvtec, "metal_nut", split, build_eval_transform(256))
    assert {int(ds[i]["label"]) for i in range(len(ds))} == expected_labels
    s = ds[len(ds) - 1]
    assert s["image"].shape == (3, 256, 256) and s["mask"].shape == (1, 256, 256)
    assert s["mask"].max() <= 1


def test_train_has_no_overlap_with_val(fake_mvtec):
    tr = {r["image_path"].split("/")[-1] for r in AnomalyDataset(fake_mvtec, "carpet", "train", None).rows}
    va = {r["image_path"].split("/")[-1] for r in AnomalyDataset(fake_mvtec, "carpet", "val", None).rows
          if r["label"] == "0"}
    assert tr and va and not tr & va


def test_loader_and_transforms(fake_mvtec):
    aug = {"hflip": True, "vflip": True, "rot90": True, "max_rotate": 3}
    ds = AnomalyDataset(fake_mvtec, "carpet", "train", build_train_transform(256, aug))
    batch = next(iter(make_loader(ds, batch_size=4, num_workers=0)))
    assert batch["image"].shape == (4, 3, 256, 256)
    for name in CORRUPTIONS:
        ds_c = AnomalyDataset(fake_mvtec, "carpet", "test", build_corruption_transform(name, 256))
        assert ds_c[0]["image"].shape == (3, 256, 256), name
