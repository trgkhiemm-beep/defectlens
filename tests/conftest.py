"""Shared fixtures: a fake MVTec-shaped dataset built with the real CLI."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRANSISTOR_ROI = [0.28, 0.15, 0.78, 0.95]


def _fake_nut(rng):  # uniform dark background, bright object
    img = np.full((400, 400, 3), 30, np.uint8)
    cv2.circle(img, (200, 200), 120, tuple(int(c) for c in rng.integers(150, 220, 3)), -1)
    return img


def _fake_transistor(rng):  # cluttered background, dark part in the middle
    img = rng.integers(90, 200, (400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (130, 90), (290, 330), (25, 25, 25), -1)
    return img


def _write(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Build a dataset with the real CLI; return (dst, cfg, src, cfg_path)."""
    base = tmp_path_factory.mktemp("pipeline")
    src, rng = base / "mvtec", np.random.default_rng(0)
    makers = {"metal_nut": _fake_nut, "transistor": _fake_transistor}
    for cat, make in makers.items():
        for i in range(20):
            _write(src / cat / "train" / "good" / f"{i:03d}.png", make(rng))
        for dtype in ["good", "scratch"]:
            for i in range(4):
                _write(src / cat / "test" / dtype / f"{i:03d}.png", make(rng))
                if dtype != "good":
                    m = np.zeros((400, 400), np.uint8)
                    m[180:220, 180:220] = 255
                    _write(src / cat / "ground_truth" / dtype / f"{i:03d}_mask.png", m)

    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    cfg.update(synthetic_per_category=12, min_val_good=2)
    cfg["categories"] = {k: cfg["categories"][k] for k in makers}
    cfg_path = base / "data.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    dst = base / "processed"
    subprocess.run([sys.executable, "scripts/prepare_data.py", "--src", str(src), "--dst", str(dst),
                    "--config", str(cfg_path)], cwd=ROOT, check=True)
    return dst, cfg, src, cfg_path


@pytest.fixture(scope="session")
def edge(built, tmp_path_factory):
    out = tmp_path_factory.mktemp("edge") / "edge"
    subprocess.run([sys.executable, "scripts/build_edge.py", "--data", str(built[0]), "--out", str(out),
                    "--ratios", "0.5", "0.05", "--save-all-max-ratio", "0.1", "--calib-size", "8",
                    "--backbone", "resnet18", "--no-pretrained"], cwd=ROOT, check=True)
    return out
